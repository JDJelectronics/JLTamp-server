"""The engine: keeps the library and its vectors in sync, and answers prompts.

Startup order matters. We need the embedding server's dimension before we can
open the vector store, because the dimension is what tells us whether the
vectors on disk belong to the model now running.
"""
from __future__ import annotations

import json
import os
import random
import re
import threading
import time

import numpy as np

from . import config, genre_infer, scoring
from . import redenen as reden_opslag
from . import genre_groepen
from . import intent_llm
from .embed_client import EmbedClient, EmbedError
from .embed_store import DimensionMismatch, EmbeddingStore
from .jltamp_client import JLTampClient, Library, Track

# De app toont de voorbeeldzinnen in de taal van de gebruiker (en, de, es, pt, fy)
# en stuurt ze zo naar de AI. Alleen Nederlandse woorden herkennen betekende dat
# "wake me up without a fright" of "Spannungsbogen: hoch und wieder runter" een
# gewone lijst werd in plaats van een opbouw of een boog. Daarom staan de sleutel-
# woorden uit elke vertaling van ai.suggestions hier ook in.
BEST_OF_TRIGGERS = ("leukste", "beste", "best of", "songs van", "playlist van", "top van",
                    "das beste von", "lo mejor de", "o melhor de", "o melhor dos",
                    "it moaiste fan", "greatest hits van")
DISCOVERY_TRIGGERS = ("vergelijkbaar", "similar", "discovery", "radio", "lijkt op",
                      "zoals", "in de stijl van", "familiair",
                      "like ", "sounds like", "wie ", "ähnlich", "como ", "parecido",
                      "tipo ", "parecido com", "lykas ")
# A wind-down is an *ordering* intent, not just "calm music": the tempo has to
# descend. So these are explicit descent / sleep cues — plain "rustige muziek"
# is deliberately left out so it still gets a normal calm playlist, not a ramp.
WIND_DOWN_TRIGGERS = ("afbouw", "afbouwen", "wind down", "wind-down", "winddown",
                      "in slaap", "inslapen", "slapen", "om te slapen",
                      "steeds rustiger", "aflopend tempo", "tempo omlaag",
                      "tot rust komen", "cool down", "cooldown",
                      "afkoelen", "naar rust",
                      "fall asleep", "falling asleep", "to sleep",
                      "runterfahren", "einschlafen", "zum schlafen",
                      "bajar el ritmo", "quedarme dormido", "para dormir",
                      "diminuir o ritmo", "até eu dormir", "abrandar", "adormecer",
                      "ôfbouwe", "yn 'e sliep")
# The wind-down's siblings: same measured tempo, a different shape. Each of
# these is an *ordering* intent, so they are matched before the artist
# strategies for the same reason wind-down is.
BUILD_UP_TRIGGERS = ("opbouw", "opbouwen", "wakker worden", "warm draaien",
                     "warmdraaien", "warming-up", "warming up", "opstarten",
                     "steeds sneller", "tempo omhoog", "op gang komen",
                     # Niet kaal "wake me up" of "despertar": dat zijn ook
                     # songtitels (Avicii), en "wake me up van avicii" hoort
                     # geen opbouw-lijst te worden.
                     "wake me up without", "wake up without", "waking up",
                     "wake up slowly", "wake up gently", "warm up",
                     "wach werden", "aufwachen", "despertarme",
                     "acordar", "wekker wurde", "opbouwe")
# Deliberately compound words. A bare "boog" would fire on "regenboog", and a
# bare "piek" on any lyric about one.
PARTY_ARC_TRIGGERS = ("feestboog", "energieboog", "opbouw en afbouw",
                      "eerst opbouwen dan afbouwen", "naar een piek",
                      "piek en dan rustig",
                      "party arc", "build up, then", "build up and then",
                      "spannungsbogen", "hoch und wieder runter",
                      "subir y luego", "subir e depois", "feestbôge")
INTERVAL_TRIGGERS = ("interval", "intervaltraining", "hiit", "tempoblokken",
                     "blokken van", "series:", "treino:")
ARTIST_SPLITTERS = ("zoals ", "lijkt op ", "van ", "voor ", "artiest ", "artist ",
                    "bij ", "naar ",
                    "like ", "best of ", "wie ", "von ", "como ", "mejor de ",
                    "tipo os ", "tipo ", "melhor dos ", "melhor de ", "lykas ", "fan ")

# How long a validated token, and the taste signals fetched with it, stay
# cached. Long enough that a burst of requests costs one round trip to JLTamp;
# short enough that revoking a token takes effect within minutes.
USER_CACHE_SEC = 300


class Engine:
    def __init__(self):
        self.client = JLTampClient()
        self.library = Library(self.client)
        self.embedder = EmbedClient()
        self.store: EmbeddingStore | None = None
        self.features: dict = {}
        self.ready = False
        self.status = "starting"
        self.last_error = ""
        self._embed_lock = threading.Lock()
        # token -> {client, user, likes, skips, at}
        self._users: dict[str, dict] = {}
        self._users_lock = threading.Lock()
        # Maintained by the embed worker so /health stays O(1).
        self._stale_count = 0
        # {rating_key: genre} guessed from embedding neighbours for untagged
        # tracks. Loaded from disk, recomputed by the embed worker once the
        # vectors it needs exist. Overlaid onto a snapshot, never persisted
        # into JLTamp — a guess must not masquerade as the user's own tag.
        self.inferred_genres: dict[str, str] = genre_infer.load(
            config.INFERRED_GENRES_FILE) if config.INFER_GENRES else {}
        self._inferred_at = 0    # vector count the overlay was last built for
        self._features_mtime = 0.0
        # Tracks in the library that actually carry a measured tempo. Counted
        # after each refresh so /health stays O(1).
        self._measured = 0
        # {lowercase artist name: name as written}, rebuilt when the library is.
        self._artists: dict[str, str] = {}
        self._artists_at = 0.0

    # ── startup ──────────────────────────────────────────────────────────────
    def start(self) -> None:
        threading.Thread(target=self._boot, daemon=True).start()

    def _boot(self) -> None:
        self.features = self._load_features()

        # Log in with the same patience as the library load below. At a cold
        # boot — a power cut brings the Jetson and JLTamp up together — the
        # server often isn't answering yet. Giving up on the first failure left
        # the engine permanently dead; and because _boot runs in a daemon
        # thread, the process kept serving a broken /health, so systemd's
        # Restart never fired either. Retry, then exit so systemd resurrects us.
        for attempt in range(1, 13):
            try:
                self.client.login()
                break
            except Exception as e:                  # noqa: BLE001
                self.status = "jltamp-auth-failed"
                self.last_error = str(e)
                wait = min(30, attempt * 5)
                print(f"⚠️  JLTamp login failed (attempt {attempt}): {e} — "
                      f"retrying in {wait}s")
                self.client.refresh_connection()    # drop any half-dead socket
                time.sleep(wait)
        else:
            print("❌ Could not log in to JLTamp after repeated attempts — "
                  "exiting so systemd restarts us.")
            os._exit(1)

        # Retry the initial load: a transient JLTamp hiccup at boot (a restart,
        # a moment of load) used to leave the engine permanently dead until
        # someone noticed and restarted it. Back off and keep trying.
        for attempt in range(1, 13):
            try:
                count = self.library.refresh(self.features)
                self._apply_inferred()
                self._count_measured()
                print(f"✅ JLTamp connected: {count} tracks.")
                break
            except Exception as e:                  # noqa: BLE001
                self.status = "jltamp-unreachable"
                self.last_error = str(e)
                wait = min(30, attempt * 5)
                print(f"⚠️  Library load failed (attempt {attempt}): {e} — "
                      f"retrying in {wait}s")
                self.client.refresh_connection()    # drop any half-dead socket
                time.sleep(wait)
        else:
            print("❌ Could not load the library after repeated attempts — "
                  "exiting so systemd restarts us.")
            os._exit(1)

        print(f"🔍 Waiting for the embedding server at {config.EMBED_URL} ...")
        if not self.embedder.probe(attempts=40, delay=3.0):
            self.status = "embedder-offline"
            self.last_error = f"no embedding server at {config.EMBED_URL}"
            print("⚠️  Embedding server did not answer — running in fallback mode.")
            return

        try:
            self.store = EmbeddingStore(config.DATA_DIR, self.embedder.dim,
                                        self.embedder.model_id)
        except DimensionMismatch as e:
            self.status = "vector-store-mismatch"
            self.last_error = str(e)
            print(f"❌ {e}")
            return

        self.ready = True
        self.status = "ok"
        print(f"✅ AI ready: {self.embedder.dim} dims, model "
              f"'{self.embedder.model_id or 'unknown'}', "
              f"{len(self.store)} vectors stored.")

        threading.Thread(target=self._embed_worker, daemon=True).start()
        threading.Thread(target=self._refresh_worker, daemon=True).start()

    def _load_features(self) -> dict:
        """Audio features keyed by track ratingKey, if the analyser has run."""
        path = config.FEATURES_FILE
        if not path.exists():
            return {}
        try:
            self._features_mtime = path.stat().st_mtime
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"⚠️  Could not read {path}: {e}")
            return {}
        print(f"🎚️  Audio features loaded for {len(data)} tracks.")
        return data

    def _count_measured(self) -> None:
        """How many tracks in the library actually have a tempo.

        Not the same as the size of the features file, which also holds
        measurements for tracks that have since left the library: on a real
        library that file held far more entries than there were tracks that
        could be put on a tempo curve, and reporting the bigger number hid a
        third of the gap.
        """
        self._measured = sum(1 for t in self.library.snapshot()
                             if t.features.get("bpm"))

    def _reload_features_if_changed(self) -> None:
        """Pick up a fresh analyser run without a restart.

        The features file used to be read once at boot, so every newly
        measured track — the analyser only ever measures the ones it has not
        seen — stayed invisible until someone restarted the service. Tracks
        without a tempo cannot sit on a wind-down curve and earn no audio
        boost, so that silently held the newest music out of exactly the
        prompts these measurements exist for.
        """
        path = config.FEATURES_FILE
        try:
            mtime = path.stat().st_mtime
        except OSError:
            return
        if mtime <= self._features_mtime:
            return
        fresh = self._load_features()
        if fresh:
            self.features = fresh

    # ── background work ──────────────────────────────────────────────────────
    def _embed_worker(self) -> None:
        """Fill in vectors for tracks that don't have one yet."""
        # A chunk the embedding server cannot handle is skipped, which leaves
        # those tracks stale — so the next pass finds the same work, fails the
        # same way, and comes round again five seconds later, forever. Back off
        # when a pass achieves nothing, so one unembeddable title costs a log
        # line rather than a permanent hammering of llama.cpp.
        idle_backoff = 5
        while True:
            if not self.ready or self.store is None:
                time.sleep(10)
                continue
            tracks = self.library.snapshot()
            # Stale, not just missing: when the way we phrase a track changes
            # (a compilation's real artist, a cleaned-up title), its stored
            # vector no longer matches its text and has to be rebuilt.
            by_key = {t.rating_key: t for t in tracks}
            stale = set(self.store.stale({t.rating_key: t.text for t in tracks}))
            todo = [by_key[k] for k in stale if k in by_key]
            self._stale_count = len(todo)
            if not todo:
                # Embeddings are complete and stable — the only point at which
                # rebuilding the genre overlay is worthwhile. Doing it after
                # every batch (as this once did) reran a 100 s neighbour search
                # dozens of times per fill and starved the HTTP server.
                self._recompute_inferred()
                time.sleep(300)
                continue

            print(f"🧠 Embedding {len(todo)} tracks (new or changed) ...")
            done = 0
            with self._embed_lock:
                for offset, vecs in self.embedder.embed_batched([t.text for t in todo]):
                    chunk = todo[offset:offset + len(vecs)]
                    self.store.add_many(
                        {t.rating_key: np.asarray(v, dtype=np.float32)
                         for t, v in zip(chunk, vecs)},
                        texts={t.rating_key: t.text for t in chunk},
                    )
                    done += len(vecs)
                    self._stale_count = max(0, len(todo) - done)
                    # Persist as we go: an interrupted run keeps its progress.
                    if done % (config.EMBED_BATCH * 8) == 0:
                        self.store.save()
                self.store.save()
            if done:
                print(f"✅ {done} embeddings added ({len(self.store)} total).")
                idle_backoff = 5
            else:
                idle_backoff = min(300, idle_backoff * 3)
                print(f"⚠️  {len(todo)} tracks could not be embedded — retrying "
                      f"in {idle_backoff}s.")
            time.sleep(idle_backoff)

    def _refresh_worker(self) -> None:
        while True:
            time.sleep(config.LIBRARY_REFRESH_SEC)
            try:
                # Before the refresh: refresh() is what copies features onto
                # the fresh Track objects, so a reload afterwards would sit
                # unused for another hour.
                self._reload_features_if_changed()
                count = self.library.refresh(self.features)
                self._apply_inferred()
                self._count_measured()
                print(f"🔄 Library refreshed: {count} tracks.")
            except Exception as e:                  # noqa: BLE001
                print(f"⚠️  Library refresh failed: {e}")
                continue
            try:
                self._nieuw_check()
            except Exception as e:                  # noqa: BLE001
                print(f"⚠️  Nieuw voor jou mislukt: {e}")

    # ── inferred genres ──────────────────────────────────────────────────────
    def _apply_inferred(self) -> None:
        """Overlay the inferred genre onto untagged tracks in the snapshot.

        Only fills a placeholder — a real tag always wins. Runs after every
        library refresh, since refresh builds fresh Track objects that do not
        carry the overlay.
        """
        if not config.INFER_GENRES or not self.inferred_genres:
            return
        n = 0
        for t in self.library.snapshot():
            if genre_infer.is_placeholder(t.genre):
                g = self.inferred_genres.get(t.rating_key)
                if g:
                    t.inferred_genre = g    # separate field — never enters text
                    n += 1
        if n:
            print(f"🏷️  Applied inferred genre to {n} untagged tracks.")

    def _recompute_inferred(self) -> None:
        """Rebuild the inferred-genre overlay once the vectors it needs exist.

        Only when the vector count has grown since the last build: the
        neighbour search is ~100 s, and there is nothing to gain from repeating
        it against an unchanged store.
        """
        if not config.INFER_GENRES or self.store is None:
            return
        if len(self.store) <= self._inferred_at:
            return
        try:
            tracks = self.library.snapshot()
            inferred = {k: g for k, (g, c)
                        in genre_infer.infer_with_confidence(tracks, self.store).items()
                        if c >= config.INFER_THRESHOLD}
        except Exception as e:                       # noqa: BLE001
            print(f"⚠️  Genre inference failed: {e}")
            return
        self.inferred_genres = inferred
        self._inferred_at = len(self.store)
        genre_infer.save(inferred, config.INFERRED_GENRES_FILE)
        self._apply_inferred()
        print(f"🏷️  Genre inference: {len(inferred)} untagged tracks labelled.")

    # ── per-user context ─────────────────────────────────────────────────────
    def user_context(self, token: str) -> dict | None:
        """Resolve a caller's JLTamp token into a client and their own taste
        signals. Returns None when JLTamp rejects the token.

        Playlists must be created as the person who asked — otherwise everyone
        else's requests land in whichever account happens to be in .env — and
        their likes and skips are what should shape their playlist, not the
        service account's.
        """
        if not token:
            return None
        now = time.time()
        with self._users_lock:
            ctx = self._users.get(token)
            if ctx and now - ctx["at"] < USER_CACHE_SEC:
                return ctx

        client = JLTampClient(token=token)
        user = client.me()
        if not user:
            return None
        try:
            likes = client.liked_ids()
            skips = client.skip_counts()
        except Exception:                           # noqa: BLE001
            likes, skips = set(), {}
        libraries = client.library_keys()           # None = onbekend, niet filteren

        ctx = {"client": client, "user": user, "likes": likes,
               "skips": skips, "libraries": libraries, "at": now}
        with self._users_lock:
            self._users[token] = ctx
            # Bound the cache: one entry per active token, oldest dropped first.
            if len(self._users) > 50:
                oldest = min(self._users, key=lambda k: self._users[k]["at"])
                self._users.pop(oldest, None)
        return ctx

    @staticmethod
    def _visible(tracks: list[Track], libraries: set[str] | None) -> list[Track]:
        """Alleen de nummers uit bibliotheken die deze gebruiker mag zien.

        `libraries` None betekent: onbekend (geen token, of het ophalen mislukte)
        — dan niet filteren, zoals het altijd was. Een lege set betekent echt
        geen toegang, en dan is er ook niets.
        """
        if libraries is None:
            return tracks
        return [t for t in tracks if t.library_id in libraries]

    @staticmethod
    def _signals(ctx: dict | None) -> scoring.Signals:
        """The calling user's own likes and skips, to score with.

        This used to write those values onto the shared Track objects, which
        is a data race: every request scores the same objects, one thread per
        job. Whoever asked last decided what "liked" meant for everyone else,
        including callers with no token at all. Now nothing is mutated — the
        history is looked up per request instead.
        """
        if not ctx:
            return scoring.LIBRARY_SIGNALS
        return scoring.Signals(ctx["likes"], ctx["skips"])

    # ── prompt handling ──────────────────────────────────────────────────────
    def handle(self, prompt: str, token: str = "") -> dict:
        """Route a prompt to the right strategy and build the playlist.

        `token` is the caller's own JLTamp session token. Without one we fall
        back to the service account, which is only right for single-user use.
        """
        prompt = (prompt or "").strip()
        if not prompt:
            return {"status": "error", "message": "lege prompt"}

        low = prompt.lower()
        tracks = self.library.snapshot()
        if not tracks:
            # Reading a whole library and backfilling genres takes minutes, so
            # this is the normal state right after a restart — not a failure.
            # Say which it is, because "no tracks" reads like the library is
            # empty when it is merely still loading.
            if self.status in ("starting", "ok"):
                return {"status": "error",
                        "message": "de bibliotheek wordt nog ingelezen — "
                                   "probeer het over een minuut opnieuw"}
            return {"status": "error",
                    "message": f"geen verbinding met JLTamp ({self.status})"}

        ctx = self.user_context(token) if token else None
        if ctx:
            tracks = self._visible(tracks, ctx.get("libraries"))
            if not tracks:
                return {"status": "error",
                        "message": "je hebt nog geen toegang tot een muziekbibliotheek"}
        signals = self._signals(ctx)
        client = ctx["client"] if ctx else self.client

        if not self.ready:
            return self._fallback(tracks, prompt, client)

        # Tempo curves are checked before artist strategies: "afbouw playlist
        # met Coldplay" is a tempo ramp flavoured by an artist, not a best-of.
        # Most specific first — an interval session says "interval" and often
        # "opbouwen" too, and it is the interval that was asked for.
        if any(w in low for w in INTERVAL_TRIGGERS):
            return self.interval(prompt, tracks, client, signals)
        if any(w in low for w in PARTY_ARC_TRIGGERS):
            return self.energy_arc(prompt, tracks, client, signals, "peak")
        if any(w in low for w in BUILD_UP_TRIGGERS):
            return self.energy_arc(prompt, tracks, client, signals, "up")
        if any(w in low for w in WIND_DOWN_TRIGGERS):
            return self.wind_down(prompt, tracks, client, signals)

        artist = self._find_artist(low, tracks)
        if artist and any(w in low for w in BEST_OF_TRIGGERS):
            return self._best_of(artist, tracks, client, signals)
        if artist and any(w in low for w in DISCOVERY_TRIGGERS):
            return self._discovery(artist, tracks, client, signals)
        return self._semantic(prompt, tracks, client, signals)

    def foryou(self, token: str = "") -> dict:
        """Een persoonlijke mix uit het smaakprofiel — geen prompt nodig.

        Pakt de dichtstbijzijnde tracks bij de smaak-centroïde die de gebruiker
        NOG NIET kent, gespreid over artiesten. Publiceert als 'Voor jou ✨'.
        """
        tracks = self.library.snapshot()
        if not tracks or not self.ready or self.store is None:
            return {"status": "error",
                    "message": "de engine is nog niet gereed — probeer het zo opnieuw"}
        ctx = self.user_context(token) if token else None
        client = ctx["client"] if ctx else self.client
        if ctx:
            tracks = self._visible(tracks, ctx.get("libraries"))
            if not tracks:
                return {"status": "error",
                        "message": "je hebt nog geen toegang tot een muziekbibliotheek"}

        # De favorieten als INDIVIDUELE zaad-vectoren (niet één gemiddelde):
        # meest-gespeeld eerst, dan likes — die volgorde is de rangschikking.
        favourites: list[str] = []
        seen_fav: set[str] = set()
        for key in client.most_played_ids(80) + list(client.liked_ids()):
            if key not in seen_fav:
                seen_fav.add(key)
                favourites.append(key)
        if len(favourites) < config.MIN_TASTE_SEED:
            return {"status": "error",
                    "message": "nog te weinig luistergeschiedenis voor een persoonlijke mix — "
                               "luister eerst wat, dan leer ik je smaak"}
        seed_mat, known_list = self.store.matrix(favourites)
        if len(known_list) < config.MIN_TASTE_SEED:
            return {"status": "error",
                    "message": "nog te weinig luistergeschiedenis voor een persoonlijke mix — "
                               "luister eerst wat, dan leer ik je smaak"}
        known = set(known_list)
        skips = ctx["skips"] if ctx else {}
        by_key = {t.rating_key: t for t in tracks}
        # Wat je kent, als liedje: anders komt de verzamelalbum-kopie van je
        # favoriet terug als 'iets nieuws voor jou'.
        known_songs = {by_key[k].song_key for k in known if k in by_key}
        cap = config.SCORING["MAX_TRACKS"]
        # Elke kandidaat scoort op zijn DICHTSTBIJZIJNDE favoriet, niet op het
        # gemiddelde — zo blijft alles vlak bij muziek die je echt draait.
        band = self.store.nearest_to_any(seed_mat, [t.rating_key for t in tracks],
                                         cap + len(known) + 400)
        picks: list[Track] = []
        seen_artist: dict[str, int] = {}
        for key, _sim in band:
            if key in known:
                continue
            if skips.get(key, 0) >= 3:          # herhaald weggeklikt → overslaan
                continue
            t = by_key.get(key)
            if t is None or any(k in t.haystack for k in scoring.KIDS_WORDS):
                continue
            if t.song_key in known_songs:       # kent hij al, of staat er al in
                continue
            a = t.real_artist
            if seen_artist.get(a, 0) >= 2:      # spreiding over artiesten
                continue
            picks.append(t)
            known_songs.add(t.song_key)
            seen_artist[a] = seen_artist.get(a, 0) + 1
            if len(picks) >= cap:
                break
        if not picks:
            return {"status": "error", "message": "geen persoonlijke aanbevelingen gevonden"}
        dichtst = self._dichtstbij(picks, known_list)
        redenen = {t.rating_key: self._reden_favoriet(t, by_key.get(dichtst.get(t.rating_key, "")))
                   for t in picks}
        return self._publish("Voor jou ✨", scoring.dj_volgorde(picks), client, redenen)

    # ── 🕒 voor nu: muziek bij dit moment van de dag ─────────────────────────
    _DAGEN = ("maandag", "dinsdag", "woensdag", "donderdag", "vrijdag", "zaterdag", "zondag")

    @staticmethod
    def _dagdeel(uur: int) -> str:
        if 5 <= uur < 12:
            return "ochtend"
        if 12 <= uur < 18:
            return "middag"
        if 18 <= uur < 24:
            return "avond"
        return "nacht"

    def moment(self, token: str = "", uur: int | None = None, dag: int | None = None,
               tz_minuten: int = 0) -> dict:
        """🕒 Voor nu — een mix die past bij dit moment van de dag en de week.

        Kijkt in je eigen luistergeschiedenis naar wat je op dit soort momenten
        draait: rond hetzelfde uur (twee uur ervoor of erna telt, minder naarmate
        het verder weg is), en zwaarder als het toen óók doordeweeks of weekend
        was zoals nu. Die nummers zijn de basis; de helft van de mix, de rest zijn
        buren ervan die je nog niet draaide. Het tempo blijft in de buurt van wat
        je op dit moment meestal draait — maandagochtend is niet vrijdagavond.

        De app stuurt uur, weekdag (0 = maandag) en tijdzone van het toestel mee:
        de server weet niet waar je bent, en de tijdstempels zijn UTC.
        `tz_minuten` is wat JavaScript getTimezoneOffset() geeft (UTC min lokaal).
        """
        tracks = self.library.snapshot()
        if not tracks or not self.ready or self.store is None:
            return {"status": "error",
                    "message": "de engine is nog niet gereed — probeer het zo opnieuw"}
        ctx = self.user_context(token) if token else None
        client = ctx["client"] if ctx else self.client
        if ctx:
            tracks = self._visible(tracks, ctx.get("libraries"))
            if not tracks:
                return {"status": "error",
                        "message": "je hebt nog geen toegang tot een muziekbibliotheek"}

        nu = time.gmtime(time.time() - tz_minuten * 60)
        uur = nu.tm_hour if uur is None else int(uur) % 24
        dag = nu.tm_wday if dag is None else int(dag) % 7
        weekend = dag >= 5
        dagdeel = self._dagdeel(uur)
        label = f"{self._DAGEN[dag]}{dagdeel}"
        by_key = {t.rating_key: t for t in tracks}

        def kinder(t: Track) -> bool:
            return any(kw in t.haystack for kw in scoring.KIDS_WORDS)

        # Wat draai je op dit soort momenten?
        gewicht: dict[str, float] = {}
        for ev in client.history_events(days=120):
            rk = str(ev.get("ratingKey") or "")
            ts = ev.get("startedAt")
            if not rk or rk not in by_key or not ts:
                continue
            if not (ev.get("completed") or int(ev.get("percent") or 0) >= 50
                    or int(ev.get("listenedMs") or 0) >= 60_000):
                continue
            toen = time.gmtime(int(ts) - tz_minuten * 60)
            afstand = abs(toen.tm_hour - uur)
            afstand = min(afstand, 24 - afstand)
            if afstand > 2:
                continue
            factor = 1.0 - 0.3 * afstand
            if (toen.tm_wday >= 5) != weekend:
                factor *= 0.4
            gewicht[rk] = gewicht.get(rk, 0.0) + factor

        zaden = scoring.unique_songs(
            [by_key[k] for k in sorted(gewicht, key=lambda k: -gewicht[k])
             if not kinder(by_key[k])])[:40]
        eigen_moment = len(zaden) >= 8
        if not eigen_moment:
            # Te weinig geschiedenis op dit uur: dan je favorieten als basis.
            zaden = scoring.unique_songs(
                [by_key[k] for k in client.most_played_ids(60)
                 if k in by_key and not kinder(by_key[k])])[:40]
        if len(zaden) < 5:
            return {"status": "error",
                    "message": "nog te weinig luistergeschiedenis voor een mix bij dit "
                               "moment — luister eerst wat, dan leer ik je ritme"}

        # Het tempo dat je dan draait (mediaan), met een band eromheen. Zonder
        # gemeten tempo een redelijke gok per dagdeel.
        bpms = sorted(t.features["bpm"] for t in zaden if t.features and t.features.get("bpm"))
        doel_bpm = (bpms[len(bpms) // 2] if bpms else
                    {"ochtend": 105, "middag": 118, "avond": 112, "nacht": 85}[dagdeel])

        seed_mat, seed_keys = self.store.matrix([t.rating_key for t in zaden])
        if not seed_keys:
            return {"status": "error",
                    "message": "nog geen embeddings voor je muziek van dit moment"}
        skips = ctx["skips"] if ctx else {}
        cap = config.SCORING["MAX_TRACKS"]
        zaad_set = set(seed_keys)
        buren: list[Track] = []
        for key, _sim in self.store.nearest_to_any(seed_mat, list(by_key),
                                                   cap * 8 + len(zaad_set)):
            if key in zaad_set or skips.get(key, 0) >= 3:
                continue
            t = by_key.get(key)
            if t is None or kinder(t):
                continue
            bpm = t.features.get("bpm") if t.features else None
            if bpm and abs(bpm - doel_bpm) > 18:
                continue
            buren.append(t)

        rng = random.Random()
        bekend = self._greep(zaden, cap // 2, 3, rng)
        nieuw = self._greep(
            scoring.unique_songs(buren, known={t.song_key for t in zaden})[:cap * 3],
            cap - len(bekend), 3, rng)
        # Bekend en nieuw door elkaar, in een volgorde die loopt als een set.
        picks = scoring.dj_volgorde(bekend + nieuw)
        if len(picks) < 10:
            return {"status": "error", "message": "niets gevonden dat bij dit moment past"}

        dichtst = self._dichtstbij(nieuw, seed_keys)
        redenen = {t.rating_key: (f"Draai je vaak op {label}" if eigen_moment
                                  else "Een van je favorieten")
                   for t in bekend}
        for t in nieuw:
            redenen[t.rating_key] = self._reden_favoriet(
                t, by_key.get(dichtst.get(t.rating_key, "")))
        res = self._publish("🕒 Voor nu", picks, client, redenen)
        if res.get("status") == "success":
            res["message"] = (f"Mix voor {label} gemaakt: {res.get('tracks', len(picks))} "
                              f"nummers rond {int(round(doel_bpm))} BPM.")
        return res

    # ── 👥 samen: een mix die tussen iedereen in ligt ─────────────────────────
    def _client_voor(self, uid: int) -> JLTampClient | None:
        """Een client die handelt als deze gebruiker, via het admin-account.

        Het token blijft een uur bewaard: anders komt er bij elke mix voor elke
        deelnemer een nieuwe sessie in de database bij."""
        cache = self.__dict__.setdefault("_sessie_tokens", {})
        hit = cache.get(uid)
        if hit and time.time() - hit[1] < 3600:
            return JLTampClient(token=hit[0])
        token = self.client.session_for(uid)
        if not token:
            return None
        cache[uid] = (token, time.time())
        return JLTampClient(token=token)

    def samen(self, token: str, code: str) -> dict:
        """👥 Samen — een mix voor iedereen in een Listen Together-sessie.

        Niet het gemiddelde van ieders smaak: dat valt samen met wie het meest
        luistert. Elk nummer telt voor de deelnemer die het het MINST zou
        waarderen (plus een beetje het gemiddelde), dus wat bovenaan komt ligt
        dicht bij iedereen tegelijk.

        Alleen de host mag dit, en alleen voor mensen die echt in zijn sessie
        zitten. De server geeft de deelnemers van een sessie aan iedereen die
        is ingelogd, dus die controle staat hier. Alleen muziek uit bibliotheken
        die ALLE deelnemers mogen afspelen, en de redenen noemen namen, nooit
        nummers uit iemands geschiedenis.
        """
        if not self.ready or self.store is None:
            return {"status": "error",
                    "message": "de engine is nog niet gereed — probeer het zo opnieuw"}
        ctx = self.user_context(token) if token else None
        if not ctx:
            return {"status": "error", "message": "log eerst in om een mix voor iedereen te maken"}
        mij = ctx["user"].get("id")
        try:
            info = ctx["client"]._get(f"/session/{(code or '').strip().upper()}")
        except Exception:                                   # noqa: BLE001
            return {"status": "error", "message": "die sessie bestaat niet (meer)"}
        deelnemers = info.get("participants") or []
        if not any(p.get("userId") == mij and p.get("isHost") for p in deelnemers):
            return {"status": "error",
                    "message": "alleen de host van de sessie kan een mix voor iedereen maken"}
        mensen: dict[int, str] = {}
        for p in deelnemers:
            uid = p.get("userId")
            if uid is not None and uid not in mensen:
                mensen[uid] = (p.get("name") or "iemand").split("@")[0]
        if len(mensen) < 2:
            return {"status": "error",
                    "message": "er luistert nog niemand mee — nodig eerst iemand uit"}

        tracks = self.library.snapshot()
        by_key = {t.rating_key: t for t in tracks}
        cap = config.SCORING["MAX_TRACKS"]
        smaken: list[tuple[str, np.ndarray]] = []
        toegang: set[str] | None = None
        for uid, naam in mensen.items():
            client = ctx["client"] if uid == mij else self._client_voor(uid)
            if client is None:
                continue
            try:
                ranked, _heard, _info = self._weekly_favourites(client)
                libs = client.library_keys()
            except Exception as e:                          # noqa: BLE001
                print(f"⚠️  samen: smaak van gebruiker {uid} niet op te halen: {e}")
                continue
            if libs is not None:
                toegang = set(libs) if toegang is None else toegang & set(libs)
            seeds, present = self.store.matrix([k for k in ranked if k in by_key][:40])
            if len(present) >= 5:
                smaken.append((naam, seeds))
        if len(smaken) < 2:
            return {"status": "error",
                    "message": "van te weinig mensen in de sessie is genoeg luistergeschiedenis "
                               "om een gezamenlijke smaak te bepalen"}

        zicht = (by_key if toegang is None else
                 {k: t for k, t in by_key.items() if t.library_id in toegang})
        if not zicht:
            return {"status": "error",
                    "message": "jullie hebben geen muziekbibliotheek die iedereen mag afspelen"}
        keys = list(zicht)
        pool: list[str] = []
        gezien: set[str] = set()
        for _naam, seeds in smaken:
            for key, _sim in self.store.nearest_to_any(seeds, keys, cap * 6):
                if key not in gezien:
                    gezien.add(key)
                    pool.append(key)
        mat, have = self.store.matrix(pool)
        if not have:
            return {"status": "error", "message": "nog geen embeddings voor jullie muziek"}
        per_persoon = np.stack([(mat @ seeds.T).max(axis=1) for _naam, seeds in smaken])
        eerlijk = per_persoon.min(axis=0) + 0.2 * per_persoon.mean(axis=0)
        gerangschikt: list[Track] = []
        for i in np.argsort(-eerlijk):
            t = zicht.get(have[int(i)])
            if t is None or any(kw in t.haystack for kw in scoring.KIDS_WORDS):
                continue
            gerangschikt.append(t)
        gerangschikt = scoring.unique_songs(gerangschikt)[:cap * 3]
        picks = scoring.dj_volgorde(self._greep(gerangschikt, cap, 2, random.Random()))
        if len(picks) < 10:
            return {"status": "error", "message": "niets gevonden dat bij iedereen past"}

        namen = [naam for naam, _seeds in smaken]
        wie = " en ".join(namen) if len(namen) <= 2 else ", ".join(namen[:-1]) + " en " + namen[-1]
        redenen = {t.rating_key: self._met_kenmerken(f"Past bij {wie}", t) for t in picks}
        res = self._publish("👥 Samen", picks, ctx["client"], redenen)
        if res.get("status") == "success":
            res["message"] = (f"Mix voor {wie} gemaakt: {res.get('tracks', len(picks))} "
                              f"nummers die bij iedereen passen.")
        return res

    def _artist_index(self, tracks: list[Track]) -> dict[str, str]:
        """{lowercase artist name: the name as written}.

        Built once per library refresh rather than per prompt: the old scan
        lowercased tens of thousands of artist fields on every single request. Compilation
        placeholders are left out — "various artists" is not an artist — and
        the performer credited on a compilation is included, so "de beste van
        Doe Maar" finds them even though the album says otherwise.
        """
        stamp = self.library.loaded_at
        if self._artists and self._artists_at == stamp:
            return self._artists
        index: dict[str, str] = {}
        for t in tracks:
            for name in (t.artist, t.orig_artist):
                clean = (name or "").strip()
                low = clean.lower()
                # names_artist() ignores anything shorter than 4 characters, so
                # there is nothing to gain from indexing it.
                if len(clean) > 3 and low not in Track._COMPILATION:
                    index.setdefault(low, clean)
        self._artists = index
        self._artists_at = stamp
        return index

    def _find_artist(self, prompt: str, tracks: list[Track]) -> str | None:
        """The artist a prompt is about, if any.

        Tries the phrase after a connector first ("zoals Adele"), then any
        known name the prompt genuinely names, longest first so "Bruce
        Springsteen" beats a stray "Bruce".

        The candidate is confirmed by scoring.names_artist rather than a plain
        substring test. That function exists because a bare substring cannot
        tell "instrumentale focus muziek" from a request for the band Focus,
        and routing on the loose test sent such prompts down the best-of path,
        where the whole playlist becomes one wrongly-guessed artist — a much
        louder failure than the misplaced boost it was written to prevent.
        """
        index = self._artist_index(tracks)
        for splitter in ARTIST_SPLITTERS:
            if splitter in prompt:
                tail = prompt.split(splitter)[-1].strip()
                tail = re.sub(r"[^\w\s].*$", "", tail).strip()
                if tail in index:
                    return index[tail]
        # Substring first (cheap over tens of thousands of names), then the strict check on the
        # handful that survive.
        hits = sorted((n for n in index if n in prompt), key=len, reverse=True)
        for name in hits:
            if scoring.names_artist(prompt, index[name]):
                return index[name]
        return None

    def _artist_vector(self, artist: str, tracks: list[Track]) -> np.ndarray | None:
        """An artist's centroid: the average of their tracks' vectors.

        Matched on the performer, not the album's grouping artist: two thirds
        of this library sits on compilations credited to "Various Artists", and
        those tracks are exactly the ones a grouping-artist match cannot find.
        """
        keys = [t.rating_key for t in tracks
                if t.real_artist.lower() == artist.lower()]
        mat, present = self.store.matrix(keys)
        if not present:
            return None
        return mat.mean(axis=0)

    @staticmethod
    def _zaad_namen(artiest_zaden: list[str], seed_keys: list[str], by_key: dict) -> list[str]:
        """De artiesten van de zaadjes, belangrijkste eerst (hooguit drie)."""
        tel: dict[str, int] = {}
        namen: dict[str, str] = {}
        for naam in artiest_zaden:
            if naam:
                tel[naam.lower()] = tel.get(naam.lower(), 0) + 1000
                namen[naam.lower()] = naam
        for k in seed_keys:
            t = by_key.get(k)
            naam = (t.real_artist or "").strip() if t else ""
            if naam and naam.lower() not in Track._COMPILATION:
                tel[naam.lower()] = tel.get(naam.lower(), 0) + 1
                namen.setdefault(naam.lower(), naam)
        return [namen[k] for k in sorted(tel, key=lambda k: -tel[k])[:3]]

    def _naamgenoten_trekken(self, target: np.ndarray, artiest_zaden: list[str],
                             seed_keys: list[str], by_key: dict, tracks: list) -> bool:
        """Trekt de naam de zender naar artiesten die alleen zo HETEN?

        Kijkt naar de 60 dichtstbijzijnde nummers van andere artiesten: heeft
        minstens 15% daarvan een naam die op een zaad-artiest lijkt (difflib ≥
        0.6, of de ene naam begint met de eerste drie letters van de andere), dan
        ja. Joost: 17 van de 30 (JoJo, Joop, Jo En Stef). Adele: 0 — Sade lijkt
        wel een beetje, maar dat is een enkele, en die hoort er ook bij.
        """
        import difflib
        plat = lambda s: re.sub(r"[^a-z0-9]", "", (s or "").lower())
        namen = [plat(n) for n in self._zaad_namen(artiest_zaden, seed_keys, by_key)]
        namen = [n for n in namen if len(n) >= 3]
        if not namen:
            return False
        band = self.store.top_keys(target, [t.rating_key for t in tracks], 400)
        gezien, lijkt = 0, 0
        for key, _sim in band:
            t = by_key.get(key)
            if t is None:
                continue
            a = plat(t.real_artist)
            if not a or any(a == n or n in a for n in namen):
                continue                          # de artiest zelf (ook met gasten)
            gezien += 1
            if any(difflib.SequenceMatcher(None, a, n).ratio() >= 0.6 or a[:3] == n[:3]
                   for n in namen):
                lijkt += 1
            if gezien >= 60:
                break
        return gezien > 0 and lijkt / gezien >= 0.15

    def _zonder_namen(self, target: np.ndarray, artiest_zaden: list[str],
                      seed_keys: list[str], by_key: dict) -> np.ndarray:
        """Het radio-doel zonder de richting van de artiestnamen van de zaadjes.

        Namen: de artiesten waarop de zender start, en bij een zender vanaf
        nummers (album, playlist) de drie die daar het vaakst in voorkomen.
        Ingebed zoals de bibliotheek zelf (zonder zoekvoorvoegsel), orthonormaal
        gemaakt en dan uit het doel geprojecteerd. Lukt het inbedden niet, dan
        blijft het doel zoals het was.
        """
        kies = self._zaad_namen(artiest_zaden, seed_keys, by_key)
        if not kies:
            return target
        try:
            with self._embed_lock:
                vecs = self.embedder.embed(kies)
        except Exception as e:                      # noqa: BLE001
            print(f"⚠️  radio: namen niet ingebed ({e}) — zender zonder correctie")
            return target
        basis: list[np.ndarray] = []
        for v in vecs:
            u = np.asarray(v, dtype=np.float32)
            for b in basis:
                u = u - float(u @ b) * b
            n = float(np.linalg.norm(u))
            if n > 1e-6:
                basis.append(u / n)
        uit = np.asarray(target, dtype=np.float32)
        for b in basis:
            uit = uit - float(uit @ b) * b
        return uit

    def _seed_vector(self, s: str, by_key: dict, tracks: list):
        """Los één radio-zaadje op naar (vector, label, track_key|None)."""
        if s in by_key:
            mat, present = self.store.matrix([s])
            if present:
                st = by_key[s]
                return mat[0], f"{st.real_artist} - {st.clean_title}", s
        v = self._artist_vector(s, tracks)
        return (v, s, None) if v is not None else (None, s, None)

    def radio(self, seed, exclude: set[str], count: int = 20, dislike=None,
              token: str = "", energie: int = 0) -> dict:
        """The next batch of an endless 'more like this' station.

        `seed` is a track key ("t123") or an artist name. `exclude` is what has
        already played or is queued, so repeated calls advance through the
        nearest tracks rather than returning the same ones. Stateless on
        purpose: the app owns the growing exclude set, which keeps it naturally
        per-user and lets the station run forever without the engine tracking
        sessions.

        Stays close to the seed — this is a station, not a journey — so it is
        pure nearest-neighbour with only a tie-break jitter, no drift.
        """
        if not self.ready or self.store is None:
            return {"status": "error", "message": "engine niet gereed"}
        tracks = self.library.snapshot()
        # Alleen wat de luisteraar zelf mag afspelen; een radio die nummers uit
        # een bibliotheek zonder toegang voorstelt, speelt stilte.
        ctx = self.user_context(token) if token else None
        if ctx:
            tracks = self._visible(tracks, ctx.get("libraries"))
        by_key = {t.rating_key: t for t in tracks}

        # Een zender op genre ("genre:rock") is een eigen soort radio.
        for s in (seed if isinstance(seed, (list, tuple)) else [seed]):
            if isinstance(s, str) and s.strip().lower().startswith("genre:"):
                return self._genre_radio(s.strip()[6:].strip().lower(), exclude, count,
                                         dislike, ctx, by_key, energie)

        # Multi-seed: elk zaadje (track-key of artiest) -> vector, gemiddelde =
        # het middelpunt van de "meng deze" -radio.
        seeds = seed if isinstance(seed, (list, tuple)) else [seed]
        # Een radio vanaf een hele playlist ("pl12") of een album ("al34"): de
        # nummers van die lijst zijn de zaadjes. Alle nummers van de lijst gaan
        # hieronder in `skip`, zodat de zender NIEUWE nummers speelt die erop
        # lijken — de lijst zelf heb je al.
        lijst_naam = ""
        lijst_alle: set[str] = set()
        uitgebreid: list = []
        for s in seeds:
            if isinstance(s, str) and re.fullmatch(r"(?:pl|al)\d+", s.strip()):
                lijst_client = ctx["client"] if ctx else self.client
                keys, naam = lijst_client.lijst_nummers(s.strip())
                keys = [k for k in keys if k in by_key]
                lijst_alle.update(keys)
                if len(keys) > 30:
                    stap = len(keys) / 30
                    keys = [keys[int(i * stap)] for i in range(30)]
                uitgebreid.extend(keys)
                lijst_naam = lijst_naam or (f"Radio: {naam}" if naam else "Radio: deze lijst")
            else:
                uitgebreid.append(s)
        seeds = uitgebreid
        seed_vecs, labels, seed_keys = [], [], []
        artiest_zaden: list[str] = []      # zaadjes die een artiestnaam zijn
        for s in seeds:
            s = str(s).strip()
            if not s:
                continue
            v, lbl, k = self._seed_vector(s, by_key, tracks)
            if v is not None:
                seed_vecs.append(v)
                labels.append(lbl)
                if k:
                    seed_keys.append(k)
                else:
                    artiest_zaden.append(lbl.strip().lower())
        if not seed_vecs:
            return {"status": "error",
                    "message": f"kon geen radio starten vanaf '{seed}'"}
        target = np.mean(np.stack(seed_vecs), axis=0)

        # "Minder zoals dit": duw het middelpunt weg van niet-leuke tracks.
        dislike = set(str(x) for x in (dislike or []))
        dis_vecs = []
        for d in dislike:
            if d in by_key:
                m, p = self.store.matrix([d])
                if p:
                    dis_vecs.append(m[0])
        if dis_vecs:
            target = target - config.RADIO_DISLIKE_WEIGHT * np.mean(np.stack(dis_vecs), axis=0)

        # De naam eruit. Elke vector is gemaakt van "artiest - titel - genre -
        # album", dus bij een korte, eigen naam trekt die naam de zender naar
        # artiesten met een naam die erop LIJKT: de radio op Joost speelde JoJo,
        # Joop, Jo En Stef en Dj Jozz. Haal de richting van de naam(en) uit het
        # doel; wat overblijft is wat de nummers verder gemeen hebben.
        # Alleen als het nodig is: bij Adele draagt de naam juist nuttige smaak
        # (zonder correctie Dido en Sade, met correctie The Who en Black Eyed Peas).
        if config.RADIO_ZONDER_NAAM and self._naamgenoten_trekken(
                target, artiest_zaden, seed_keys, by_key, tracks):
            target = self._zonder_namen(target, artiest_zaden, seed_keys, by_key)

        label = " + ".join(labels) if len(labels) > 1 else labels[0]
        if lijst_naam:
            label = lijst_naam
        skip = set(exclude) | set(seed_keys) | dislike | lijst_alle
        picks: list[Track] = []
        # Pull a wide band, then keep the nearest that are not excluded. The
        # band grows the deeper the station runs (exclude gets large), so ask
        # for generously more than needed.
        band = self.store.top_keys(target, [t.rating_key for t in tracks],
                                   # Ruim: met ontdubbelen op liedje (alle live-
                                   # en remasterversies tellen als één) en max 3
                                   # per artiest raakte een kring van ~220 bij een
                                   # artiest als Queen leeg — 5 nummers in plaats
                                   # van 20. argpartition maakt 2000 goedkoop.
                                   count + len(skip) + 2000)
        seen_artist: dict[str, int] = {}
        # Ook op liedje, niet alleen op id: dezelfde song van een verzamelalbum
        # is voor wie luistert gewoon een herhaling van wat net speelde.
        skip_songs = {by_key[k].song_key for k in skip if k in by_key}

        # Naamgenoten. De embedding leest een artiestnaam ook als gewoon woord,
        # dus een radio vanaf "Queen" trok "Rocket Queen" (Guns N' Roses) en
        # "Queen of the Highway" (The Doors) aan, en die vanaf Adele
        # "Adele Bloemendaal – Vingerlied". Weg met een andere artiest die de
        # naam alleen bevat, en met een titel waar hij in staat — maar niet de
        # artiest zelf, ook niet met gasten ("Queen and David Bowie").
        def naamgenoot(t: Track) -> bool:
            eigen = (t.orig_artist or t.real_artist or "").strip().lower()
            for naam in artiest_zaden:
                woord = rf"\b{re.escape(naam)}\b"
                if eigen == naam or re.match(
                        rf"{re.escape(naam)}\s*(?:,|&|\+|\band\b|\ben\b|\bx\b|\bfeat\.?|\bft\.?|\bmet\b)",
                        eigen):
                    return False
                if re.search(woord, eigen) or re.search(woord, t.clean_title.lower()):
                    return True
            return False

        for key, _sim in band:
            if key in skip:
                continue
            t = by_key.get(key)
            if t is None or any(k in t.haystack for k in scoring.KIDS_WORDS):
                continue
            if t.song_key in skip_songs or (artiest_zaden and naamgenoot(t)):
                continue
            # Loose per-artist cap so a station does not become one artist on
            # repeat, but still stays firmly in the seed's neighbourhood.
            a = t.real_artist
            if seen_artist.get(a, 0) >= 3:
                continue
            picks.append(t)
            skip_songs.add(t.song_key)
            seen_artist[a] = seen_artist.get(a, 0) + 1
            # Wie om rustiger of energieker vraagt, krijgt een ruimere kring om
            # uit te kiezen: van de dichtstbijzijnde drie keer zoveel, en daarvan
            # de helft die de goede kant op ligt.
            if len(picks) >= count * (3 if energie else 1):
                break
        if energie:
            picks = scoring.stuur_energie(picks, energie)[:count]

        # Als een DJ-set in plaats van geschud: elk nummer sluit in tempo en
        # energie aan op het vorige.
        picks = scoring.dj_volgorde(picks)
        return {
            "status": "ok",
            "seed": label,
            "tracks": [{"ratingKey": t.rating_key,
                        "title": t.clean_title,
                        "artist": t.real_artist} for t in picks],
        }

    # ── 📻 radio op genre ────────────────────────────────────────────────────
    def _genre_leden(self, tracks: list[Track]) -> dict[str, list[str]]:
        """{groep: [ratingKey, …]} over de HELE bibliotheek. Eén keer per
        verversing; wat een gebruiker mag zien wordt er per aanvraag uit gefilterd.
        Zonder echt genre telt het afgeleide (inferred_genre)."""
        stamp = self.library.loaded_at
        cache = self.__dict__.get("_genre_cache")
        if cache and cache[0] == stamp:
            return cache[1]
        leden: dict[str, list[str]] = {}
        for t in tracks:
            for groep in genre_groepen.groepen_van(t.genre or t.inferred_genre):
                leden.setdefault(groep, []).append(t.rating_key)
        self.__dict__["_genre_cache"] = (stamp, leden)
        return leden

    def genres(self, token: str = "") -> dict:
        """De genres om een radio op te starten: de opgeschoonde groepen, met hoeveel
        nummers je ervan mag afspelen en hoeveel van jouw favorieten erin zitten.
        Jouw genres bovenaan — wie vooral hip-hop draait, zoekt niet eerst naar Pop."""
        tracks = self.library.snapshot()
        if not tracks:
            return {"status": "error", "message": "de bibliotheek wordt nog ingelezen"}
        leden = self._genre_leden(tracks)
        ctx = self.user_context(token) if token else None
        client = ctx["client"] if ctx else self.client
        libs = ctx.get("libraries") if ctx else None
        zicht = {t.rating_key for t in tracks if t.library_id in libs} if libs is not None else None
        try:
            eigen = set(client.most_played_ids(200)) | set(client.liked_ids())
        except Exception:                                   # noqa: BLE001
            eigen = set()
        uit = []
        for gid, (naam, _tags) in genre_groepen.GROEPEN.items():
            keys = leden.get(gid, [])
            if zicht is not None:
                keys = [k for k in keys if k in zicht]
            if len(keys) < 30:
                continue
            uit.append({"id": gid, "label": naam, "aantal": len(keys),
                        "jij": len(eigen.intersection(keys))})
        uit.sort(key=lambda g: (-g["jij"], -g["aantal"]))
        return {"status": "ok", "genres": uit}

    def _genre_radio(self, groep: str, exclude: set[str], count: int, dislike,
                     ctx: dict | None, by_key: dict[str, Track], energie: int = 0) -> dict:
        """Een eindeloze zender binnen één genre, gericht op jouw kant ervan.

        Alleen nummers uit dat genre. Het middelpunt ligt tussen het hart van het
        genre (een vaste steekproef, zodat de zender niet bij elke aanvulling
        verspringt) en je eigen favorieten binnen dat genre: rock voor iemand
        die Queen draait klinkt anders dan voor iemand die Metallica draait.
        Weggeklikte nummers duwen het middelpunt weg, net als bij gewone radio.
        Elke reeks is een wisselende greep uit de dichtstbijzijnde, hooguit twee
        per artiest, zonder dubbele liedjes, in DJ-volgorde.
        """
        if groep not in genre_groepen.GROEPEN:
            return {"status": "error", "message": f"onbekend genre '{groep}'"}
        naam = genre_groepen.GROEPEN[groep][0]
        leden = self._genre_leden(self.library.snapshot())
        pool = [k for k in leden.get(groep, []) if k in by_key]
        if len(pool) < 10:
            return {"status": "error",
                    "message": f"te weinig {naam.lower()} in je bibliotheek voor een radio"}
        poolset = set(pool)

        steekproef = random.Random(groep).sample(pool, min(400, len(pool)))
        mat, have = self.store.matrix(steekproef)
        if not have:
            return {"status": "error", "message": f"nog geen embeddings voor {naam.lower()}"}
        midden = mat.mean(axis=0)

        client = ctx["client"] if ctx else self.client
        try:
            eigen = [k for k in client.most_played_ids(200) + list(client.liked_ids())
                     if k in poolset]
        except Exception:                                   # noqa: BLE001
            eigen = []
        target = midden
        persoonlijk = False
        if len(eigen) >= 5:
            emat, epresent = self.store.matrix(list(dict.fromkeys(eigen))[:60])
            if epresent:
                target = 0.6 * emat.mean(axis=0) + 0.4 * midden
                persoonlijk = True

        dislike = {str(x) for x in (dislike or [])}
        dmat, dpresent = self.store.matrix([k for k in dislike if k in by_key])
        if dpresent:
            target = target - config.RADIO_DISLIKE_WEIGHT * dmat.mean(axis=0)

        skip = set(exclude) | dislike
        skip_songs = {by_key[k].song_key for k in skip if k in by_key}
        kandidaten: list[Track] = []
        for key, _sim in self.store.top_keys(target, pool, count * 6 + len(skip) + 200):
            if key in skip:
                continue
            t = by_key.get(key)
            if t is None:
                continue
            if groep != "kids" and any(kw in t.haystack for kw in scoring.KIDS_WORDS):
                continue
            kandidaten.append(t)
        kandidaten = scoring.unique_songs(kandidaten, known=skip_songs)[:count * 3]
        if energie:
            kandidaten = scoring.stuur_energie(kandidaten, energie)
        picks = scoring.dj_volgorde(self._greep(kandidaten, count, 2, random.Random()))
        return {
            "status": "ok",
            "seed": f"{naam}-radio" + (" in jouw smaak" if persoonlijk else ""),
            "tracks": [{"ratingKey": t.rating_key, "title": t.clean_title,
                        "artist": t.real_artist} for t in picks],
        }

    def _best_of(self, artist: str, tracks: list[Track], client,
                 signals: scoring.Signals | None = None) -> dict:
        signals = signals or scoring.LIBRARY_SIGNALS
        own = [t for t in tracks if t.real_artist.lower() == artist.lower()]
        if not own:
            return {"status": "error", "message": f"'{artist}' niet gevonden"}
        own.sort(key=lambda t: (signals.liked(t), t.play_count, -signals.skips(t)),
                 reverse=True)
        redenen = {t.rating_key: ("Staat bij je favorieten ♥" if signals.liked(t)
                                  else self._met_kenmerken(f"Van {t.album or artist}", t))
                   for t in own[:40]}
        return self._publish(f"⭐ Best of {artist}", own[:40], client, redenen)

    def _discovery(self, artist: str, tracks: list[Track], client,
                   signals: scoring.Signals | None = None) -> dict:
        signals = signals or scoring.LIBRARY_SIGNALS
        target = self._artist_vector(artist, tracks)
        if target is None:
            return {"status": "error",
                    "message": f"nog geen embeddings voor '{artist}'"}
        # Zelfde correctie als bij de radio: trekt de naam naar artiesten die
        # alleen zo heten ("Dancing Queen.mp3", Q65), haal hem dan uit het doel.
        by_key = {t.rating_key: t for t in tracks}
        if config.RADIO_ZONDER_NAAM and self._naamgenoten_trekken(target, [artist], [], by_key, tracks):
            target = self._zonder_namen(target, [artist], [], by_key)
        naam = artist.strip().lower()

        def zelf(t: Track) -> bool:
            # De artiest zelf, ook met gasten: "Queen and David Bowie" is Queen.
            eigen = t.real_artist.strip().lower()
            return eigen == naam or re.match(
                rf"{re.escape(naam)}\s*(?:,|&|\+|\band\b|\ben\b|\bx\b|\bfeat\.?|\bft\.?|\bmet\b|\bwith\b)",
                eigen) is not None

        sims = scoring.similarity_map(self.store, target, tracks)
        rng = random.Random()
        scored = []
        seen: dict[str, int] = {}
        for t in tracks:
            sim = sims.get(t.rating_key)
            if sim is None:
                continue
            if any(k in t.haystack for k in scoring.KIDS_WORDS):
                continue
            # Ontdekken betekent ANDERE artiesten. Dit was een straf van 0.25, en
            # een ongespeeld nummer kreeg er 0.3 bij — dus kwam de artiest zelf
            # er gewoon in ("like Queen, but not Queen" gaf Queen and David Bowie).
            if zelf(t):
                continue
            # Naamgenoten: een andere artiest die de naam als woord bevat. De
            # correctie hierboven helpt alleen als ze massaal komen; één
            # "Dancing Queen.mp3" (een bestandsnaam als artiest) glipte er zo nog
            # door. Q65 bevat het woord niet en blijft, dat is een echte band.
            if re.search(rf"\b{re.escape(naam)}\b", t.real_artist.lower()):
                continue
            score = sim - seen.get(t.real_artist, 0) * 0.15 + rng.uniform(0, 0.05)
            if t.play_count == 0:
                score += 0.3
            skips = signals.skips(t)
            if skips:
                score -= min(skips, 4) * 0.2
            scored.append((t, score))
            if sim > 0.6:
                seen[t.real_artist] = seen.get(t.real_artist, 0) + 1

        scored.sort(key=lambda p: p[1], reverse=True)
        picked = [t for t, _ in scored[:config.SCORING["MAX_TRACKS"]]]
        redenen = {t.rating_key: self._met_kenmerken(f"Klinkt als {artist}", t) for t in picked}
        return self._publish(f"🕵️ Lijkt op {artist}", picked, client, redenen)

    def _semantic(self, prompt: str, tracks: list[Track], client,
                  signals: scoring.Signals | None = None) -> dict:
        try:
            # Short prompts get expanded before embedding; scoring still sees
            # the original, so keyword and artist matching stay honest.
            expanded = scoring.expand_query(prompt)
            extra = intent_llm.enrich_query(prompt)   # sfeer-woorden of ""
            if extra:
                expanded = f"{expanded} {extra}"
            query = self.embedder.embed_one(expanded)
            # Personaliseer élke aanbeveling: meng het smaakprofiel van de
            # gebruiker subtiel in de queryvector (Spotify-stijl). De prompt
            # blijft leidend; store.search normaliseert de blend. Bij te weinig
            # luistergeschiedenis geeft _taste_vector None terug -> geen blend.
            if config.TASTE_BLEND > 0:
                tv = self._taste_vector(client)
                if tv is not None:
                    query = query + config.TASTE_BLEND * tv[0]
        except (EmbedError, Exception) as e:        # noqa: BLE001
            print(f"⚠️  Prompt embedding failed: {e}")
            return self._fallback(tracks, prompt, client)

        sims = scoring.similarity_map(self.store, query, tracks)
        if not sims:
            return {"status": "error",
                    "message": "nog geen embeddings — de engine is nog aan het indexeren"}

        scored = scoring.score_tracks(prompt, tracks, sims, signals=signals)
        low = prompt.lower()
        relaxed = bool(scoring.active_contexts(low) or scoring.extract_year(low))
        picked = scoring.select(scored, relaxed=relaxed)
        if not picked:
            return {"status": "error", "message": "niets gevonden dat hierbij past"}
        kort = prompt if len(prompt) <= 40 else prompt[:37].rstrip() + "…"
        redenen = {t.rating_key: self._met_kenmerken(f"Past bij “{kort}”", t) for t in picked}
        return self._publish(scoring.playlist_name(low), scoring.dj_volgorde(picked),
                             client, redenen)

    # ── tempo curves: wind-down, build-up, party arc, intervals ──────────────
    def _tempo_pool(self, prompt: str, tracks: list[Track], client,
                    signals: scoring.Signals | None = None) -> list[Track]:
        """Candidate tracks for any tempo curve: the caller's own favourites,
        plus whatever the prompt asks for, so 'rustige jazz om te slapen' stays
        jazzy while a bare 'afbouw playlist' still draws on *their* music.

        Kids' tracks are dropped — a wind-down is not a nursery playlist.
        """
        by_key = {t.rating_key: t for t in tracks}
        pool: dict[str, Track] = {}

        # Prompt-driven half — honour any genre / mood the user typed.
        prompt = (prompt or "").strip()
        if prompt:
            try:
                query = self.embedder.embed_one(scoring.expand_query(prompt))
                sims = scoring.similarity_map(self.store, query, tracks)
                for t, _ in scoring.score_tracks(prompt, tracks, sims,
                                                 signals=signals)[:250]:
                    pool[t.rating_key] = t
            except Exception as e:                  # noqa: BLE001
                print(f"⚠️  wind-down semantic pool failed: {e}")

        # Personal half — the caller's own favourites.
        try:
            for key in client.most_played_ids(80) + list(client.liked_ids()):
                t = by_key.get(key)
                if t:
                    pool[t.rating_key] = t
        except Exception as e:                      # noqa: BLE001
            print(f"⚠️  wind-down favourites failed: {e}")

        return [t for t in pool.values()
                if not any(k in t.haystack for k in scoring.KIDS_WORDS)]

    def wind_down(self, prompt: str, tracks: list[Track], client,
                  signals: scoring.Signals | None = None) -> dict:
        """A playlist whose tempo glides down to something you could sleep to.

        Built from the caller's own taste (and whatever the prompt asks for),
        so it is personal and lands in their own account. Only tracks with a
        measured tempo can sit on a tempo curve, so those are all it uses.
        """
        graded = self._graded_pool(prompt, tracks, client, signals)
        if len(graded) < 12:
            return {"status": "error",
                    "message": "te weinig nummers met een gemeten tempo voor "
                               "een afbouw-playlist"}
        curve = scoring.descending_tempo_curve(
            graded, config.SCORING["MAX_TRACKS"])
        return self._publish("🌙 Afbouw — rustig naar het einde", curve, client,
                             self._tempo_redenen(curve, "Het tempo zakt"))

    def _graded_pool(self, prompt: str, tracks: list[Track], client,
                     signals: scoring.Signals | None
                     ) -> list[tuple[Track, float]]:
        """The tempo pool as (track, bpm) pairs. Only measured tracks: putting
        a track with no tempo on a tempo curve is ordering it at random."""
        pool = self._tempo_pool(prompt, tracks, client, signals)
        # Hooguit drie per artiest. "afbouwen tot ik in slaap val" haalt anders
        # vooral één slaapmuziek-kanaal binnen: 19 van de 43 nummers waren van
        # Sleep Fruits Music (414 nummers in de bibliotheek). De pool staat op
        # volgorde van hoe goed hij past, dus de beste van elke artiest blijven.
        per: dict[str, int] = {}
        uit: list[tuple[Track, float]] = []
        for t in pool:
            if not t.features.get("bpm"):
                continue
            a = t.real_artist.strip().lower()
            if a not in Track._COMPILATION:
                if per.get(a, 0) >= 3:
                    continue
                per[a] = per.get(a, 0) + 1
            uit.append((t, t.features["bpm"]))
        return uit

    def energy_arc(self, prompt: str, tracks: list[Track], client,
                   signals: scoring.Signals | None, shape: str) -> dict:
        """A playlist that climbs ("up"), or climbs and comes back ("peak").

        The wind-down's siblings: same measured tempo, a different shape.
        """
        graded = self._graded_pool(prompt, tracks, client, signals)
        if len(graded) < 12:
            return {"status": "error",
                    "message": "te weinig nummers met een gemeten tempo voor "
                               "een tempo-playlist"}
        curve = scoring.tempo_arc(graded, config.SCORING["MAX_TRACKS"], shape)
        if not curve:
            return {"status": "error", "message": "kon geen tempocurve maken"}
        name = ("🌅 Opbouw — op gang komen" if shape == "up"
                else "🎉 Energieboog — opbouwen en weer afbouwen")
        return self._publish(name, curve, client,
                             self._tempo_redenen(curve, "Past op dit punt in de boog"))

    @staticmethod
    def _tempo_redenen(tracks: list[Track], basis: str) -> dict[str, str]:
        """Bij een tempo-curve is het tempo zelf de reden."""
        uit = {}
        for t in tracks:
            bpm = t.features.get("bpm") if t.features else None
            uit[t.rating_key] = f"{basis} · {int(round(bpm))} BPM" if bpm else basis
        return uit

    def interval(self, prompt: str, tracks: list[Track], client,
                 signals: scoring.Signals | None) -> dict:
        """An interval session: fast and calm blocks that follow the clock.

        Blocks land on track boundaries, because a playlist cannot cut a song
        in half. The message says what they actually became — someone running
        to this needs to know the 2-minute recovery is really 3.
        """
        total, hard, easy = scoring.interval_spec(prompt)
        graded = self._graded_pool(prompt, tracks, client, signals)
        if len(graded) < 12:
            return {"status": "error",
                    "message": "te weinig nummers met een gemeten tempo voor "
                               "een intervaltraining"}
        plan = scoring.interval_blocks(graded, total, hard, easy)
        if len(plan) < 4:
            return {"status": "error",
                    "message": "te weinig passende nummers voor een "
                               "intervaltraining"}

        picks = [t for t, _ in plan]
        seconds = sum(t.duration_ms for t in picks) / 1000.0
        blocks = sum(1 for i, (_, phase) in enumerate(plan)
                     if i == 0 or plan[i - 1][1] != phase)
        name = f"🏃 Intervaltraining {total // 60} min"
        redenen = {}
        for t, phase in plan:
            blok = "Hard blok" if str(phase).lower().startswith(("hard", "fast", "snel", "high")) \
                else "Rustig blok"
            bpm = t.features.get("bpm") if t.features else None
            redenen[t.rating_key] = f"{blok} · {int(round(bpm))} BPM" if bpm else blok
        res = self._publish(name, picks, client, redenen)
        if res.get("status") == "success":
            res["message"] = (
                f"Playlist '{name}' aangemaakt: {len(picks)} nummers, "
                f"{seconds / 60:.0f} min in {blocks} blokken van "
                f"{hard // 60} min hard en {easy // 60} min rustig. "
                f"Blokken lopen tot het einde van een nummer, dus ze zijn "
                f"iets langer dan gevraagd.")
        return res

    def _fallback(self, tracks: list[Track], prompt: str, client) -> dict:
        """No AI available — still give the user music rather than an error."""
        sample = random.sample(tracks, min(50, len(tracks)))
        try:
            client.create_playlist("🎲 Random Mix", sample)
        except Exception as e:                      # noqa: BLE001
            return {"status": "error", "message": f"AI offline en playlist mislukt: {e}"}
        return {
            "status": "fallback",
            "playlist": "🎲 Random Mix",
            "message": "AI is even offline — hier is een willekeurige mix.",
            "tracks": len(sample),
        }

    # ── waarom dit nummer? ───────────────────────────────────────────────────
    @staticmethod
    def _kenmerken(t: Track) -> list[str]:
        """Wat er MEETBAAR over een nummer te zeggen valt: genre, decennium,
        tempo. Alleen echte gegevens — een reden die niet klopt is erger dan
        geen reden."""
        uit: list[str] = []
        genre = (t.genre or t.inferred_genre or "").split(",")[0].strip()
        if genre and genre.lower() not in ("unknown", "other", "overig", "misc"):
            uit.append(genre.lower())
        if t.year and 1950 <= int(t.year) <= 2099:
            uit.append(f"jaren {str(int(t.year) // 10 * 10)[2:]}")
        bpm = t.features.get("bpm") if t.features else None
        if bpm:
            uit.append("rustig" if bpm < 90 else ("snel" if bpm > 125 else "midtempo"))
        return uit

    def _met_kenmerken(self, basis: str, t: Track) -> str:
        k = self._kenmerken(t)
        return basis + (" · " + ", ".join(k) if k else "")

    @staticmethod
    def _korte_titel(t: Track, max_len: int = 40) -> str:
        """De titel zoals je hem zou uitspreken: zonder alles tussen haakjes
        ("(feat. Wyclef Jean) (En Español)") en niet langer dan één regel."""
        titel = re.sub(r"\s*[\(\[][^\)\]]*[\)\]]", "", t.clean_title).strip() or t.clean_title
        return titel if len(titel) <= max_len else titel[:max_len - 1].rstrip() + "…"

    def _reden_favoriet(self, t: Track, fav: Track | None) -> str:
        """'Omdat je vaak X – Y draait', met wat ze gemeen hebben."""
        if fav is None:
            return self._met_kenmerken("Past bij wat je graag draait", t)
        eigen = self._kenmerken(fav)
        gedeeld = [k for k in self._kenmerken(t) if k in eigen]
        # De artiest van het nummer, niet die van de map: anders stond er
        # "Omdat je vaak Foute Uur Top 1500 – Left Outside Alone draait".
        artiest = fav.orig_artist or fav.real_artist
        basis = f"Omdat je vaak {artiest} – {self._korte_titel(fav)} draait"
        return basis + (f" · ook {', '.join(gedeeld)}" if gedeeld else "")

    def _dichtstbij(self, picks: list[Track], seed_keys: list[str]) -> dict[str, str]:
        """Per gekozen nummer: welke van de zaad-nummers er het dichtst bij ligt.
        De vectoren zijn genormaliseerd, dus één matmul geeft alle cosines."""
        if self.store is None or not picks or not seed_keys:
            return {}
        seeds, present = self.store.matrix(seed_keys)
        mat, have = self.store.matrix([t.rating_key for t in picks])
        if not present or not have:
            return {}
        best = (mat @ seeds.T).argmax(axis=1)
        return {k: present[int(i)] for k, i in zip(have, best)}

    def _publish(self, name: str, tracks: list[Track], client,
                 redenen: dict[str, str] | None = None) -> dict:
        # Elk liedje één keer, in ELKE lijst die de AI maakt. Alleen de
        # prompt-route ontdubbelde (in scoring.select); Voor jou, best-of,
        # lijkt-op en de tempo-curves zetten hetzelfde nummer van album,
        # verzamelalbum en single gewoon twee of drie keer neer. Hier komt
        # alles langs, dus hier hoort het.
        tracks = scoring.unique_songs(tracks)
        if not tracks:
            return {"status": "error", "message": "geen resultaten"}
        try:
            playlist_key = client.create_playlist(name, tracks)
        except Exception as e:                      # noqa: BLE001
            print(f"❌ Playlist creation failed: {e}")
            return {"status": "error", "message": f"playlist aanmaken mislukt: {e}"}
        print(f"🆕 Playlist '{name}' — {len(tracks)} tracks.")
        if redenen:
            keys = {t.rating_key for t in tracks}
            reden_opslag.bewaar(playlist_key,
                                {k: v for k, v in redenen.items() if k in keys})
        # Say so when the library could not fill the request. A short playlist
        # is a fine answer; presenting it as a full one is not, and the user
        # otherwise has no way to tell "that's all there is" from "the search
        # went wrong".
        wanted = config.SCORING["MAX_TRACKS"]
        message = f"Playlist '{name}' aangemaakt."
        if len(tracks) < wanted * 0.6:
            message = (f"Playlist '{name}' aangemaakt met {len(tracks)} nummers — "
                       f"meer passends staat er niet in je bibliotheek.")
        # `playlist` is the field aiService.ts reads for the name it shows;
        # `playlist_key` is the Plex ratingKey so the app can OPEN and play the
        # freshly-made playlist instead of leaving it buried in the library.
        return {
            "status": "success",
            "playlist": name,
            "playlist_key": playlist_key,
            "message": message,
            "tracks": len(tracks),
        }

    # ── weekly per-user playlists ────────────────────────────────────────────
    def _taste_vector(self, client) -> tuple[np.ndarray, set[str]] | None:
        """A user's taste centroid, plus the set of tracks they already know.

        Built from their liked + most-played tracks. Returns None when there is
        too little to go on: a "personal" playlist for someone who has barely
        listened is a guess dressed up as a recommendation. Someone who has
        played only a handful of tracks gets nothing rather than noise — better
        no weekly playlist than a wrong one.
        """
        # Order matters: most-played first, then likes. The DNA mix is these
        # tracks themselves — the user's actual favourites — so the order is
        # the ranking. Deduplicated but kept in that order.
        favourites: list[str] = []
        seen: set[str] = set()
        for key in client.most_played_ids(80) + list(client.liked_ids()):
            if key not in seen:
                seen.add(key)
                favourites.append(key)
        if len(seen) < config.MIN_TASTE_SEED:
            return None
        mat, present = self.store.matrix(favourites)
        if len(present) < config.MIN_TASTE_SEED:
            return None
        centroid = mat.mean(axis=0)
        return centroid, present    # present is in favourites order

    WEEKLY_DNA = "🧬 Jouw DNA Mix"
    WEEKLY_DISCO = "🔮 Ontdekking van de Week"

    def _weekly_favourites(self, client) -> tuple[list[str], set[str], dict]:
        """Iemands favorieten op volgorde, plus alles wat hij al kent.

        Drie bronnen, elk met een eigen gewicht:
          • meest gespeeld over alle tijd (tot 200) — wie je bent;
          • wat je de laatste acht weken echt hoorde — wat je NU draait, en het
            deel waardoor de DNA-mix van week tot week meeschuift. Alleen op
            'alle tijd' was het elke week dezelfde lijst;
          • likes — wat je zelf aanwees.
        `heard` is de optelsom: voor Ontdekking, waar niets in mag wat je kent.
        """
        most = client.most_played_ids(200)
        recent = client.recent_play_counts(56)
        liked = client.liked_ids()
        score: dict[str, float] = {}
        for i, key in enumerate(most):
            score[key] = score.get(key, 0.0) + 2.0 * (1 - i / max(1, len(most)))
        for key, n in recent.items():
            score[key] = score.get(key, 0.0) + 0.6 * min(n, 10)
        for key in liked:
            score[key] = score.get(key, 0.0) + 1.5
        ranked = sorted(score, key=lambda k: score[k], reverse=True)
        # De losse bronnen gaan mee, voor de reden onder elk nummer.
        info = {"recent": recent, "liked": liked,
                "most": {k: i for i, k in enumerate(most)}}
        return ranked, set(most) | set(recent) | set(liked), info

    @staticmethod
    def _reden_dna(t: Track, info: dict) -> str:
        key = t.rating_key
        n = info["recent"].get(key, 0)
        if n >= 2:
            return f"Je draaide dit de laatste weken {n} keer"
        if key in info["liked"]:
            return "Staat bij je favorieten ♥"
        rang = info["most"].get(key)
        if rang is not None and rang < 50:
            return "Een van je meest gedraaide nummers ooit"
        return "Hoort bij wat je vaak draait"

    @staticmethod
    def _greep(ranked: list[Track], cap: int, per_artist: int,
               rng: random.Random) -> list[Track]:
        """Een wisselende greep uit een gerangschikte lijst.

        Gewogen trekken zonder teruglegging: bovenaan is de kans het grootst,
        maar niet zeker. Zo lijkt de lijst van deze week op die van vorige week
        zonder dezelfde te zijn — anders heeft wekelijks verversen geen zin.
        Hooguit `per_artist` per artiest; raakt de lijst daarmee niet vol, dan
        wordt hij aangevuld uit wat die grens tegenhield (wie maar één artiest
        draait, krijgt toch een volle mix). Het resultaat staat op rangorde.
        """
        order = {id(t): i for i, t in enumerate(ranked)}
        pool = list(ranked)
        weights = [1.0 / (1.0 + i * 0.05) for i in range(len(pool))]
        picks: list[Track] = []
        held: list[Track] = []
        count: dict[str, int] = {}
        while pool and len(picks) < cap:
            j = rng.choices(range(len(pool)), weights=weights)[0]
            t = pool.pop(j)
            weights.pop(j)
            artist = t.real_artist.strip().lower()
            if artist not in Track._COMPILATION and count.get(artist, 0) >= per_artist:
                held.append(t)
                continue
            count[artist] = count.get(artist, 0) + 1
            picks.append(t)
        for t in held:
            if len(picks) >= cap:
                break
            picks.append(t)
        picks.sort(key=lambda t: order[id(t)])
        return picks

    def generate_weekly(self, only_user_id: int | None = None) -> list[dict]:
        """🧬 Jouw DNA Mix en 🔮 Ontdekking van de Week, voor elke gebruiker.

        Runs as the admin service, minting a per-user token so each playlist is
        created in that user's own library from that user's own taste — never
        shared, never in the wrong account. Draait elke maandag (cron of launchd, zie
        scripts/weekly_playlists.sh), en is zo gebouwd dat een week later
        de lijsten echt anders zijn:

        * DNA Mix — je favorieten, met recent luisteren erin, als gewogen greep
          in plaats van elke week dezelfde top 50.
        * Ontdekking — dicht bij wat je draait maar nog niet gehoord, en nooit
          wat vorige week al in Ontdekking stond.
        * Iedereen krijgt een Ontdekking. Te weinig eigen geschiedenis: dan uit
          de smaak van iedereen die er wel een heeft. Een DNA-mix zonder eigen
          favorieten zou verzonnen zijn; die wacht tot er genoeg is.
        * Elk liedje één keer — ook album-, verzamelalbum- en remaster-kopieën.
        """
        if not self.ready or self.store is None:
            return [{"status": "error", "message": "engine not ready"}]
        tracks = self.library.snapshot()
        by_key = {t.rating_key: t for t in tracks}
        cap = config.SCORING["MAX_TRACKS"]
        rng = random.Random()
        results: list[dict] = []
        visible_for: dict[frozenset | None, dict[str, Track]] = {}

        def kids(t: Track) -> bool:
            return any(k in t.haystack for k in scoring.KIDS_WORDS)

        # The service client has been idle since boot; its keep-alive socket to
        # JLTamp is likely dead. Reconnect once up front rather than stall on
        # the first call.
        self.client.refresh_connection()

        # Ronde 1: wie doet er mee, en wat draait ieder.
        profiles = []
        club: list[str] = []
        for u, uclient in self._gebruikers(only_user_id, results):
            uid = u.get("id")
            try:
                ranked, heard, info = self._weekly_favourites(uclient)
            except Exception as e:                    # noqa: BLE001
                results.append({"user": uid, "status": "error",
                                "message": f"luistergeschiedenis ophalen mislukt: {e}"})
                continue
            profiles.append((u, uclient, ranked, heard, info))
            if len(ranked) >= config.MIN_TASTE_SEED:
                club.extend(ranked[:40])

        # De gedeelde smaak, voor wie te weinig eigen geschiedenis heeft.
        club_seeds, _ = self.store.matrix(list(dict.fromkeys(club)))

        # Ronde 2: de lijsten.
        for u, uclient, ranked, heard, info in profiles:
            uid = u.get("id")
            made: list[str] = []
            own_taste = len(ranked) >= config.MIN_TASTE_SEED

            zicht = self._zicht_voor(u, by_key, visible_for)

            # 🧬 DNA Mix
            favourites = scoring.unique_songs(
                [zicht[k] for k in ranked if k in zicht and not kids(zicht[k])])
            dna = scoring.dj_volgorde(self._greep(favourites[:cap * 2], cap, 4, rng))
            # Een eigen, lagere drempel dan het smaakprofiel: vijftien echte
            # favorieten zijn al een DNA-mix. Op de algemene 25 kregen mensen met
            # 23 nummers die ze telkens opnieuw draaien er geen.
            if len(favourites) >= config.MIN_DNA_FAVOURITES and len(dna) >= 10:
                try:
                    rk = uclient.create_playlist(self.WEEKLY_DNA, dna)
                    reden_opslag.bewaar(rk, {t.rating_key: self._reden_dna(t, info) for t in dna})
                    made.append(f"{self.WEEKLY_DNA} ({len(dna)})")
                except Exception as e:                # noqa: BLE001
                    made.append(f"{self.WEEKLY_DNA} MISLUKT: {e}")
            else:
                made.append(f"{self.WEEKLY_DNA} overgeslagen: te weinig "
                            f"favorieten ({len(favourites)})")

            # 🔮 Ontdekking van de Week
            if own_taste:
                seeds, _present = self.store.matrix(ranked[:60])
                bron = "eigen smaak"
            else:
                seeds, bron = club_seeds, "smaak van iedereen"
            if seeds is None or len(seeds) == 0:
                made.append(f"{self.WEEKLY_DISCO} overgeslagen: nog geen "
                            f"smaakprofiel om uit te putten")
            else:
                last_week = set(uclient.playlist_track_keys(self.WEEKLY_DISCO))
                avoid = {by_key[k].song_key for k in heard | last_week if k in by_key}
                candidates: list[Track] = []
                band = self.store.nearest_to_any(
                    seeds, list(zicht), cap * 12 + len(heard) + len(last_week))
                for key, _sim in band:
                    if key in heard or key in last_week:
                        continue
                    t = zicht.get(key)
                    if t is None or kids(t):
                        continue
                    candidates.append(t)
                candidates = scoring.unique_songs(candidates, known=avoid)[:cap * 4]
                disco = scoring.dj_volgorde(self._greep(candidates, cap, 2, rng))
                if len(disco) >= 10:
                    try:
                        rk = uclient.create_playlist(self.WEEKLY_DISCO, disco)
                        if own_taste:
                            dichtst = self._dichtstbij(disco, ranked[:60])
                            redenen = {t.rating_key: self._reden_favoriet(
                                           t, zicht.get(dichtst.get(t.rating_key, "")))
                                       for t in disco}
                        else:
                            # Niet noemen wélk nummer van wie: dat zijn de
                            # favorieten van andere gebruikers.
                            redenen = {t.rating_key: self._met_kenmerken(
                                           "Geliefd bij andere luisteraars van JLTamp", t)
                                       for t in disco}
                        reden_opslag.bewaar(rk, redenen)
                        made.append(f"{self.WEEKLY_DISCO} ({len(disco)}, {bron})")
                    except Exception as e:            # noqa: BLE001
                        made.append(f"{self.WEEKLY_DISCO} MISLUKT: {e}")
                else:
                    made.append(f"{self.WEEKLY_DISCO} overgeslagen: te weinig "
                                f"nieuwe nummers ({len(disco)})")

            print(f"📅 weeklijsten gebruiker {uid}: " + "; ".join(made))
            results.append({"user": uid, "email": u.get("email"),
                            "status": "ok", "playlists": made})
        return results

    def _gebruikers(self, only_user_id: int | None, results: list[dict]):
        """Wie een eigen AI-lijst krijgt, met een client op zijn eigen token.

        Voor de weekronde en Nieuw voor jou: actieve accounts met een
        geaccepteerde uitnodiging. Wie afvalt, komt met de reden in `results`.
        """
        for u in self.client.list_users():
            uid = u.get("id")
            if only_user_id is not None and uid != only_user_id:
                continue
            if not u.get("isActive", u.get("is_active", True)):
                continue
            if u.get("pending"):
                results.append({"user": uid, "status": "skipped",
                                "message": "uitnodiging nog niet geaccepteerd"})
                continue
            created = u.get("createdAt", u.get("created_at", 0)) or 0
            if (config.MIN_ACCOUNT_AGE_SEC and created
                    and (time.time() - created) < config.MIN_ACCOUNT_AGE_SEC):
                results.append({"user": uid, "status": "skipped",
                                "message": "account nog te nieuw"})
                continue
            token = self.client.session_for(uid)
            if not token:
                results.append({"user": uid, "status": "skipped",
                                "message": "geen sessie (endpoint uitgerold?)"})
                continue
            yield u, JLTampClient(token=token)

    @staticmethod
    def _zicht_voor(u: dict, by_key: dict[str, Track],
                    cache: dict[frozenset | None, dict[str, Track]]) -> dict[str, Track]:
        """De nummers die deze gebruiker mag zien.

        Admins zien alles; de rest alleen hun toegewezen bibliotheken. Zonder dit
        kregen de Demo-gebruikers een Ontdekking uit Muzyk, waarvan de server bij
        het opslaan 44 van de 50 nummers weggooide.
        """
        if u.get("isAdmin") or u.get("is_admin"):
            access = None
        else:
            access = frozenset(str(x) for x in (u.get("libraryIds") or []))
        if access not in cache:
            cache[access] = (by_key if access is None else
                             {k: t for k, t in by_key.items() if t.library_id in access})
        return cache[access]

    # ── ✨ nieuw voor jou ────────────────────────────────────────────────────
    NIEUW = "✨ Nieuw voor jou"

    @staticmethod
    def _track_id(key: str) -> int:
        return int(key[1:]) if key.startswith("t") and key[1:].isdigit() else 0

    def _nieuw_state(self) -> dict:
        try:
            return json.loads(config.NIEUW_STATE_FILE.read_text())
        except (OSError, ValueError):
            return {}

    @staticmethod
    def _nieuw_bewaar(state: dict) -> None:
        config.NIEUW_STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        tmp = config.NIEUW_STATE_FILE.with_suffix(".tmp")
        tmp.write_text(json.dumps(state))
        tmp.replace(config.NIEUW_STATE_FILE)

    def _nieuw_check(self) -> None:
        """Na elke bibliotheek-verversing: is er nieuwe muziek binnen?

        Nieuw = een hoger track-id dan de vorige keer. Niet `added_at`: een
        volledige herscan zet die voor de hele bibliotheek opnieuw, terwijl de id's
        blijven. De lijst komt pas als alle nieuwe nummers een vector hebben én er
        een verversing lang niets meer bij kwam — een scan die nog loopt, of een
        weekronde die in stappen zichtbaar wordt, geeft anders drie halve lijsten.
        """
        if not self.ready or self.store is None:
            return
        tracks = self.library.snapshot()
        if not tracks:
            return
        hoogste = max(self._track_id(t.rating_key) for t in tracks)
        state = self._nieuw_state()
        if "max_id" not in state:
            # Eerste keer: alles wat er nu staat is niet nieuw.
            self._nieuw_bewaar({"max_id": hoogste})
            return
        vanaf = int(state["max_id"])
        nieuw = [t for t in tracks if self._track_id(t.rating_key) > vanaf]
        if len(nieuw) > config.NIEUW_MAX_BINNEN:
            print(f"✨ {len(nieuw)} 'nieuwe' nummers: vast een herscan — geen lijst.")
            self._nieuw_bewaar({"max_id": hoogste})
            return
        if len(nieuw) < config.NIEUW_MIN_BINNEN:
            return
        if any(not self.store.has(t.rating_key) for t in nieuw):
            return                                   # de embed-worker is nog bezig
        if state.get("wacht") != len(nieuw):
            state["wacht"] = len(nieuw)              # nog een verversing afwachten
            self._nieuw_bewaar(state)
            return
        print(f"✨ {len(nieuw)} nieuwe nummers binnen — Nieuw voor jou maken.")
        results = self.generate_nieuw(sinds_id=vanaf)
        gemaakt = sum(1 for r in results if r.get("status") == "ok")
        self._nieuw_bewaar({"max_id": hoogste, "laatst": int(time.time()),
                            "binnen": len(nieuw), "lijsten": gemaakt})

    def generate_nieuw(self, only_user_id: int | None = None, sinds_id: int | None = None,
                       dagen: int | None = None) -> list[dict]:
        """✨ Nieuw voor jou: van de nieuwe muziek alleen wat bij jou past.

        Nieuw is alles met een track-id boven `sinds_id` (zo roept de
        verversing hem aan), of — om met de hand te draaien — wat de laatste
        `dagen` is toegevoegd. Per gebruiker krijgt elk nieuw nummer een score:
        hoe dicht het bij zijn dichtstbijzijnde favoriet ligt. Alleen wat beter
        past dan zijn gewone bibliotheek (boven het 60e percentiel van een
        steekproef daaruit) komt erin; iemand die van rock houdt krijgt dus geen
        lijst vol nieuwe schlagers omdat die toevallig binnenkwamen. Wie te weinig
        eigen geschiedenis heeft, of voor wie niets past, krijgt niets.
        """
        if not self.ready or self.store is None:
            return [{"status": "error", "message": "engine not ready"}]
        if sinds_id is None and not dagen:
            return [{"status": "error", "message": "sinds_id of dagen nodig"}]
        tracks = self.library.snapshot()
        by_key = {t.rating_key: t for t in tracks}
        if sinds_id is not None:
            nieuw = [t for t in tracks if self._track_id(t.rating_key) > sinds_id]
        else:
            grens = time.time() - int(dagen) * 86400
            nieuw = [t for t in tracks if t.added_at >= grens]
        nieuw = [t for t in nieuw if self.store.has(t.rating_key)
                 and not any(k in t.haystack for k in scoring.KIDS_WORDS)]
        results: list[dict] = []
        if not nieuw:
            return [{"status": "skipped", "message": "geen nieuwe nummers"}]
        rng = random.Random()
        visible_for: dict[frozenset | None, dict[str, Track]] = {}
        self.client.refresh_connection()

        for u, uclient in self._gebruikers(only_user_id, results):
            uid = u.get("id")
            try:
                ranked, heard, _info = self._weekly_favourites(uclient)
            except Exception as e:                    # noqa: BLE001
                results.append({"user": uid, "status": "error",
                                "message": f"luistergeschiedenis ophalen mislukt: {e}"})
                continue
            if len(ranked) < config.MIN_TASTE_SEED:
                results.append({"user": uid, "status": "skipped",
                                "message": f"te weinig eigen smaak ({len(ranked)})"})
                continue
            zicht = self._zicht_voor(u, by_key, visible_for)
            kandidaten = [t.rating_key for t in nieuw
                          if t.rating_key in zicht and t.rating_key not in heard]
            if len(kandidaten) < config.NIEUW_MIN_NUMMERS:
                results.append({"user": uid, "status": "skipped",
                                "message": f"te weinig nieuwe nummers te zien ({len(kandidaten)})"})
                continue
            zaden, _present = self.store.matrix(ranked[:60])
            if zaden is None or len(zaden) == 0:
                results.append({"user": uid, "status": "skipped", "message": "geen smaakvectoren"})
                continue
            # Eén matmul voor de kandidaten én een steekproef als maatstaf.
            kand = set(kandidaten)
            rest = [k for k in zicht if k not in kand]
            proef = rng.sample(rest, min(2000, len(rest)))
            scores = self.store.nearest_to_any(zaden, kandidaten + proef,
                                               len(kandidaten) + len(proef))
            ref = [s for k, s in scores if k not in kand]
            drempel = float(np.percentile(ref, 60)) if ref else -1.0
            passend = [zicht[k] for k, s in scores if k in kand and s >= drempel]
            passend = scoring.unique_songs(passend)

            # Hooguit vier per artiest: een nieuw album is anders meteen de hele
            # lijst. Niet aanvullen met wat die grens tegenhield: in de proef werd
            # dat 16 keer Belinda Carlisle van 30. Liever een kortere lijst.
            lijst: list[Track] = []
            per: dict[str, int] = {}
            for t in passend:
                if len(lijst) >= config.NIEUW_MAX_NUMMERS:
                    break
                a = t.real_artist.strip().lower()
                if a not in Track._COMPILATION and per.get(a, 0) >= 4:
                    continue
                per[a] = per.get(a, 0) + 1
                lijst.append(t)
            if len(lijst) < config.NIEUW_MIN_NUMMERS:
                results.append({"user": uid, "status": "skipped",
                                "message": f"te weinig passend nieuws ({len(lijst)})"})
                continue
            lijst = scoring.dj_volgorde(lijst)
            try:
                rk = uclient.create_playlist(self.NIEUW, lijst)
            except Exception as e:                    # noqa: BLE001
                results.append({"user": uid, "status": "error", "message": f"aanmaken mislukt: {e}"})
                continue
            dichtst = self._dichtstbij(lijst, ranked[:60])
            reden_opslag.bewaar(rk, {
                t.rating_key: "Nieuw binnen · " + self._reden_favoriet(
                    t, zicht.get(dichtst.get(t.rating_key, "")))
                for t in lijst})
            print(f"✨ Nieuw voor jou gebruiker {uid}: {len(lijst)} van {len(kandidaten)} nieuwe")
            results.append({"user": uid, "email": u.get("email"), "status": "ok",
                            "playlist_key": rk, "tracks": len(lijst),
                            "nieuw": len(kandidaten)})
        return results

    # ── introspection ────────────────────────────────────────────────────────
    def health(self) -> dict:
        tracks = self.library.snapshot()
        # `embeddings` counts stored vectors, including ones whose text has
        # since changed — during a rebuild that number sits at the total and
        # shows no progress at all. `stale` is what is actually left to do.
        #
        # Read from a counter the embed worker maintains — never recomputed
        # here. Hashing every track text on each request made /health take
        # about a second, and polling it backed the whole server up behind a
        # 17-deep queue.
        stale = self._stale_count
        return {
            "status": "ok",
            "ai_ready": self.ready,
            "state": self.status,
            "error": self.last_error or None,
            "tracks": len(tracks),
            "embeddings": len(self.store) if self.store else 0,
            "stale": stale,
            "dimensions": self.embedder.dim,
            "model": self.embedder.model_id,
            # Tracks with a usable tempo, and the raw size of the cache behind
            # it. The two differ by whatever the file still knows about music
            # that is no longer in the library.
            "features": self._measured,
            "feature_entries": len(self.features),
            "jltamp": config.JLTAMP_URL,
        }
