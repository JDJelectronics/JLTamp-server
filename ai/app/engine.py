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
from . import intent_llm
from .embed_client import EmbedClient, EmbedError
from .embed_store import DimensionMismatch, EmbeddingStore
from .jltamp_client import JLTampClient, Library, Track

BEST_OF_TRIGGERS = ("leukste", "beste", "best of", "songs van", "playlist van", "top van")
DISCOVERY_TRIGGERS = ("vergelijkbaar", "similar", "discovery", "radio", "lijkt op",
                      "zoals", "in de stijl van", "familiair")
# A wind-down is an *ordering* intent, not just "calm music": the tempo has to
# descend. So these are explicit descent / sleep cues — plain "rustige muziek"
# is deliberately left out so it still gets a normal calm playlist, not a ramp.
WIND_DOWN_TRIGGERS = ("afbouw", "afbouwen", "wind down", "wind-down", "winddown",
                      "in slaap", "inslapen", "slapen", "om te slapen",
                      "steeds rustiger", "aflopend tempo", "tempo omlaag",
                      "tot rust komen", "cool down", "cooldown",
                      "afkoelen", "naar rust")
# The wind-down's siblings: same measured tempo, a different shape. Each of
# these is an *ordering* intent, so they are matched before the artist
# strategies for the same reason wind-down is.
BUILD_UP_TRIGGERS = ("opbouw", "opbouwen", "wakker worden", "warm draaien",
                     "warmdraaien", "warming-up", "warming up", "opstarten",
                     "steeds sneller", "tempo omhoog", "op gang komen")
# Deliberately compound words. A bare "boog" would fire on "regenboog", and a
# bare "piek" on any lyric about one.
PARTY_ARC_TRIGGERS = ("feestboog", "energieboog", "opbouw en afbouw",
                      "eerst opbouwen dan afbouwen", "naar een piek",
                      "piek en dan rustig")
INTERVAL_TRIGGERS = ("interval", "intervaltraining", "hiit", "tempoblokken",
                     "blokken van")
ARTIST_SPLITTERS = ("zoals ", "lijkt op ", "van ", "voor ", "artiest ", "artist ",
                    "bij ", "naar ")

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

        ctx = {"client": client, "user": user, "likes": likes,
               "skips": skips, "at": now}
        with self._users_lock:
            self._users[token] = ctx
            # Bound the cache: one entry per active token, oldest dropped first.
            if len(self._users) > 50:
                oldest = min(self._users, key=lambda k: self._users[k]["at"])
                self._users.pop(oldest, None)
        return ctx

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
            a = t.real_artist
            if seen_artist.get(a, 0) >= 2:      # spreiding over artiesten
                continue
            picks.append(t)
            seen_artist[a] = seen_artist.get(a, 0) + 1
            if len(picks) >= cap:
                break
        if not picks:
            return {"status": "error", "message": "geen persoonlijke aanbevelingen gevonden"}
        return self._publish("Voor jou ✨", picks, client)

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

    def _seed_vector(self, s: str, by_key: dict, tracks: list):
        """Los één radio-zaadje op naar (vector, label, track_key|None)."""
        if s in by_key:
            mat, present = self.store.matrix([s])
            if present:
                st = by_key[s]
                return mat[0], f"{st.real_artist} - {st.clean_title}", s
        v = self._artist_vector(s, tracks)
        return (v, s, None) if v is not None else (None, s, None)

    def radio(self, seed, exclude: set[str], count: int = 20, dislike=None) -> dict:
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
        by_key = {t.rating_key: t for t in tracks}

        # Multi-seed: elk zaadje (track-key of artiest) -> vector, gemiddelde =
        # het middelpunt van de "meng deze" -radio.
        seeds = seed if isinstance(seed, (list, tuple)) else [seed]
        seed_vecs, labels, seed_keys = [], [], []
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

        label = " + ".join(labels) if len(labels) > 1 else labels[0]
        skip = set(exclude) | set(seed_keys) | dislike
        rng = random.Random()
        picks: list[Track] = []
        # Pull a wide band, then keep the nearest that are not excluded. The
        # band grows the deeper the station runs (exclude gets large), so ask
        # for generously more than needed.
        band = self.store.top_keys(target, [t.rating_key for t in tracks],
                                   count + len(skip) + 200)
        seen_artist: dict[str, int] = {}
        for key, _sim in band:
            if key in skip:
                continue
            t = by_key.get(key)
            if t is None or any(k in t.haystack for k in scoring.KIDS_WORDS):
                continue
            # Loose per-artist cap so a station does not become one artist on
            # repeat, but still stays firmly in the seed's neighbourhood.
            a = t.real_artist
            if seen_artist.get(a, 0) >= 3:
                continue
            picks.append(t)
            seen_artist[a] = seen_artist.get(a, 0) + 1
            if len(picks) >= count:
                break

        rng.shuffle(picks)                   # mild ordering variety within the batch
        return {
            "status": "ok",
            "seed": label,
            "tracks": [{"ratingKey": t.rating_key,
                        "title": t.clean_title,
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
        return self._publish(f"⭐ Best of {artist}", own[:40], client)

    def _discovery(self, artist: str, tracks: list[Track], client,
                   signals: scoring.Signals | None = None) -> dict:
        signals = signals or scoring.LIBRARY_SIGNALS
        target = self._artist_vector(artist, tracks)
        if target is None:
            return {"status": "error",
                    "message": f"nog geen embeddings voor '{artist}'"}

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
            score = sim - seen.get(t.real_artist, 0) * 0.15 + rng.uniform(0, 0.05)
            if t.play_count == 0:
                score += 0.3
            if t.real_artist.lower() == artist.lower():
                score -= 0.25          # discovery means *other* artists
            skips = signals.skips(t)
            if skips:
                score -= min(skips, 4) * 0.2
            scored.append((t, score))
            if sim > 0.6:
                seen[t.real_artist] = seen.get(t.real_artist, 0) + 1

        scored.sort(key=lambda p: p[1], reverse=True)
        picked = [t for t, _ in scored[:config.SCORING["MAX_TRACKS"]]]
        return self._publish(f"🕵️ Lijkt op {artist}", picked, client)

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
        return self._publish(scoring.playlist_name(low), picked, client)

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
        return self._publish("🌙 Afbouw — rustig naar het einde", curve, client)

    def _graded_pool(self, prompt: str, tracks: list[Track], client,
                     signals: scoring.Signals | None
                     ) -> list[tuple[Track, float]]:
        """The tempo pool as (track, bpm) pairs. Only measured tracks: putting
        a track with no tempo on a tempo curve is ordering it at random."""
        pool = self._tempo_pool(prompt, tracks, client, signals)
        return [(t, t.features["bpm"]) for t in pool if t.features.get("bpm")]

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
        return self._publish(name, curve, client)

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
        res = self._publish(name, picks, client)
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

    def _publish(self, name: str, tracks: list[Track], client) -> dict:
        if not tracks:
            return {"status": "error", "message": "geen resultaten"}
        try:
            playlist_key = client.create_playlist(name, tracks)
        except Exception as e:                      # noqa: BLE001
            print(f"❌ Playlist creation failed: {e}")
            return {"status": "error", "message": f"playlist aanmaken mislukt: {e}"}
        print(f"🆕 Playlist '{name}' — {len(tracks)} tracks.")
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

    def generate_weekly(self, only_user_id: int | None = None) -> list[dict]:
        """Build 'DNA Mix' and 'Discovery' playlists for each active user.

        Runs as the admin service, minting a per-user token so each playlist is
        created in that user's own library from that user's own taste — never
        shared, never in the wrong account.
        """
        if not self.ready:
            return [{"status": "error", "message": "engine not ready"}]
        tracks = self.library.snapshot()
        by_key = {t.rating_key: t for t in tracks}
        results = []

        # The service client has been idle since boot; its keep-alive socket to
        # JLTamp is likely dead. Reconnect once up front rather than stall on
        # the first call.
        self.client.refresh_connection()
        for u in self.client.list_users():
            uid = u.get("id")
            if only_user_id is not None and uid != only_user_id:
                continue
            if not u.get("isActive", u.get("is_active", True)):
                continue
            # A month on the account before we claim to know their taste. New
            # users have not listened enough to profile, however active — the
            # listening-volume check below is the other half of the same rule.
            created = u.get("createdAt", u.get("created_at", 0)) or 0
            if created and (time.time() - created) < config.MIN_ACCOUNT_AGE_SEC:
                results.append({"user": uid, "status": "skipped",
                                "message": "account younger than a month"})
                continue
            token = self.client.session_for(uid)
            if not token:
                results.append({"user": uid, "status": "skipped",
                                "message": "no session (endpoint deployed?)"})
                continue

            uclient = JLTampClient(token=token)
            taste = self._taste_vector(uclient)
            if taste is None:
                results.append({"user": uid, "status": "skipped",
                                "message": "too little listening history"})
                continue
            centroid, favourites = taste
            cap = config.SCORING["MAX_TRACKS"]
            known = set(favourites)

            # DNA Mix = the user's own favourites, in play-count order. These
            # ARE what they love; the earlier version searched near the taste
            # centroid and filtered to known tracks, but favourites sit
            # scattered around their own average, not on top of it, so it found
            # almost none (1 of 50).
            dna = []
            for key in favourites:
                t = by_key.get(key)
                if t and not any(k in t.haystack for k in scoring.KIDS_WORDS):
                    dna.append(t)
                if len(dna) >= cap:
                    break

            # Discovery = nearest the taste centroid, but only tracks NOT yet
            # played — the point is to surface things they would like but have
            # not heard.
            disco = []
            for key, _sim in self.store.top_keys(
                    centroid, [t.rating_key for t in tracks], cap * 6):
                if key in known:
                    continue
                t = by_key.get(key)
                if t and not any(k in t.haystack for k in scoring.KIDS_WORDS):
                    disco.append(t)
                if len(disco) >= cap:
                    break

            made = []
            for name, picks in (("🧬 Jouw DNA Mix", dna),
                                ("🔮 Ontdekking van de Week", disco)):
                if len(picks) >= 10:
                    try:
                        uclient.create_playlist(name, picks)
                        made.append(f"{name} ({len(picks)})")
                    except Exception as e:        # noqa: BLE001
                        made.append(f"{name} FAILED: {e}")
            results.append({"user": uid, "email": u.get("email"),
                            "status": "ok", "playlists": made})
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
