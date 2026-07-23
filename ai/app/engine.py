"""The engine: keeps the library and its vectors in sync, and answers prompts.

Startup order matters. We need the embedding server's dimension before we can
open the vector store, because the dimension is what tells us whether the
vectors on disk belong to the model now running.
"""
from __future__ import annotations

import json
import random
import re
import threading
import time

import numpy as np

from . import config, genre_infer, scoring
from .embed_client import EmbedClient, EmbedError
from .embed_store import DimensionMismatch, EmbeddingStore
from .jltamp_client import JLTampClient, JLTampError, Library, Track

BEST_OF_TRIGGERS = ("leukste", "beste", "best of", "songs van", "playlist van", "top van")
DISCOVERY_TRIGGERS = ("vergelijkbaar", "similar", "discovery", "radio", "lijkt op",
                      "zoals", "in de stijl van", "familiair")
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

    # ── startup ──────────────────────────────────────────────────────────────
    def start(self) -> None:
        threading.Thread(target=self._boot, daemon=True).start()

    def _boot(self) -> None:
        self.features = self._load_features()

        try:
            self.client.login()
        except (JLTampError, Exception) as e:      # noqa: BLE001 - report, keep trying
            self.status = "jltamp-auth-failed"
            self.last_error = str(e)
            print(f"❌ JLTamp login failed: {e}")
            return

        try:
            count = self.library.refresh(self.features)
            self._apply_inferred()
            print(f"✅ JLTamp connected: {count} tracks.")
        except Exception as e:                      # noqa: BLE001
            self.status = "jltamp-unreachable"
            self.last_error = str(e)
            print(f"❌ Could not load the library: {e}")
            return

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
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"⚠️  Could not read {path}: {e}")
            return {}
        print(f"🎚️  Audio features loaded for {len(data)} tracks.")
        return data

    # ── background work ──────────────────────────────────────────────────────
    def _embed_worker(self) -> None:
        """Fill in vectors for tracks that don't have one yet."""
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
            print(f"✅ {done} embeddings added ({len(self.store)} total).")
            time.sleep(5)

    def _refresh_worker(self) -> None:
        while True:
            time.sleep(config.LIBRARY_REFRESH_SEC)
            try:
                count = self.library.refresh(self.features)
                self._apply_inferred()
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
                    t.genre = g
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
    def _apply_user_signals(tracks: list[Track], ctx: dict | None) -> None:
        """Re-point liked/skipped onto the calling user's own history."""
        if not ctx:
            return
        likes, skips = ctx["likes"], ctx["skips"]
        for t in tracks:
            t.liked = t.rating_key in likes
            t.skips = skips.get(t.rating_key, 0)

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
            # Reading 69k tracks and backfilling genres takes a few minutes, so
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
        self._apply_user_signals(tracks, ctx)
        client = ctx["client"] if ctx else self.client

        if not self.ready:
            return self._fallback(tracks, prompt, client)

        artist = self._find_artist(low, tracks)
        if artist and any(w in low for w in BEST_OF_TRIGGERS):
            return self._best_of(artist, tracks, client)
        if artist and any(w in low for w in DISCOVERY_TRIGGERS):
            return self._discovery(artist, tracks, client)
        return self._semantic(prompt, tracks, client)

    def _find_artist(self, prompt: str, tracks: list[Track]) -> str | None:
        """The artist a prompt is about, if any.

        Tries the phrase after a connector first ("zoals Adele"), then falls
        back to finding any known artist name inside the prompt. Longest match
        wins so "Bruce Springsteen" beats a stray "Bruce".
        """
        for splitter in ARTIST_SPLITTERS:
            if splitter in prompt:
                tail = prompt.split(splitter)[-1].strip()
                tail = re.sub(r"[^\w\s].*$", "", tail).strip()
                if tail:
                    for t in tracks:
                        if t.artist and t.artist.lower() == tail:
                            return t.artist
        best = None
        for t in tracks:
            name = (t.artist or "").lower()
            if len(name) > 3 and name in prompt:
                if best is None or len(name) > len(best.lower()):
                    best = t.artist
        return best

    def _artist_vector(self, artist: str, tracks: list[Track]) -> np.ndarray | None:
        """An artist's centroid: the average of their tracks' vectors."""
        keys = [t.rating_key for t in tracks if t.artist.lower() == artist.lower()]
        mat, present = self.store.matrix(keys)
        if not present:
            return None
        return mat.mean(axis=0)

    def _best_of(self, artist: str, tracks: list[Track], client) -> dict:
        own = [t for t in tracks if t.artist.lower() == artist.lower()]
        if not own:
            return {"status": "error", "message": f"'{artist}' niet gevonden"}
        own.sort(key=lambda t: (t.liked, t.play_count, -t.skips), reverse=True)
        return self._publish(f"⭐ Best of {artist}", own[:40], client)

    def _discovery(self, artist: str, tracks: list[Track], client) -> dict:
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
            score = sim - seen.get(t.artist, 0) * 0.15 + rng.uniform(0, 0.05)
            if t.play_count == 0:
                score += 0.3
            if t.artist.lower() == artist.lower():
                score -= 0.25          # discovery means *other* artists
            if t.skips:
                score -= min(t.skips, 4) * 0.2
            scored.append((t, score))
            if sim > 0.6:
                seen[t.artist] = seen.get(t.artist, 0) + 1

        scored.sort(key=lambda p: p[1], reverse=True)
        picked = [t for t, _ in scored[:config.SCORING["MAX_TRACKS"]]]
        return self._publish(f"🕵️ Lijkt op {artist}", picked, client)

    def _semantic(self, prompt: str, tracks: list[Track], client) -> dict:
        try:
            # Short prompts get expanded before embedding; scoring still sees
            # the original, so keyword and artist matching stay honest.
            query = self.embedder.embed_one(scoring.expand_query(prompt))
        except (EmbedError, Exception) as e:        # noqa: BLE001
            print(f"⚠️  Prompt embedding failed: {e}")
            return self._fallback(tracks, prompt, client)

        sims = scoring.similarity_map(self.store, query, tracks)
        if not sims:
            return {"status": "error",
                    "message": "nog geen embeddings — de engine is nog aan het indexeren"}

        scored = scoring.score_tracks(prompt, tracks, sims)
        low = prompt.lower()
        relaxed = bool(scoring.active_contexts(low) or scoring.extract_year(low))
        picked = scoring.select(scored, relaxed=relaxed)
        if not picked:
            return {"status": "error", "message": "niets gevonden dat hierbij past"}
        return self._publish(scoring.playlist_name(low), picked, client)

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
            client.create_playlist(name, tracks)
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
        # `playlist` is the field aiService.ts reads for the name it shows.
        return {
            "status": "success",
            "playlist": name,
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
        # here. Hashing 69k track texts on every request made /health take
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
            "features": len(self.features),
            "jltamp": config.JLTAMP_URL,
        }
