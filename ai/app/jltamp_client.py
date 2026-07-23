"""Talks to the JLTamp server.

This replaces the Plex coupling of the old engine. Everything the AI needs —
the track list, listening history, likes, and playlist creation — comes from
JLTamp's own API, so the playlists it builds land where the app can see them.

Auth is a session token sent as `X-Plex-Token` (JLTamp kept Plex's header name).
Supply `JLTAMP_TOKEN`, or an email/password pair that we exchange for one.
"""
from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config


@dataclass
class Track:
    """The subset of a JLTamp track the scorer actually reads."""
    rating_key: str            # "t123"
    title: str
    artist: str                # album/grouping artist
    orig_artist: str           # the track's own artist, set on compilations
    album: str
    year: int | None
    genre: str
    duration_ms: int
    play_count: int
    last_played_at: int
    rating: float
    library_id: str = ""
    path: str = ""

    # Derived listening signals, filled in by the engine (not by the API).
    skips: int = 0
    liked: bool = False
    features: dict = field(default_factory=dict)

    # Placeholders a compilation uses where a real artist name would go.
    _COMPILATION = {"various artists", "various", "va", "verschillende artiesten",
                    "diverse artiesten", "soundtrack", "unknown artist", "unknown"}

    @property
    def real_artist(self) -> str:
        """Who actually performs this — not the album's grouping artist.

        On compilations `artist` is "Various Artists", which carries no musical
        meaning at all. Embedded as-is, such tracks land near the middle of the
        vector space and come back moderately similar to *every* prompt: they
        surfaced under "slaapmuziek", "metal" and "focus" alike. JLTamp already
        stores the performer in originalTitle for exactly this case.
        """
        if self.artist.strip().lower() in self._COMPILATION and self.orig_artist:
            return self.orig_artist
        return self.artist

    @property
    def clean_title(self) -> str:
        """The title without the scanner's leftovers.

        Some tags come through as "1 - The Weeknd - Blinding Lights": a track
        number and the artist repeated inside the title. Embedded as-is, that
        doubles the artist's weight and injects a meaningless number.
        """
        title = re.sub(r"^\s*\d+\s*[-.]\s*", "", self.title or "")
        for name in (self.artist, self.orig_artist):
            if name:
                title = re.sub(rf"^{re.escape(name)}\s*-\s*", "", title,
                               flags=re.IGNORECASE)
        return title.strip() or (self.title or "")

    @property
    def text(self) -> str:
        """What gets embedded. Genre carries most of the "what does this sound
        like" signal; artist and title alone barely carry any.

        The album is deliberately left out when the artist is a compilation
        placeholder — "Radio Piepschuim" describes the CD, not the music.
        """
        artist = self.real_artist
        bits = [artist, self.clean_title]
        if self.genre:
            bits.append(self.genre)
        compilation = self.artist.strip().lower() in self._COMPILATION
        if (self.album and not compilation
                and self.album.lower() not in (self.title or "").lower()):
            bits.append(self.album)
        return " - ".join(b for b in bits if b)

    @property
    def haystack(self) -> str:
        """Lowercased blob for keyword/exclusion matching."""
        return (f"{self.artist} {self.orig_artist} {self.title} "
                f"{self.album} {self.genre}").lower()


class JLTampError(RuntimeError):
    pass


class JLTampClient:
    def __init__(self, url: str | None = None, token: str | None = None):
        self.url = (url or config.JLTAMP_URL).rstrip("/")
        self._token = token or config.JLTAMP_TOKEN
        self._lock = threading.Lock()
        self.session = self._make_session()

    @staticmethod
    def _make_session() -> requests.Session:
        s = requests.Session()
        # Retry only idempotent reads; a retried playlist POST would duplicate
        # it. A dead keep-alive socket the server closed while we were idle is a
        # connection error, so retrying READ re-opens a fresh one rather than
        # hanging on the read timeout — the weekly job sat idle between calls
        # and paid a 60 s+ stall on the first stale socket without this.
        retries = Retry(total=3, backoff_factor=0.5, connect=3, read=2,
                        status_forcelist=[502, 503, 504],
                        allowed_methods=["GET"])
        adapter = HTTPAdapter(max_retries=retries)
        s.mount("http://", adapter)
        s.mount("https://", adapter)
        return s

    def refresh_connection(self) -> None:
        """Drop pooled sockets and start fresh. Called before a batch job that
        has been idle long enough for the server to have closed keep-alive."""
        try:
            self.session.close()
        except Exception:                            # noqa: BLE001
            pass
        self.session = self._make_session()

    # ── auth ─────────────────────────────────────────────────────────────────
    def _headers(self) -> dict:
        h = {"Accept": "application/json", "X-Plex-Product": "JLTamp AI"}
        if self._token:
            h["X-Plex-Token"] = self._token
        return h

    def login(self) -> str:
        """Exchange email+password for a session token. No-op if we have one."""
        if self._token:
            return self._token
        if not (config.JLTAMP_EMAIL and config.JLTAMP_PASSWORD):
            raise JLTampError(
                "No JLTAMP_TOKEN and no JLTAMP_EMAIL/JLTAMP_PASSWORD — cannot "
                "authenticate to JLTamp. See .env.example."
            )
        r = self.session.post(
            f"{self.url}/auth/login",
            json={"email": config.JLTAMP_EMAIL, "password": config.JLTAMP_PASSWORD},
            timeout=20,
        )
        if r.status_code == 401:
            raise JLTampError("JLTamp rejected the credentials (401).")
        r.raise_for_status()
        token = (r.json() or {}).get("token")
        if not token:
            raise JLTampError("JLTamp login returned no token.")
        self._token = token
        return token

    def list_users(self) -> list[dict]:
        """All JLTamp users. Needs the admin account — used by the weekly job
        to know who to build playlists for."""
        data = self._get("/admin/users")
        return data.get("users", []) if isinstance(data, dict) else []

    def session_for(self, user_id: int) -> str | None:
        """Mint a session token for another user (admin only).

        The weekly playlists run as a background job with no user logged in, so
        the engine asks JLTamp — as admin — for a token per user, then acts as
        that user. Returns None if the endpoint is absent (server not yet
        deployed) or the user is gone, so the caller can skip rather than crash.
        """
        try:
            r = self.session.post(f"{self.url}/admin/users/{user_id}/session",
                                  headers=self._headers(), timeout=20)
        except requests.RequestException:
            return None
        if r.status_code != 200:
            return None
        return (r.json() or {}).get("token")

    def me(self) -> dict | None:
        """Who this token belongs to, or None if JLTamp rejects it.

        This is what lets the AI trust a caller: rather than keeping its own
        user list or a shared password, it asks JLTamp whether the token the
        app already holds is valid, and for whom.
        """
        try:
            r = self.session.get(f"{self.url}/auth/me", headers=self._headers(),
                                 timeout=15)
        except requests.RequestException:
            return None
        if r.status_code != 200:
            return None
        try:
            data = r.json() or {}
        except ValueError:
            return None
        return data.get("user", data) or None

    def _get(self, path: str, **params) -> dict:
        r = self.session.get(f"{self.url}{path}", headers=self._headers(),
                             params=params, timeout=60)
        if r.status_code == 401:
            # Token expired or was revoked — get a fresh one and retry once.
            self._token = ""
            self.login()
            r = self.session.get(f"{self.url}{path}", headers=self._headers(),
                                 params=params, timeout=60)
        r.raise_for_status()
        return r.json() or {}

    # ── library ──────────────────────────────────────────────────────────────
    def healthz(self) -> dict:
        r = self.session.get(f"{self.url}/healthz", timeout=10)
        r.raise_for_status()
        return r.json() or {}

    def libraries(self) -> list[dict]:
        data = self._get("/library/sections")
        return data.get("MediaContainer", {}).get("Directory", []) or []

    def fetch_tracks(self, page_size: int = 500) -> list[Track]:
        """Every track the authenticated user can see, across all libraries.

        Paged: a full library in one response would be a many-megabyte JSON that
        both ends have to hold in memory at once.
        """
        out: list[Track] = []
        for lib in self.libraries():
            key = lib.get("key")
            if not key:
                continue
            start = 0
            while True:
                data = self._get(
                    f"/library/sections/{key}/all",
                    type=10,
                    **{"X-Plex-Container-Start": start,
                       "X-Plex-Container-Size": page_size},
                )
                mc = data.get("MediaContainer", {})
                items = mc.get("Metadata", []) or []
                if not items:
                    break
                for m in items:
                    out.append(self._to_track(m, str(key)))
                start += len(items)
                total = int(mc.get("totalSize") or 0)
                if total and start >= total:
                    break
                if len(items) < page_size:
                    break
        return out

    @staticmethod
    def _to_track(m: dict, library_id: str) -> Track:
        media = (m.get("Media") or [{}])[0]
        part = (media.get("Part") or [{}])[0]
        return Track(
            rating_key=str(m.get("ratingKey") or ""),
            title=m.get("title") or "",
            artist=m.get("grandparentTitle") or "",
            orig_artist=m.get("originalTitle") or "",
            album=m.get("parentTitle") or "",
            year=m.get("year"),
            # `genre` needs the serializer patch; older servers just omit it.
            genre=(m.get("genre") or "") or "",
            duration_ms=int(m.get("duration") or 0),
            play_count=int(m.get("viewCount") or 0),
            last_played_at=int(m.get("lastViewedAt") or 0),
            rating=float(m.get("userRating") or 0),
            library_id=library_id,
            path=part.get("file") or "",
        )

    def genres(self, library_key: str) -> list[str]:
        data = self._get(f"/library/sections/{library_key}/genre")
        mc = data.get("MediaContainer", {})
        items = mc.get("Directory") or mc.get("Metadata") or []
        return [g.get("title") for g in items if g.get("title")]

    def backfill_genres(self, tracks: list[Track]) -> int:
        """Fill in genres by asking per genre which tracks have it.

        Older servers do not serialise `genre` on a track (it is on the model
        but was missing from track_dict), yet they will happily filter by it.
        So we walk the genre list and label what comes back. Without this the
        embedding text is just "artist - title - album", which says nothing
        about what the music sounds like — "Radiohead - Creep" gives the model
        no way to know it is not hip-hop.

        Skipped entirely when the server already sends genres.
        """
        by_key = {t.rating_key: t for t in tracks}
        filled = 0
        for lib in self.libraries():
            key = lib.get("key")
            if not key:
                continue
            for genre in self.genres(str(key)):
                start = 0
                while True:
                    try:
                        data = self._get(
                            f"/library/sections/{key}/all", type=10, genre=genre,
                            **{"X-Plex-Container-Start": start,
                               "X-Plex-Container-Size": 500},
                        )
                    except requests.HTTPError:
                        break
                    items = data.get("MediaContainer", {}).get("Metadata", []) or []
                    if not items:
                        break
                    for m in items:
                        t = by_key.get(str(m.get("ratingKey")))
                        if t is not None and not t.genre:
                            t.genre = genre
                            filled += 1
                    start += len(items)
                    if len(items) < 500:
                        break
        return filled

    # ── listening signals ────────────────────────────────────────────────────
    def liked_ids(self) -> set[str]:
        try:
            data = self._get("/likes/ids")
        except requests.HTTPError:
            return set()
        mc = data.get("MediaContainer", data)
        ids = mc.get("ratingKeys") or mc.get("Metadata") or []
        out = set()
        for i in ids:
            out.add(str(i.get("ratingKey")) if isinstance(i, dict) else str(i))
        return out

    def most_played_ids(self, limit: int = 60) -> list[str]:
        """This user's most-played track keys — the core of their taste."""
        try:
            data = self._get("/history/mostPlayed", type=10, limit=limit)
        except requests.HTTPError:
            return []
        items = data.get("MediaContainer", {}).get("Metadata", []) or []
        return [str(m.get("ratingKey")) for m in items if m.get("ratingKey")]

    def skip_counts(self, days: int = 90, max_events: int = 5000) -> dict[str, int]:
        """How often each track was abandoned early.

        A skip is a real behavioural signal only when the listener bailed out —
        `completed` false and under a third played. The old engine counted every
        pause as a skip, which quietly punished tracks you liked enough to pause.
        """
        skips: dict[str, int] = {}
        start = 0
        page = 500
        while start < max_events:
            try:
                data = self._get("/stats/history", days=days, limit=page, start=start)
            except requests.HTTPError:
                break
            items = data.get("MediaContainer", {}).get("History", []) or []
            if not items:
                break
            for ev in items:
                if ev.get("completed"):
                    continue
                if int(ev.get("percent") or 0) >= 33:
                    continue
                if int(ev.get("listenedMs") or 0) > 45_000:
                    continue  # sat through most of a long intro; not a rejection
                rk = str(ev.get("ratingKey") or "")
                if rk:
                    skips[rk] = skips.get(rk, 0) + 1
            start += len(items)
            if len(items) < page:
                break
        return skips

    # ── playlists ────────────────────────────────────────────────────────────
    def find_playlist(self, title: str) -> str | None:
        data = self._get("/playlists")
        for p in data.get("MediaContainer", {}).get("Metadata", []) or []:
            if (p.get("title") or "").strip().lower() == title.strip().lower():
                return str(p.get("ratingKey"))
        return None

    def delete_playlist(self, rating_key: str) -> None:
        self.session.delete(f"{self.url}/playlists/{rating_key}",
                            headers=self._headers(), timeout=30)

    def create_playlist(self, title: str, tracks: list[Track]) -> str:
        """Create (replacing a same-named one) and return its ratingKey.

        JLTamp takes track keys in Plex's `uri=` form. We chunk the additions:
        a 50-track uri is fine, but a few hundred would blow past sane URL
        lengths, and this is the one call that must not half-fail.
        """
        if not tracks:
            raise JLTampError("refusing to create an empty playlist")

        with self._lock:
            existing = self.find_playlist(title)
            if existing:
                self.delete_playlist(existing)

            head, tail = tracks[:50], tracks[50:]
            uri = self._uri([t.rating_key for t in head])
            r = self.session.post(
                f"{self.url}/playlists", headers=self._headers(),
                params={"title": title, "uri": uri, "type": "audio", "smart": 0},
                timeout=60,
            )
            r.raise_for_status()
            meta = (r.json() or {}).get("MediaContainer", {}).get("Metadata", [])
            if not meta:
                raise JLTampError("playlist create returned no ratingKey")
            rk = str(meta[0].get("ratingKey"))

            for i in range(0, len(tail), 50):
                chunk = tail[i:i + 50]
                self.session.put(
                    f"{self.url}/playlists/{rk}/items", headers=self._headers(),
                    params={"uri": self._uri([t.rating_key for t in chunk])},
                    timeout=60,
                )
            return rk

    @staticmethod
    def _uri(keys: list[str]) -> str:
        return ("server://jltamp-ai/com.plexapp.plugins.library/library/metadata/"
                + ",".join(keys))


class Library:
    """An in-memory snapshot of the library, refreshed on a timer.

    Scoring runs over every track on every request, so re-fetching per request
    would put the whole library on the wire each time. One snapshot, shared,
    swapped atomically.
    """

    def __init__(self, client: JLTampClient):
        self.client = client
        self.tracks: list[Track] = []
        self.by_key: dict[str, Track] = {}
        self.loaded_at = 0.0
        self._lock = threading.Lock()

    def refresh(self, features: dict | None = None) -> int:
        tracks = self.client.fetch_tracks()

        # Genre is the strongest "what does this sound like" signal we have.
        # If the server did not send any, ask for it the long way round.
        if tracks and not any(t.genre for t in tracks):
            filled = self.client.backfill_genres(tracks)
            print(f"🏷️  Genres backfilled for {filled} tracks "
                  f"(server does not serialise `genre` yet).")

        skips = self.client.skip_counts()
        liked = self.client.liked_ids()
        feats = features or {}
        for t in tracks:
            t.skips = skips.get(t.rating_key, 0)
            t.liked = t.rating_key in liked
            t.features = feats.get(t.rating_key, {})
        with self._lock:
            self.tracks = tracks
            self.by_key = {t.rating_key: t for t in tracks}
            self.loaded_at = time.time()
        return len(tracks)

    def snapshot(self) -> list[Track]:
        with self._lock:
            return self.tracks
