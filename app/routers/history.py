"""Per-user playback history: play counts, last-played time and resume offset.

The clients (phone + web) already send Plex's `/:/timeline` heartbeat every 10s
while playing, so implementing it here is enough to record history — no client
change needed. `/:/scrobble` is honoured too for players that use it.

State is per user (see UserTrackState), never on Track: two people playing the
same file must not share a play count or a resume position.
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy import select, func

from ..db import SessionLocal
from ..deps import require_user, accessible_library_ids
from ..ids import parse_key
from ..models import Track, Album, User, UserTrackState, PlayEvent
from ..serializers import track_dict, album_dict, container

router = APIRouter()

# A play is counted once the listener passes this much of the track — the same
# "you actually listened to it" rule Plex uses, so a skipped-through track does
# not inflate the count.
PLAY_THRESHOLD = 0.5


def _track_id(rating_key: str | None) -> int | None:
    if not rating_key:
        return None
    if rating_key.isdigit():
        # Some players send the bare metadata id.
        return int(rating_key)
    p = parse_key(rating_key)
    return p[1] if p and p[0] == "track" else None


def _accessible(db, user: User, track_id: int) -> bool:
    """Guard for the write endpoints (rate/scrobble): the track must exist and be
    in a library the user was granted. Stops a user from creating state rows for —
    and thereby probing the existence of — tracks outside their libraries."""
    track = db.get(Track, track_id)
    if not track:
        return False
    allowed = accessible_library_ids(db, user)
    return allowed is None or track.library_id in allowed


def _state(db, user_id: int, track_id: int) -> UserTrackState:
    st = db.execute(
        select(UserTrackState).where(
            UserTrackState.user_id == user_id, UserTrackState.track_id == track_id
        )
    ).scalar_one_or_none()
    if st is None:
        st = UserTrackState(user_id=user_id, track_id=track_id, play_count=0,
                            last_played_at=0, view_offset_ms=0)
        db.add(st)
        db.flush()
    return st


# ── live sessions ────────────────────────────────────────────────────────────
# Who is listening to what, right now — Plex's /status/sessions. Kept in memory
# on purpose: it is live state, worthless after a restart, and writing a row per
# heartbeat would hammer SQLite for nothing.
SESSION_TTL = 90  # a client heartbeats every 10s; 90s without one = gone
_SESSIONS: dict[tuple[int, str], dict] = {}


def _touch_session(user: User, session_id: str, track_id: int, state: str,
                   offset_ms: int, device: str) -> None:
    key = (user.id, session_id or "default")
    if state == "stopped":
        _SESSIONS.pop(key, None)
        return
    _SESSIONS[key] = {
        "user_id": user.id,
        "user_name": user.display_name or user.email,
        "track_id": track_id,
        "state": state,
        "offset_ms": offset_ms,
        "device": device or "JLTamp",
        "updated_at": time.time(),
    }


# The open history row per (user, session). Deliberately NOT stored inside
# _SESSIONS: that dict is dropped the moment a client says "stopped", and the
# final heartbeat of a play is exactly when the row must be closed, not lost.
_OPEN_EVENTS: dict[tuple[int, str], tuple[int, int]] = {}  # key -> (event_id, track_id)


def _log_play(db, user: User, track: Track, session_id: str, state: str,
              offset_ms: int, duration_ms: int, device: str) -> None:
    """Append-only history: open a PlayEvent when a session starts a track, then
    extend it while that session keeps playing it. Playing a track three times is
    three rows — that is the point of a log, as opposed to a counter."""
    key = (user.id, session_id or "default")
    now = int(time.time())
    total = duration_ms or track.duration_ms or 0

    open_event = _OPEN_EVENTS.get(key)
    if open_event and open_event[1] == track.id:
        ev = db.get(PlayEvent, open_event[0])
        if ev:
            ev.ended_at = now
            ev.listened_ms = max(ev.listened_ms, offset_ms)
            ev.completed = bool(total) and offset_ms >= total * PLAY_THRESHOLD
            if state == "stopped":
                _OPEN_EVENTS.pop(key, None)
            return
        _OPEN_EVENTS.pop(key, None)

    # A different track on this session (or a fresh one) starts a new row.
    ev = PlayEvent(
        user_id=user.id, track_id=track.id,
        # Credit the track's OWN artist (originalTitle) for the stats — artist_name
        # is the album/grouping artist, which is "Various Artists" on compilations
        # and would otherwise mis-attribute every play on such an album.
        artist_name=(track.orig_artist or track.artist_name), album_title=track.album_title,
        track_title=track.title,
        started_at=now, ended_at=now,
        listened_ms=offset_ms, duration_ms=total,
        completed=bool(total) and offset_ms >= total * PLAY_THRESHOLD,
        device=device or "JLTamp", session_id=session_id or "default",
    )
    db.add(ev)
    db.flush()
    if state != "stopped":
        _OPEN_EVENTS[key] = (ev.id, track.id)


def _live_sessions() -> list[dict]:
    now = time.time()
    for key, s in list(_SESSIONS.items()):
        if now - s["updated_at"] > SESSION_TTL:
            del _SESSIONS[key]
    return list(_SESSIONS.values())


@router.get("/:/timeline")
def timeline(
    request: Request,
    user: User = Depends(require_user),
    ratingKey: str | None = None,
    key: str | None = None,
    state: str = "playing",
    time_ms: int = Query(0, alias="time"),
    duration: int = 0,
    session: str | None = None,
):
    """Playback heartbeat. Records the resume offset, stamps last-played, counts
    the play once the listener crosses PLAY_THRESHOLD, and keeps the live session
    (for /status/sessions) fresh."""
    tid = _track_id(ratingKey) or _track_id((key or "").rsplit("/", 1)[-1])
    if not tid:
        return {"MediaContainer": {"size": 0}}

    db = SessionLocal()
    try:
        track = db.get(Track, tid)
        if not track:
            return {"MediaContainer": {"size": 0}}
        allowed = accessible_library_ids(db, user)
        if allowed is not None and track.library_id not in allowed:
            return {"MediaContainer": {"size": 0}}

        st = _state(db, user.id, tid)
        total = duration or track.duration_ms or 0
        threshold = int(total * PLAY_THRESHOLD) if total else 0

        # Monotonic progress within one playthrough: count exactly when we cross
        # the threshold. Replaying the track rewinds the offset, re-arming it.
        if threshold and st.view_offset_ms < threshold <= time_ms:
            st.play_count += 1

        st.view_offset_ms = max(0, int(time_ms))
        if state in ("playing", "paused", "stopped"):
            st.last_played_at = int(time.time())

        # Prefer the concrete device name the client now sends (e.g. "SM-T500"),
        # then the platform ("Android"/"iOS"/"Web"), then the generic product.
        device = (request.headers.get("X-Plex-Device-Name")
                  or request.headers.get("X-Plex-Platform")
                  or request.headers.get("X-Plex-Product") or "")
        _touch_session(user, session or "", tid, state, max(0, int(time_ms)), device)
        _log_play(db, user, track, session or "", state, max(0, int(time_ms)), total, device)
        db.commit()
        return {"MediaContainer": {"size": 0}}
    finally:
        db.close()


@router.get("/status/sessions")
def sessions(user: User = Depends(require_user)):
    """Who is listening right now. Admins see everyone (Plex's behaviour — it is
    the server dashboard); a normal user only sees their own devices."""
    live = _live_sessions()
    if not user.is_admin:
        live = [s for s in live if s["user_id"] == user.id]

    db = SessionLocal()
    try:
        items = []
        for s in live:
            track = db.get(Track, s["track_id"])
            if not track:
                continue
            d = track_dict(track, state=_state(db, s["user_id"], s["track_id"]))
            d["viewOffset"] = s["offset_ms"]
            d["User"] = {"id": s["user_id"], "title": s["user_name"]}
            d["Player"] = {"state": s["state"], "title": s["device"], "product": "JLTamp"}
            items.append(d)
        return container(items)
    finally:
        db.close()


# ── on deck / continue listening ─────────────────────────────────────────────
# Plex's On Deck: what you were in the middle of. A track counts as "in progress"
# once you are past the intro and not yet at the outro — otherwise every track
# you merely started or finished would clutter the row.
ONDECK_MIN_MS = 15_000
ONDECK_MAX_FRACTION = 0.95


def _on_deck_rows(db, user: User, limit: int):
    allowed = accessible_library_ids(db, user)
    stmt = (
        select(Track, UserTrackState)
        .join(UserTrackState, UserTrackState.track_id == Track.id)
        .where(
            UserTrackState.user_id == user.id,
            UserTrackState.view_offset_ms > ONDECK_MIN_MS,
            UserTrackState.view_offset_ms < Track.duration_ms * ONDECK_MAX_FRACTION,
        )
    )
    if allowed is not None:
        stmt = stmt.where(Track.library_id.in_(allowed or [-1]))
    return db.execute(
        stmt.order_by(UserTrackState.last_played_at.desc()).limit(limit)
    ).all()


@router.get("/library/onDeck")
def on_deck(user: User = Depends(require_user), limit: int = 20):
    db = SessionLocal()
    try:
        return container([track_dict(t, state=st)
                          for t, st in _on_deck_rows(db, user, limit)])
    finally:
        db.close()


# ── ratings (per user, Plex 0-10 stars) ──────────────────────────────────────
@router.get("/:/rate")
@router.put("/:/rate")
def rate(user: User = Depends(require_user), key: str | None = None,
         ratingKey: str | None = None, rating: float = 0.0, identifier: str | None = None):
    tid = _track_id(ratingKey or key)
    if not tid:
        return {"MediaContainer": {"size": 0}}
    db = SessionLocal()
    try:
        if not _accessible(db, user, tid):
            return {"MediaContainer": {"size": 0}}
        st = _state(db, user.id, tid)
        st.rating = max(0.0, min(10.0, float(rating)))
        db.commit()
        return {"MediaContainer": {"size": 0}}
    finally:
        db.close()


@router.get("/:/scrobble")
def scrobble(user: User = Depends(require_user), key: str | None = None,
             ratingKey: str | None = None):
    """Mark a track as played (explicit, from players that scrobble)."""
    tid = _track_id(ratingKey or key)
    if not tid:
        return {"MediaContainer": {"size": 0}}
    db = SessionLocal()
    try:
        if not _accessible(db, user, tid):
            return {"MediaContainer": {"size": 0}}
        st = _state(db, user.id, tid)
        st.play_count += 1
        st.last_played_at = int(time.time())
        st.view_offset_ms = 0
        db.commit()
        return {"MediaContainer": {"size": 0}}
    finally:
        db.close()


@router.get("/:/unscrobble")
def unscrobble(user: User = Depends(require_user), key: str | None = None,
               ratingKey: str | None = None):
    """Mark a track as unplayed again."""
    tid = _track_id(ratingKey or key)
    if not tid:
        return {"MediaContainer": {"size": 0}}
    db = SessionLocal()
    try:
        if not _accessible(db, user, tid):
            return {"MediaContainer": {"size": 0}}
        st = _state(db, user.id, tid)
        st.play_count = 0
        st.view_offset_ms = 0
        db.commit()
        return {"MediaContainer": {"size": 0}}
    finally:
        db.close()


def _history_rows(db, user: User, order, limit: int, type: int):
    """Tracks (type 10) or albums (type 9) from the user's history, restricted to
    the libraries they may see.

    'recent' asks what you listened to, so anything you started counts — a track
    you skipped through still belongs in Recently Played. 'plays' is the play
    counter, so it only lists what actually crossed the play threshold."""
    allowed = accessible_library_ids(db, user)
    played = (
        UserTrackState.last_played_at > 0 if order == "recent"
        else UserTrackState.play_count > 0
    )

    stmt = (
        select(Track, UserTrackState)
        .join(UserTrackState, UserTrackState.track_id == Track.id)
        .where(UserTrackState.user_id == user.id, played)
    )
    if allowed is not None:
        stmt = stmt.where(Track.library_id.in_(allowed or [-1]))

    if type == 9:
        # Roll the per-track state up to the album.
        agg = (
            select(
                Track.album_id,
                func.max(UserTrackState.last_played_at).label("last"),
                func.sum(UserTrackState.play_count).label("plays"),
            )
            .join(UserTrackState, UserTrackState.track_id == Track.id)
            .where(UserTrackState.user_id == user.id, played)
        )
        if allowed is not None:
            agg = agg.where(Track.library_id.in_(allowed or [-1]))
        agg = agg.group_by(Track.album_id)
        agg = agg.order_by(agg.selected_columns["last" if order == "recent" else "plays"].desc())
        rows = db.execute(agg.limit(limit)).all()
        albums = []
        for album_id, _last, _plays in rows:
            a = db.get(Album, album_id)
            if a:
                albums.append(album_dict(a))
        return albums

    stmt = stmt.order_by(
        UserTrackState.last_played_at.desc() if order == "recent"
        else UserTrackState.play_count.desc()
    ).limit(limit)
    return [track_dict(t, state=st) for t, st in db.execute(stmt).all()]


@router.get("/history/recentlyPlayed")
def recently_played(user: User = Depends(require_user), type: int = 10, limit: int = 30):
    db = SessionLocal()
    try:
        return container(_history_rows(db, user, "recent", limit, type))
    finally:
        db.close()


@router.get("/history/mostPlayed")
def most_played(user: User = Depends(require_user), type: int = 10, limit: int = 30):
    db = SessionLocal()
    try:
        return container(_history_rows(db, user, "plays", limit, type))
    finally:
        db.close()


# ── home hubs ────────────────────────────────────────────────────────────────
@router.get("/hubs")
def hubs(user: User = Depends(require_user), count: int = 20):
    """Plex's home screen is hub-driven: one call returns the rows, each with its
    own key so a client can page into it. Empty hubs are dropped, so a brand-new
    user sees only what actually has content."""
    db = SessionLocal()
    try:
        allowed = accessible_library_ids(db, user)
        recent_added = select(Album).order_by(Album.added_at.desc()).limit(count)
        if allowed is not None:
            recent_added = recent_added.where(Album.library_id.in_(allowed or [-1]))

        rows = [
            ("home.continue", "Continue Listening", "/library/onDeck",
             [track_dict(t, state=st) for t, st in _on_deck_rows(db, user, count)]),
            ("home.recentlyPlayed", "Recently Played", "/history/recentlyPlayed",
             _history_rows(db, user, "recent", count, 10)),
            ("home.mostPlayed", "Most Played", "/history/mostPlayed",
             _history_rows(db, user, "plays", count, 10)),
            ("home.recentlyAdded", "Recently Added", "/library/recentlyAdded",
             [album_dict(a) for a in db.execute(recent_added).scalars()]),
        ]
        hubs = [
            {"hubIdentifier": ident, "title": title, "key": key,
             "type": "track" if "/history" in key or "onDeck" in key else "album",
             "size": len(items), "Metadata": items}
            for ident, title, key, items in rows if items
        ]
        return {"MediaContainer": {"size": len(hubs), "Hub": hubs}}
    finally:
        db.close()
