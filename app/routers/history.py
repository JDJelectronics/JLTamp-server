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
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from ..db import SessionLocal
from ..deps import require_user, accessible_library_ids
from ..ids import parse_key, track_key
from ..models import Track, Album, User, UserTrackState, PlayEvent
from ..serializers import track_dict, album_dict, container, user_thumb_ref, art_ref

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
    """Get-or-create the per-user state row for a track.

    SELECT-then-INSERT would race: `/:/timeline` is a heartbeat that every client
    sends every 10s, so a phone and a cast (or the web app) reporting the SAME
    account on the SAME track both see "no row yet" and both insert. The loser
    hit `UNIQUE constraint failed: user_track_state.user_id, .track_id` and the
    whole heartbeat 500'd, losing that resume offset and play count.

    So the INSERT is atomic instead: SQLite decides the winner, the loser is
    ignored rather than raised, and both requests then read back the one row.
    """
    sel = select(UserTrackState).where(
        UserTrackState.user_id == user_id, UserTrackState.track_id == track_id
    )
    st = db.execute(sel).scalar_one_or_none()
    if st is None:
        db.execute(
            sqlite_insert(UserTrackState)
            .values(user_id=user_id, track_id=track_id, play_count=0,
                    last_played_at=0, view_offset_ms=0)
            .on_conflict_do_nothing(index_elements=["user_id", "track_id"])
        )
        db.flush()
        st = db.execute(sel).scalar_one()
    return st


# ── live sessions ────────────────────────────────────────────────────────────
# Who is listening to what, right now — Plex's /status/sessions. Kept in memory
# on purpose: it is live state, worthless after a restart, and writing a row per
# heartbeat would hammer SQLite for nothing.
# Een client klopt elke vijf seconden aan; vijfentwintig seconden stilte is dus
# ruim vier gemiste hartslagen — dan speelt daar niets meer. Dit stond op
# negentig, uit de tijd dat de hartslag om de tien seconden kwam, en dat is
# precies het spook dat je zag: de muziek was al lang uit en de kaart bleef nog
# anderhalve minuut staan.
SESSION_TTL = 25
_SESSIONS: dict[tuple[int, str], dict] = {}


# Hoe lang een "playing" zonder voortgang nog geloofd wordt. Ruim boven een
# bufferhikje, ruim onder SESSION_TTL, zodat een vastgelopen client vanzelf
# uit /status/presence valt in plaats van er dagen te blijven staan.
STALL_GRACE = 12


# Per sessie: (positie, wanneer die positie voor het laatst veranderde).
# Overleeft het verlopen van de sessie zelf — zie de uitleg in _touch_session.
_LAST_MOVE: dict[tuple[int, str], tuple[int, float]] = {}


def _touch_session(user: User, session_id: str, track_id: int, state: str,
                   offset_ms: int, device: str) -> None:
    key = (user.id, session_id or "default")
    if state == "stopped":
        _SESSIONS.pop(key, None)
        _LAST_MOVE.pop(key, None)
        return

    # Een client die "playing" roept terwijl de positie niet opschuift, speelt
    # niet. Dat is geen theorie: een JLTamp-tabblad op de achtergrond wordt door
    # de browser afgeknepen tot één timertik per minuut en blijft dan "playing"
    # met een bevroren positie sturen. Gemeten op 2026-08-06: twee hartslagen,
    # 62 seconden uit elkaar, allebei time=151737 — en de eigenaar stond
    # daardoor twee dagen lang bij iedereen in beeld als luisteraar, mét het
    # nummer waar hij toen op stond.
    #
    # We laten zo'n hartslag de sessie niet verversen. Hij verdwijnt dan vanzelf
    # via SESSION_TTL, en komt meteen terug zodra de positie wél beweegt.
    # De laatst BEWOGEN positie wordt apart bijgehouden, niet in _SESSIONS.
    # Dat is het hele punt: een vastgelopen sessie valt na SESSION_TTL weg, en
    # dan zou de volgende bevroren hartslag hem doodleuk opnieuw aanmaken —
    # je knippert dan in en uit beeld in plaats van te verdwijnen.
    now = time.time()
    last = _LAST_MOVE.get(key)
    if last is None or abs(offset_ms - last[0]) >= 1000:
        _LAST_MOVE[key] = (offset_ms, now)
    elif state == "playing" and now - last[1] > STALL_GRACE:
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

    # Eén regel per toestel, niet één per nummer.
    #
    # De app maakt bij elk nieuw nummer een nieuwe sessie-id aan. Zonder dit
    # bleef de vorige hier staan — met de oude titel, en nog steeds als
    # "speelt", want hij verloopt pas na SESSION_TTL. In de lijst hieronder won
    # die oude het van de nieuwe, en dus bleef er een nummer staan dat allang
    # voorbij was. Zodra hetzelfde toestel zich onder een nieuwe sessie meldt,
    # zijn zijn oudere sessies geschiedenis.
    mijn_toestel = device or "JLTamp"
    for k, v in list(_SESSIONS.items()):
        if k == key:
            continue
        if v["user_id"] == user.id and (v.get("device") or "JLTamp") == mijn_toestel:
            _SESSIONS.pop(k, None)
            _LAST_MOVE.pop(k, None)


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
    # Een client die uren geleden voor het laatst bewoog hoeven we niet meer te
    # onthouden; komt hij terug, dan telt hij gewoon weer als nieuw.
    for key, (_pos, seen) in list(_LAST_MOVE.items()):
        if now - seen > 3600 and key not in _SESSIONS:
            del _LAST_MOVE[key]
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


@router.get("/status/presence")
def presence(user: User = Depends(require_user)):
    """Who is listening right now — and deliberately NOT to what.

    /status/sessions is the server dashboard: it carries the track, and a
    normal user only ever sees their own devices there. This is the opposite
    trade: everyone may see everyone, because all it says is that a person has
    music on. What they are playing is theirs.

    That restraint is the feature. A house where you can see the lights are on
    is friendly; one where the neighbours read your playlist is not.
    """
    live = _live_sessions()
    if not live:
        return {"listeners": []}

    db = SessionLocal()
    try:
        people = {u.id: u for u in db.execute(select(User)).scalars()}
        seen: dict[int, dict] = {}
        # Nieuwste eerst. Stond er niet, en dan won bij twee regels van dezelfde
        # persoon degene die toevallig het eerst was aangemaakt — de oudste dus,
        # met het nummer van daarvoor.
        for s in sorted(live, key=lambda r: -r.get("updated_at", 0)):
            u = people.get(s["user_id"])
            if not u:
                continue

            share = (u.share_activity or "presence")
            mine = u.id == user.id
            # "none" means invisible to everyone else. You still see yourself,
            # so the setting never leaves you wondering whether it took.
            if share == "none" and not mine:
                continue

            # One entry per person, not per device: two phones is still one
            # listener, and "playing" beats "paused" if they differ.
            row = seen.get(u.id)
            if row and row["state"] == "playing":
                continue

            entry = {
                "userId": u.id,
                "name": u.display_name or u.email,
                "thumb": user_thumb_ref(u),
                "state": s["state"],
                "isMe": mine,
                "share": share,
                # Wélk toestel. Er staat één regel per persoon, maar met twee
                # toestellen in huis is "the owner is listening" de helft van het
                # antwoord: je wilt weten of dat de telefoon in de keuken is of
                # de tablet in de woonkamer.
                "device": s.get("device") or None,
            }
            # The track rides along ONLY for someone who chose "track" — not
            # even for yourself. Your own entry is filtered out by the UI
            # anyway, so sending it would widen the payload for nothing and
            # weaken the one rule worth stating plainly: nothing in this
            # response says what a person is playing unless they asked for it.
            if share == "track":
                track = db.get(Track, s["track_id"])
                if track:
                    entry["title"] = track.title
                    entry["artist"] = track.orig_artist or track.artist_name
                    entry["album"] = track.album_title
                    # The sleeve, so tapping a face shows a record and not a
                    # line of text. Same art route as everywhere else.
                    entry["thumb_track"] = art_ref(track_key(track.id))
                    entry["ratingKey"] = track_key(track.id)
            seen[u.id] = entry
        # Yourself first, then whoever is actually playing.
        out = sorted(seen.values(), key=lambda r: (not r["isMe"], r["state"] != "playing", r["name"].lower()))
        return {"listeners": out}
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


# ── handoff: pick up where another device left off ───────────────────────────
# On Deck answers "what was I in the middle of". This answers the narrower and
# more useful question a client asks when it opens: "was I *just* listening
# somewhere else?" — the walk from the car to the couch. It is deliberately not
# On Deck's list: a handoff is one track, from one other device, recent enough
# that the listener still remembers where they were.

# Older than this and it is no longer a handoff, it is just history — On Deck
# already covers that, and offering to resume this morning's track at bedtime is
# noise. Four hours covers a commute, a shopping trip and a long walk.
HANDOFF_MAX_AGE_S = 4 * 3600


def _requesting_device(request: Request) -> str:
    return (request.headers.get("X-Plex-Device-Name")
            or request.headers.get("X-Plex-Platform")
            or request.headers.get("X-Plex-Product") or "")


@router.get("/player/handoff")
def handoff(request: Request, user: User = Depends(require_user)):
    """The track this listener was playing on a DIFFERENT device just now, with
    the offset to resume at. Returns {"handoff": null} when there is nothing to
    offer, which is the common case — clients call this on every foreground."""
    here = _requesting_device(request)
    now = int(time.time())

    db = SessionLocal()
    try:
        # Walk back through recent plays rather than taking only the last row:
        # the newest event is often this very device (it heartbeats on open),
        # and a paused-then-resumed track can leave a short trailing row. A
        # handful of rows is enough to find the last *other* device.
        rows = db.execute(
            select(PlayEvent)
            .where(PlayEvent.user_id == user.id,
                   PlayEvent.ended_at >= now - HANDOFF_MAX_AGE_S)
            .order_by(PlayEvent.ended_at.desc())
            .limit(10)
        ).scalars().all()

        allowed = accessible_library_ids(db, user)
        for ev in rows:
            # Same device → there is nothing to hand over; it already knows.
            if here and ev.device and ev.device == here:
                continue
            # A live session elsewhere is not a handoff — the other device is
            # still playing it, and offering to steal it mid-track is worse than
            # saying nothing. Listen Together is the feature for that.
            if any(s["track_id"] == ev.track_id and s["state"] == "playing"
                   and s["device"] != here for s in _live_sessions()):
                continue

            track = db.get(Track, ev.track_id)
            if not track:
                continue
            if allowed is not None and track.library_id not in allowed:
                continue

            st = db.execute(
                select(UserTrackState).where(UserTrackState.user_id == user.id,
                                             UserTrackState.track_id == track.id)
            ).scalar_one_or_none()
            offset = int(st.view_offset_ms) if st else 0
            total = track.duration_ms or ev.duration_ms or 0

            # Same window On Deck uses: past the intro, short of the outro.
            # Outside it there is no meaningful point to resume at.
            if offset <= ONDECK_MIN_MS:
                continue
            if total and offset >= total * ONDECK_MAX_FRACTION:
                continue

            return {"handoff": {
                **track_dict(track, state=st),
                "offsetMs": offset,
                "device": ev.device or "",
                "endedAt": int(ev.ended_at),
                "agoSec": max(0, now - int(ev.ended_at)),
            }}

        return {"handoff": None}
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
