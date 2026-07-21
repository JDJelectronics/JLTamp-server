"""Listening statistics — the Tautulli side of the server.

Tautulli watches a Plex server from the outside and rebuilds history by polling.
Here the server IS the source: every timeline heartbeat already writes a PlayEvent
(see history.py), so the stats are exact rather than sampled — no polling, no
missed plays between polls.

Scope rules, everywhere in this router: an admin may look at the whole server (and
at one user via ?user_id=), a normal user only ever sees their own listening.
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func

from ..db import SessionLocal
from ..deps import require_user
from ..ids import track_key
from ..models import PlayEvent, User

router = APIRouter()

# What counts as "a play" in the stats: the same threshold history.py uses to
# bump a play count, so the numbers here agree with the ones on the track.
from .history import PLAY_THRESHOLD  # noqa: E402


def _scope_user(db, user: User, user_id: int | None) -> int | None:
    """Which user's events to read. None = the whole server (admin only)."""
    if user.is_admin:
        return user_id  # None → everyone
    if user_id is not None and user_id != user.id:
        raise HTTPException(403, "Not your listening history")
    return user.id


def _since(days: int) -> int:
    return int(time.time()) - max(1, days) * 86400


def _events(scoped_user: int | None, days: int):
    stmt = select(PlayEvent).where(PlayEvent.started_at >= _since(days))
    if scoped_user is not None:
        stmt = stmt.where(PlayEvent.user_id == scoped_user)
    return stmt


def _counted(stmt):
    """Only real listens — a 4-second skip is not a play."""
    return stmt.where(PlayEvent.completed.is_(True))


@router.get("/stats/history")
def history_log(user: User = Depends(require_user), user_id: int | None = None,
                days: int = 30, limit: int = 100, start: int = 0):
    """The raw log: what was played, by whom, when, for how long."""
    db = SessionLocal()
    try:
        scoped = _scope_user(db, user, user_id)
        stmt = _events(scoped, days).order_by(PlayEvent.started_at.desc()) \
            .offset(max(0, start)).limit(min(500, limit))
        rows = list(db.execute(stmt).scalars())
        total = db.execute(
            select(func.count(PlayEvent.id)).where(
                PlayEvent.started_at >= _since(days),
                *([PlayEvent.user_id == scoped] if scoped is not None else []),
            )
        ).scalar() or 0

        names = {u.id: (u.display_name or u.email) for u in db.execute(select(User)).scalars()}
        items = [{
            "id": ev.id,
            "ratingKey": track_key(ev.track_id),
            "title": ev.track_title,
            "artist": ev.artist_name,
            "album": ev.album_title,
            "user": names.get(ev.user_id, "?"),
            "userId": ev.user_id,
            "device": ev.device,
            "startedAt": ev.started_at,
            "listenedMs": ev.listened_ms,
            "durationMs": ev.duration_ms,
            "completed": bool(ev.completed),
            "percent": round(100 * ev.listened_ms / ev.duration_ms) if ev.duration_ms else 0,
        } for ev in rows]
        return {"MediaContainer": {"size": len(items), "totalSize": total, "History": items}}
    finally:
        db.close()


@router.get("/stats/summary")
def summary(user: User = Depends(require_user), user_id: int | None = None, days: int = 30):
    """Headline numbers for the dashboard."""
    db = SessionLocal()
    try:
        scoped = _scope_user(db, user, user_id)
        base = _counted(_events(scoped, days)).subquery()

        plays = db.execute(select(func.count()).select_from(base)).scalar() or 0
        listened = db.execute(select(func.sum(base.c.listened_ms))).scalar() or 0
        tracks = db.execute(select(func.count(func.distinct(base.c.track_id)))).scalar() or 0
        artists = db.execute(select(func.count(func.distinct(base.c.artist_name)))).scalar() or 0
        users = db.execute(select(func.count(func.distinct(base.c.user_id)))).scalar() or 0

        return {"days": days, "plays": plays, "listenedMs": int(listened),
                "listenedHours": round(listened / 3_600_000, 1),
                "uniqueTracks": tracks, "uniqueArtists": artists, "users": users}
    finally:
        db.close()


@router.get("/stats/top")
def top(user: User = Depends(require_user), metric: str = "artists",
        user_id: int | None = None, days: int = 30, limit: int = 10):
    """Top artists / albums / tracks / users / devices, by number of plays."""
    columns = {
        "artists": PlayEvent.artist_name,
        "albums": PlayEvent.album_title,
        "tracks": PlayEvent.track_title,
        "devices": PlayEvent.device,
        "users": PlayEvent.user_id,
    }
    col = columns.get(metric)
    if col is None:
        raise HTTPException(400, f"Unknown metric: {metric}")

    db = SessionLocal()
    try:
        scoped = _scope_user(db, user, user_id)
        if metric == "users" and not user.is_admin:
            raise HTTPException(403, "Admin only")

        stmt = (
            select(col, func.count(PlayEvent.id).label("plays"),
                   func.sum(PlayEvent.listened_ms).label("ms"))
            .where(PlayEvent.started_at >= _since(days), PlayEvent.completed.is_(True))
            .group_by(col).order_by(func.count(PlayEvent.id).desc()).limit(min(50, limit))
        )
        if scoped is not None:
            stmt = stmt.where(PlayEvent.user_id == scoped)

        rows = db.execute(stmt).all()
        names = {}
        if metric == "users":
            names = {u.id: (u.display_name or u.email) for u in db.execute(select(User)).scalars()}

        items = [{
            "title": names.get(value, str(value)) if metric == "users" else (value or "Unknown"),
            "plays": plays,
            "listenedMs": int(ms or 0),
        } for value, plays, ms in rows if value not in (None, "")]
        return {"metric": metric, "days": days, "size": len(items), "Top": items}
    finally:
        db.close()


@router.get("/stats/users")
def users_overview(user: User = Depends(require_user), days: int = 30):
    """Admin-only per-user leaderboard: who listened, how much, when last. Powers
    the dashboard's user picker and the 'all users' overview. A normal user gets
    only their own row (so the endpoint is safe to call from any client)."""
    db = SessionLocal()
    try:
        counts = (
            select(
                PlayEvent.user_id,
                func.count(PlayEvent.id).label("plays"),
                func.sum(PlayEvent.listened_ms).label("ms"),
                func.max(PlayEvent.started_at).label("last"),
            )
            .where(PlayEvent.started_at >= _since(days), PlayEvent.completed.is_(True))
            .group_by(PlayEvent.user_id)
        )
        if not user.is_admin:
            counts = counts.where(PlayEvent.user_id == user.id)
        stats = {row.user_id: row for row in db.execute(counts).all()}

        # Admins see every user (even those with zero plays); a normal user sees
        # only themselves.
        if user.is_admin:
            people = db.execute(select(User).where(User.is_active.is_(True))).scalars()
        else:
            people = [user]

        items = []
        for u in people:
            row = stats.get(u.id)
            items.append({
                "id": u.id,
                "name": u.display_name or u.email,
                "email": u.email if user.is_admin else None,
                "isAdmin": u.is_admin,
                "plays": int(row.plays) if row else 0,
                "listenedMs": int(row.ms or 0) if row else 0,
                "lastPlayedAt": int(row.last) if row else 0,
            })
        items.sort(key=lambda x: x["plays"], reverse=True)
        return {"days": days, "size": len(items), "Users": items}
    finally:
        db.close()


@router.get("/stats/activity")
def activity(user: User = Depends(require_user), user_id: int | None = None, days: int = 30):
    """Plays and listening time per day — the dashboard's chart. Days with no
    listening are returned as zeroes so the chart has no holes in it."""
    db = SessionLocal()
    try:
        scoped = _scope_user(db, user, user_id)
        stmt = (
            select(
                func.strftime("%Y-%m-%d", PlayEvent.started_at, "unixepoch").label("day"),
                func.count(PlayEvent.id),
                func.sum(PlayEvent.listened_ms),
            )
            .where(PlayEvent.started_at >= _since(days), PlayEvent.completed.is_(True))
            .group_by("day")
        )
        if scoped is not None:
            stmt = stmt.where(PlayEvent.user_id == scoped)
        found = {day: (plays, int(ms or 0)) for day, plays, ms in db.execute(stmt).all()}

        start = _since(days)
        series = []
        for i in range(days + 1):
            day = time.strftime("%Y-%m-%d", time.gmtime(start + i * 86400))
            plays, ms = found.get(day, (0, 0))
            series.append({"day": day, "plays": plays, "listenedMs": ms})
        return {"days": days, "size": len(series), "Activity": series}
    finally:
        db.close()


@router.delete("/stats")
def clear_stats(user: User = Depends(require_user)):
    """Wipe the caller's own listening history.

    Every /stats view (summary, top, activity, history) is derived from
    PlayEvent, so deleting the caller's rows empties their Statistics tab.
    Deliberately scoped to the caller only — it never removes another user's
    history, not even for an admin — and leaves resume points, likes and
    playlists untouched.
    """
    db = SessionLocal()
    try:
        n = db.query(PlayEvent).filter(PlayEvent.user_id == user.id).delete()
        db.commit()
        return {"deleted": int(n or 0)}
    finally:
        db.close()
