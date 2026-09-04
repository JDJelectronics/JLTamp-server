"""Smart playlists — rules over data the server already collects.

No AI and no second machine involved on purpose. These are database questions
over `play_events`, `tracks.added_at` and the user's Liked Songs playlist, all of
which live in this server's SQLite. Sending the library to the Jetson to run a
SELECT would be slower, more fragile, and would make core library features stop
working whenever that box is off.

The AI engine keeps what it is good at — similarity and mood ("more like this",
the endless radio). These two compose nicely: a rule here picks the seeds, the
AI builds a set around them.

Everything is per-user: play history and likes are per-user, and library access
is filtered the same way the rest of the API does it.
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func

from ..db import SessionLocal
from ..deps import require_user, accessible_library_ids
from ..models import Track, PlayEvent, User
from ..serializers import track_dict, container
from .likes import liked_track_ids

router = APIRouter(tags=["smart"])

DAY = 86400

# Each kind is one rule. Keep the list here so the app can render whatever the
# server supports without a matching release.
KINDS = {
    "forgotten": {
        "title": "Forgotten favourites",
        "subtitle": "Liked, but not played in a long time",
    },
    "never-played": {
        "title": "Never played",
        "subtitle": "In your library, never listened to",
    },
    "fresh": {
        "title": "New this month",
        "subtitle": "Added in the last 30 days",
    },
    "most-played": {
        "title": "On repeat",
        "subtitle": "Your most played this year",
    },
}


@router.get("/smart")
def list_smart(user: User = Depends(require_user)):
    """What this server can build, so the client doesn't hardcode the list."""
    return {"kinds": [{"key": k, **v} for k, v in KINDS.items()]}


@router.get("/smart/{kind}")
def smart_playlist(kind: str, limit: int = 100, user: User = Depends(require_user)):
    if kind not in KINDS:
        raise HTTPException(404, "Unknown smart playlist")
    limit = max(1, min(int(limit or 100), 500))
    db = SessionLocal()
    try:
        # None means admin → every library. An EMPTY list means a non-admin
        # with no grants, which really is "nothing to show".
        libs = accessible_library_ids(db, user)
        if libs is not None and len(libs) == 0:
            return container([])

        # Last time this user played each track. LEFT-JOINed rather than
        # filtered, so "never played" is simply "no row here".
        last_played = (
            select(PlayEvent.track_id, func.max(PlayEvent.started_at).label("last_at"),
                   func.count(PlayEvent.id).label("plays"))
            .where(PlayEvent.user_id == user.id)
            .group_by(PlayEvent.track_id)
            .subquery()
        )

        base = select(Track)
        if libs is not None:
            base = base.where(Track.library_id.in_(libs))
        now = int(time.time())

        if kind == "forgotten":
            liked = liked_track_ids(db, user.id)
            if not liked:
                return container([])
            # Liked and either never played or not in the last 90 days. Oldest
            # play first — the ones you have most thoroughly forgotten.
            q = (base.join(last_played, last_played.c.track_id == Track.id, isouter=True)
                     .where(Track.id.in_(liked))
                     .where(func.coalesce(last_played.c.last_at, 0) < now - 90 * DAY)
                     .order_by(func.coalesce(last_played.c.last_at, 0).asc())
                     .limit(limit))

        elif kind == "never-played":
            q = (base.join(last_played, last_played.c.track_id == Track.id, isouter=True)
                     .where(last_played.c.track_id.is_(None))
                     .order_by(func.random())
                     .limit(limit))

        elif kind == "fresh":
            q = (base.where(Track.added_at >= now - 30 * DAY)
                     .order_by(Track.added_at.desc())
                     .limit(limit))

        else:  # most-played
            q = (base.join(last_played, last_played.c.track_id == Track.id)
                     .where(last_played.c.last_at >= now - 365 * DAY)
                     .order_by(last_played.c.plays.desc())
                     .limit(limit))

        tracks = db.execute(q).scalars().all()
        return container([track_dict(t) for t in tracks])
    finally:
        db.close()
