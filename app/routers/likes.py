"""Per-user 'liked songs'. Simple like/unlike + list, plus a lightweight id list
the client uses to render filled/empty hearts.

**Source of truth: the user's playlist titled "Liked Songs".**

The app has always rendered liked songs from that playlist (Plex-compatible), not
from the `liked_tracks` table — so the table and the playlist were two separate
records of the same thing, with nothing keeping them equal. They drifted: on
2026-07-27 only 26 of the owner's 64 likes were in both, and one user had 8 likes
in the app and 0 in the table. That mattered because the AI engine scores on
`/likes/ids`, i.e. it was recommending against a stale set.

So these endpoints now read AND write the playlist. `liked_tracks` is still
maintained here as a mirror (cheap, and account deletion already prunes it), but
it is no longer what anyone reads — never reintroduce a read from it without
reconciling first.
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select

from ..deps import require_user, accessible_library_ids
from ..db import SessionLocal
from ..ids import parse_key, track_key
from ..models import LikedTrack, Playlist, PlaylistItem, Track, User
from ..serializers import track_dict, track_states, container

router = APIRouter()

LIKED_PLAYLIST_TITLE = "Liked Songs"


def _track_id(rk: str) -> int:
    p = parse_key(rk)
    if not p or p[0] != "track":
        raise HTTPException(404, "Not a track")
    return p[1]


def liked_playlist(db, user_id: int, create: bool = False) -> Playlist | None:
    """The user's "Liked Songs" playlist — the one the app reads and writes.

    Matched on title because that is how the app finds it too; there is no flag
    on the row. `create=True` only for write paths, so a read never leaves an
    empty playlist behind for someone who has never liked anything.
    """
    pl = db.execute(
        select(Playlist).where(Playlist.user_id == user_id,
                               Playlist.title == LIKED_PLAYLIST_TITLE)
    ).scalars().first()
    if pl is None and create:
        now = int(time.time())
        pl = Playlist(user_id=user_id, title=LIKED_PLAYLIST_TITLE,
                      created_at=now, updated_at=now)
        db.add(pl)
        db.flush()
    return pl


def liked_track_ids(db, user_id: int) -> list[int]:
    """Track ids in the user's Liked Songs playlist, newest first.

    Newest first = highest position first: the app appends, so position order is
    the order things were liked in. That keeps the old `created_at DESC` feel of
    this endpoint without needing a timestamp the playlist doesn't carry.
    """
    pl = liked_playlist(db, user_id)
    if not pl:
        return []
    return [tid for (tid,) in db.execute(
        select(PlaylistItem.track_id)
        .where(PlaylistItem.playlist_id == pl.id)
        .order_by(PlaylistItem.position.desc())
    ).all()]


def _mirror(db, user_id: int, track_id: int, liked: bool) -> None:
    """Keep the legacy `liked_tracks` row in step with the playlist. Nothing reads
    it any more; it is kept so the table does not quietly rot into nonsense."""
    row = db.execute(select(LikedTrack).where(
        LikedTrack.user_id == user_id, LikedTrack.track_id == track_id)).scalar_one_or_none()
    if liked and not row:
        db.add(LikedTrack(user_id=user_id, track_id=track_id, created_at=int(time.time())))
    elif row and not liked:
        db.delete(row)


@router.get("/likes")
def list_likes(user: User = Depends(require_user)):
    db = SessionLocal()
    try:
        ids = liked_track_ids(db, user.id)
        if not ids:
            return container([])
        rows = {t.id: t for t in db.execute(
            select(Track).where(Track.id.in_(ids))).scalars()}
        # Preserve the playlist order, and skip ids whose track has since been
        # removed from disk — the playlist can outlive a rescan.
        tracks = [rows[i] for i in ids if i in rows]
        states = track_states(db, user.id, tracks)
        return container([track_dict(t, state=states.get(t.id)) for t in tracks])
    finally:
        db.close()


@router.get("/likes/ids")
def liked_ids(user: User = Depends(require_user)):
    db = SessionLocal()
    try:
        return {"likedIds": [track_key(tid) for tid in liked_track_ids(db, user.id)]}
    finally:
        db.close()


@router.put("/likes/{rk}")
def like(rk: str, user: User = Depends(require_user)):
    tid = _track_id(rk)
    db = SessionLocal()
    try:
        t = db.get(Track, tid)
        if not t:
            raise HTTPException(404, "Track not found")
        allowed = accessible_library_ids(db, user)
        if allowed is not None and t.library_id not in allowed:
            raise HTTPException(403, "Gjin tagong")
        pl = liked_playlist(db, user.id, create=True)
        exists = db.execute(select(PlaylistItem).where(
            PlaylistItem.playlist_id == pl.id,
            PlaylistItem.track_id == tid)).scalars().first()
        if not exists:
            nxt = db.execute(
                select(PlaylistItem.position)
                .where(PlaylistItem.playlist_id == pl.id)
                .order_by(PlaylistItem.position.desc())
            ).scalars().first()
            db.add(PlaylistItem(playlist_id=pl.id, track_id=tid,
                                position=(nxt + 1) if nxt is not None else 0,
                                added_by=user.id))
            pl.updated_at = int(time.time())
            _mirror(db, user.id, tid, True)
            db.commit()
        return {"ok": True, "liked": True}
    finally:
        db.close()


@router.delete("/likes/{rk}")
def unlike(rk: str, user: User = Depends(require_user)):
    tid = _track_id(rk)
    db = SessionLocal()
    try:
        pl = liked_playlist(db, user.id)
        if pl:
            rows = db.execute(select(PlaylistItem).where(
                PlaylistItem.playlist_id == pl.id,
                PlaylistItem.track_id == tid)).scalars().all()
            for row in rows:
                db.delete(row)
            if rows:
                pl.updated_at = int(time.time())
        _mirror(db, user.id, tid, False)
        db.commit()
        return {"ok": True, "liked": False}
    finally:
        db.close()
