"""Per-user 'liked songs'. Simple like/unlike + list, plus a lightweight id list
the client uses to render filled/empty hearts."""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select

from ..deps import require_user, accessible_library_ids
from ..db import SessionLocal
from ..ids import parse_key, track_key
from ..models import LikedTrack, Track, User
from ..serializers import track_dict, track_states, container

router = APIRouter()


def _track_id(rk: str) -> int:
    p = parse_key(rk)
    if not p or p[0] != "track":
        raise HTTPException(404, "Not a track")
    return p[1]


@router.get("/likes")
def list_likes(user: User = Depends(require_user)):
    db = SessionLocal()
    try:
        rows = db.execute(
            select(Track, LikedTrack.created_at)
            .join(LikedTrack, LikedTrack.track_id == Track.id)
            .where(LikedTrack.user_id == user.id)
            .order_by(LikedTrack.created_at.desc())
        ).all()
        tracks = [t for t, _ in rows]
        states = track_states(db, user.id, tracks)
        return container([track_dict(t, state=states.get(t.id)) for t in tracks])
    finally:
        db.close()


@router.get("/likes/ids")
def liked_ids(user: User = Depends(require_user)):
    db = SessionLocal()
    try:
        ids = [track_key(tid) for (tid,) in db.execute(
            select(LikedTrack.track_id).where(LikedTrack.user_id == user.id)).all()]
        return {"likedIds": ids}
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
        exists = db.execute(select(LikedTrack).where(
            LikedTrack.user_id == user.id, LikedTrack.track_id == tid)).scalar_one_or_none()
        if not exists:
            db.add(LikedTrack(user_id=user.id, track_id=tid, created_at=int(time.time())))
            db.commit()
        return {"ok": True, "liked": True}
    finally:
        db.close()


@router.delete("/likes/{rk}")
def unlike(rk: str, user: User = Depends(require_user)):
    tid = _track_id(rk)
    db = SessionLocal()
    try:
        row = db.execute(select(LikedTrack).where(
            LikedTrack.user_id == user.id, LikedTrack.track_id == tid)).scalar_one_or_none()
        if row:
            db.delete(row)
            db.commit()
        return {"ok": True, "liked": False}
    finally:
        db.close()
