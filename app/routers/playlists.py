"""Playlists (per-user): list / items / create / add / remove / rename / delete
— matching the exact Plex endpoints + `uri=server://{machineId}/.../metadata/{keys}`
mutation format the app uses. Each playlist belongs to the logged-in user.
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy import select, func

from ..deps import require_user, accessible_library_ids
from ..db import SessionLocal
from ..ids import parse_key, playlist_key
from ..models import Playlist, PlaylistItem, Track, User
from ..serializers import track_dict, track_states, playlist_dict, container


def _addable(db, user: User, tid: int) -> Track | None:
    """A track the user may actually add: it exists AND lives in a library they
    were granted. Without this, a user could add tracks from libraries they can't
    see and read their metadata + on-disk file path back out of the playlist."""
    track = db.get(Track, tid)
    if not track:
        return None
    allowed = accessible_library_ids(db, user)
    if allowed is not None and track.library_id not in allowed:
        return None
    return track


router = APIRouter()


def _track_ids_from_uri(uri: str) -> list[int]:
    """server://<id>/com.plexapp.plugins.library/library/metadata/t1,t2,t3 -> [1,2,3]"""
    if not uri or "/library/metadata/" not in uri:
        return []
    tail = uri.rsplit("/library/metadata/", 1)[-1]
    ids: list[int] = []
    for part in tail.split(","):
        p = parse_key(part.strip())
        if p and p[0] == "track":
            ids.append(p[1])
    return ids


def _playlist_id(rk: str) -> int:
    p = parse_key(rk)
    if not p or p[0] != "playlist":
        raise HTTPException(404, "Not a playlist")
    return p[1]


def _owned(db, pid: int, user: User) -> Playlist:
    pl = db.get(Playlist, pid)
    if not pl or pl.user_id != user.id:
        raise HTTPException(404, "Not found")
    return pl


def _leaf_stats(db, playlist_id: int) -> tuple[int, int]:
    row = db.execute(
        select(func.count(PlaylistItem.id), func.coalesce(func.sum(Track.duration_ms), 0))
        .join(Track, Track.id == PlaylistItem.track_id)
        .where(PlaylistItem.playlist_id == playlist_id)
    ).one()
    return int(row[0]), int(row[1])


@router.get("/playlists")
def list_playlists(user: User = Depends(require_user), playlistType: str = "audio"):
    db = SessionLocal()
    try:
        items = []
        for pl in db.execute(select(Playlist).where(Playlist.user_id == user.id)
                             .order_by(Playlist.updated_at.desc())).scalars():
            leaf, dur = _leaf_stats(db, pl.id)
            items.append(playlist_dict(pl, leaf, dur))
        return container(items)
    finally:
        db.close()


@router.get("/playlists/{rk}")
def get_playlist(rk: str, user: User = Depends(require_user)):
    pid = _playlist_id(rk)
    db = SessionLocal()
    try:
        pl = _owned(db, pid, user)
        leaf, dur = _leaf_stats(db, pid)
        return container([playlist_dict(pl, leaf, dur)])
    finally:
        db.close()


@router.get("/playlists/{rk}/items")
def playlist_items(rk: str, user: User = Depends(require_user)):
    pid = _playlist_id(rk)
    db = SessionLocal()
    try:
        _owned(db, pid, user)
        rows = db.execute(
            select(PlaylistItem, Track)
            .join(Track, Track.id == PlaylistItem.track_id)
            .where(PlaylistItem.playlist_id == pid)
            .order_by(PlaylistItem.position)
        ).all()
        states = track_states(db, user.id, [t for _, t in rows])
        items = [track_dict(t, playlist_item_id=pi.id, state=states.get(t.id)) for pi, t in rows]
        return container(items)
    finally:
        db.close()


@router.post("/playlists")
def create_playlist(user: User = Depends(require_user), title: str = "New Playlist",
                    uri: str = "", type: str = "audio", smart: int = 0):
    now = int(time.time())
    db = SessionLocal()
    try:
        pl = Playlist(user_id=user.id, title=title, created_at=now, updated_at=now)
        db.add(pl)
        db.flush()
        pos = 0
        for tid in _track_ids_from_uri(uri):
            if _addable(db, user, tid):
                db.add(PlaylistItem(playlist_id=pl.id, track_id=tid, position=pos))
                pos += 1
        db.commit()
        return container([{"ratingKey": playlist_key(pl.id), "title": pl.title, "type": "playlist"}])
    finally:
        db.close()


@router.put("/playlists/{rk}/items")
def add_items(rk: str, user: User = Depends(require_user), uri: str = ""):
    pid = _playlist_id(rk)
    db = SessionLocal()
    try:
        pl = _owned(db, pid, user)
        maxpos = db.execute(
            select(func.coalesce(func.max(PlaylistItem.position), -1))
            .where(PlaylistItem.playlist_id == pid)
        ).scalar()
        pos = int(maxpos) + 1
        for tid in _track_ids_from_uri(uri):
            if _addable(db, user, tid):
                db.add(PlaylistItem(playlist_id=pid, track_id=tid, position=pos))
                pos += 1
        pl.updated_at = int(time.time())
        db.commit()
        return container([])
    finally:
        db.close()


@router.delete("/playlists/{rk}/items/{item_id}")
def remove_item(rk: str, item_id: int, user: User = Depends(require_user)):
    pid = _playlist_id(rk)
    db = SessionLocal()
    try:
        pl = _owned(db, pid, user)
        it = db.get(PlaylistItem, item_id)
        if it and it.playlist_id == pid:
            db.delete(it)
            pl.updated_at = int(time.time())
            db.commit()
        return container([])
    finally:
        db.close()


@router.put("/playlists/{rk}")
def rename_playlist(rk: str, user: User = Depends(require_user), title: str = ""):
    pid = _playlist_id(rk)
    db = SessionLocal()
    try:
        pl = _owned(db, pid, user)
        if title:
            pl.title = title
            pl.updated_at = int(time.time())
            db.commit()
        return container([])
    finally:
        db.close()


@router.delete("/playlists/{rk}")
def delete_playlist(rk: str, user: User = Depends(require_user)):
    pid = _playlist_id(rk)
    db = SessionLocal()
    try:
        _owned(db, pid, user)
        db.query(PlaylistItem).filter(PlaylistItem.playlist_id == pid).delete()
        pl = db.get(Playlist, pid)
        if pl:
            db.delete(pl)
        db.commit()
        return container([])
    finally:
        db.close()
