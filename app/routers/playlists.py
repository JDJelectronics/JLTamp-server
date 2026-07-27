"""Playlists (per-user, optionally SHARED): list / items / create / add / remove /
rename / delete — matching the exact Plex endpoints + `uri=server://{machineId}/
.../metadata/{keys}` mutation format the app uses.

Each playlist has one OWNER (Playlist.user_id). The owner can share it: that mints
a short `share_code`; anyone who joins with that code becomes a MEMBER and may
add/remove tracks (collaborative). A live WebSocket room (`/playlists/{rk}/ws`)
pushes a `changed` ping to every viewer the moment anyone edits, so collaborators
see each other's edits immediately. Track attribution ("added by") rides along via
PlaylistItem.added_by.
"""
from __future__ import annotations

import asyncio
import secrets
import time

from fastapi import APIRouter, Depends, HTTPException, WebSocket, WebSocketDisconnect
from sqlalchemy import select, func, or_

from ..deps import require_user, accessible_library_ids, _user_for_token
from ..db import SessionLocal
from ..ids import parse_key, playlist_key
from ..models import Playlist, PlaylistItem, PlaylistMember, Track, User
from ..serializers import track_dict, track_states, playlist_dict, container, user_thumb_ref


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

# ── unambiguous short share codes (no 0/O/1/I/L) ─────────────────────────────
_CODE_ALPHABET = "ABCDEFGHJKMNPQRSTUVWXYZ23456789"


def _gen_share_code(db) -> str:
    for _ in range(20):
        code = "".join(secrets.choice(_CODE_ALPHABET) for _ in range(6))
        if not db.execute(select(Playlist.id).where(Playlist.share_code == code)).first():
            return code
    return secrets.token_hex(4).upper()


# ── live collaboration rooms (in-memory; ephemeral, like Listen Together) ─────
class _Rooms:
    """playlist_id -> set of connected WebSockets. Broadcasts a tiny `changed`
    ping so every viewer refetches. The REST mutations run in FastAPI's sync
    threadpool, so they can't await a broadcast directly; they hand the coroutine
    to the event loop we captured when the first socket connected."""

    def __init__(self) -> None:
        self._rooms: dict[int, set[WebSocket]] = {}
        self._loop: asyncio.AbstractEventLoop | None = None

    def add(self, pid: int, ws: WebSocket) -> None:
        self._loop = asyncio.get_running_loop()
        self._rooms.setdefault(pid, set()).add(ws)

    def remove(self, pid: int, ws: WebSocket) -> None:
        room = self._rooms.get(pid)
        if room:
            room.discard(ws)
            if not room:
                self._rooms.pop(pid, None)

    async def _send_all(self, pid: int, msg: dict) -> None:
        for ws in list(self._rooms.get(pid, ())):
            try:
                await ws.send_json(msg)
            except Exception:
                self.remove(pid, ws)

    def notify(self, pid: int, msg: dict) -> None:
        """Thread-safe: callable from a sync REST handler. No-op if nobody's in
        the room (viewers refetch fresh on their next open anyway)."""
        loop = self._loop
        if not loop or pid not in self._rooms:
            return
        try:
            asyncio.run_coroutine_threadsafe(self._send_all(pid, msg), loop)
        except Exception:
            pass


ROOMS = _Rooms()


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


def _is_member(db, pid: int, user_id: int) -> bool:
    return bool(db.execute(
        select(PlaylistMember.id)
        .where(PlaylistMember.playlist_id == pid, PlaylistMember.user_id == user_id)
    ).first())


def _may_access(db, pid: int, user: User) -> Playlist:
    """Owner OR a joined member may read + edit (collaborative). Others get 404."""
    pl = db.get(Playlist, pid)
    if not pl:
        raise HTTPException(404, "Not found")
    if pl.user_id != user.id and not _is_member(db, pid, user.id):
        raise HTTPException(404, "Not found")
    return pl


def _owned(db, pid: int, user: User) -> Playlist:
    """Owner-only (rename / delete / share). Members can't rename or delete."""
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


def _member_ids(db, pid: int) -> list[int]:
    return [r[0] for r in db.execute(
        select(PlaylistMember.user_id).where(PlaylistMember.playlist_id == pid)
    ).all()]


def _user_brief(u: User | None, is_owner: bool = False) -> dict | None:
    if not u:
        return None
    return {
        "userId": u.id,
        # Never fall back to the email address — that would surface a collaborator's
        # email to other members. Use a neutral label instead.
        "name": u.display_name or f"User {u.id}",
        "thumb": user_thumb_ref(u),
        "isOwner": is_owner,
    }


def _members_payload(db, pl: Playlist) -> list[dict]:
    """Owner first, then joined members — for avatar strips."""
    out: list[dict] = []
    owner = db.get(User, pl.user_id)
    ob = _user_brief(owner, is_owner=True)
    if ob:
        out.append(ob)
    for uid in _member_ids(db, pl.id):
        u = db.get(User, uid)
        b = _user_brief(u, is_owner=False)
        if b:
            out.append(b)
    return out


def _pl_dict(db, pl: Playlist) -> dict:
    """playlist_dict + shared/collab flags the client uses to badge & wire live."""
    leaf, dur = _leaf_stats(db, pl.id)
    d = playlist_dict(pl, leaf, dur)
    member_ids = _member_ids(db, pl.id)
    d["shared"] = bool(pl.share_code)
    d["memberCount"] = (len(member_ids) + 1) if pl.share_code else 0
    d["ownerId"] = pl.user_id
    return d


@router.get("/playlists")
def list_playlists(user: User = Depends(require_user), playlistType: str = "audio"):
    db = SessionLocal()
    try:
        # Owned OR joined-as-member.
        member_pl_ids = select(PlaylistMember.playlist_id).where(
            PlaylistMember.user_id == user.id
        )
        items = []
        for pl in db.execute(
            select(Playlist)
            .where(or_(Playlist.user_id == user.id, Playlist.id.in_(member_pl_ids)))
            .order_by(Playlist.updated_at.desc())
        ).scalars():
            items.append(_pl_dict(db, pl))
        return container(items)
    finally:
        db.close()


@router.get("/playlists/{rk}")
def get_playlist(rk: str, user: User = Depends(require_user)):
    pid = _playlist_id(rk)
    db = SessionLocal()
    try:
        pl = _may_access(db, pid, user)
        return container([_pl_dict(db, pl)])
    finally:
        db.close()


@router.get("/playlists/{rk}/items")
def playlist_items(rk: str, user: User = Depends(require_user)):
    pid = _playlist_id(rk)
    db = SessionLocal()
    try:
        _may_access(db, pid, user)
        rows = db.execute(
            select(PlaylistItem, Track)
            .join(Track, Track.id == PlaylistItem.track_id)
            .where(PlaylistItem.playlist_id == pid)
            .order_by(PlaylistItem.position)
        ).all()
        states = track_states(db, user.id, [t for _, t in rows])
        # Batch-load the "added by" users for attribution.
        adder_ids = {pi.added_by for pi, _ in rows if pi.added_by}
        adders = {u.id: u for u in db.execute(
            select(User).where(User.id.in_(adder_ids))
        ).scalars()} if adder_ids else {}
        items = []
        for pi, t in rows:
            d = track_dict(t, playlist_item_id=pi.id, state=states.get(t.id))
            if pi.added_by and pi.added_by in adders:
                d["addedBy"] = _user_brief(adders[pi.added_by])
            items.append(d)
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
                db.add(PlaylistItem(playlist_id=pl.id, track_id=tid, position=pos,
                                    added_by=user.id))
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
        pl = _may_access(db, pid, user)
        maxpos = db.execute(
            select(func.coalesce(func.max(PlaylistItem.position), -1))
            .where(PlaylistItem.playlist_id == pid)
        ).scalar()
        pos = int(maxpos) + 1
        added = 0
        for tid in _track_ids_from_uri(uri):
            if _addable(db, user, tid):
                db.add(PlaylistItem(playlist_id=pid, track_id=tid, position=pos,
                                    added_by=user.id))
                pos += 1
                added += 1
        pl.updated_at = int(time.time())
        db.commit()
        if added and pl.share_code:
            ROOMS.notify(pid, {"t": "changed", "by": user.id})
        return container([])
    finally:
        db.close()


@router.delete("/playlists/{rk}/items/{item_id}")
def remove_item(rk: str, item_id: int, user: User = Depends(require_user)):
    pid = _playlist_id(rk)
    db = SessionLocal()
    try:
        pl = _may_access(db, pid, user)
        it = db.get(PlaylistItem, item_id)
        removed = False
        if it and it.playlist_id == pid:
            db.delete(it)
            pl.updated_at = int(time.time())
            db.commit()
            removed = True
        if removed and pl.share_code:
            ROOMS.notify(pid, {"t": "changed", "by": user.id})
        return container([])
    finally:
        db.close()


@router.put("/playlists/{rk}")
def rename_playlist(rk: str, user: User = Depends(require_user), title: str = ""):
    pid = _playlist_id(rk)
    db = SessionLocal()
    try:
        pl = _owned(db, pid, user)   # only the owner renames
        if title:
            pl.title = title
            pl.updated_at = int(time.time())
            db.commit()
            if pl.share_code:
                ROOMS.notify(pid, {"t": "changed", "by": user.id})
        return container([])
    finally:
        db.close()


@router.delete("/playlists/{rk}")
def delete_playlist(rk: str, user: User = Depends(require_user)):
    pid = _playlist_id(rk)
    db = SessionLocal()
    try:
        pl = db.get(Playlist, pid)
        if not pl:
            return container([])
        if pl.user_id == user.id:
            # Owner deletes the whole playlist for everyone.
            db.query(PlaylistItem).filter(PlaylistItem.playlist_id == pid).delete()
            db.query(PlaylistMember).filter(PlaylistMember.playlist_id == pid).delete()
            db.delete(pl)
            db.commit()
            ROOMS.notify(pid, {"t": "ended"})
        elif _is_member(db, pid, user.id):
            # A member "deleting" just leaves the shared playlist.
            db.query(PlaylistMember).filter(
                PlaylistMember.playlist_id == pid, PlaylistMember.user_id == user.id
            ).delete()
            db.commit()
            ROOMS.notify(pid, {"t": "members"})
        return container([])
    finally:
        db.close()


# ── sharing / collaboration ──────────────────────────────────────────────────
@router.post("/playlists/{rk}/share")
def share_playlist(rk: str, user: User = Depends(require_user)):
    """Mint (or return the existing) share code for a playlist. Owner only."""
    pid = _playlist_id(rk)
    db = SessionLocal()
    try:
        pl = _owned(db, pid, user)
        if not pl.share_code:
            pl.share_code = _gen_share_code(db)
            db.commit()
        return {"code": pl.share_code, "ratingKey": playlist_key(pl.id), "title": pl.title}
    finally:
        db.close()


@router.get("/playlists/{rk}/members")
def playlist_members(rk: str, user: User = Depends(require_user)):
    pid = _playlist_id(rk)
    db = SessionLocal()
    try:
        pl = _may_access(db, pid, user)
        return {"members": _members_payload(db, pl)}
    finally:
        db.close()


@router.post("/playlists/join/{code}")
def join_playlist(code: str, user: User = Depends(require_user)):
    """Join a shared playlist by its code → become a member. Idempotent."""
    db = SessionLocal()
    try:
        pl = db.execute(
            select(Playlist).where(Playlist.share_code == code.strip().upper())
        ).scalars().first()
        if not pl:
            raise HTTPException(404, "No such shared playlist")
        if pl.user_id != user.id and not _is_member(db, pl.id, user.id):
            db.add(PlaylistMember(playlist_id=pl.id, user_id=user.id,
                                  joined_at=int(time.time())))
            db.commit()
            ROOMS.notify(pl.id, {"t": "members"})
        return {"ratingKey": playlist_key(pl.id), "title": pl.title, "ownerId": pl.user_id}
    finally:
        db.close()


# ── live collaboration socket ────────────────────────────────────────────────
@router.websocket("/playlists/{rk}/ws")
async def playlist_ws(websocket: WebSocket, rk: str):
    token = websocket.query_params.get("X-Plex-Token")
    user = _user_for_token(token)
    try:
        pid = _playlist_id(rk)
    except HTTPException:
        await websocket.close(code=4004)
        return
    if not user:
        await websocket.close(code=4001)
        return
    # Must be owner or member to listen in.
    db = SessionLocal()
    try:
        pl = db.get(Playlist, pid)
        allowed = bool(pl) and (pl.user_id == user.id or _is_member(db, pid, user.id))
        members = _members_payload(db, pl) if pl else []
    finally:
        db.close()
    if not allowed:
        await websocket.close(code=4003)
        return

    await websocket.accept()
    ROOMS.add(pid, websocket)
    try:
        await websocket.send_json({"t": "state", "you": user.id, "members": members})
        while True:
            # We don't need client messages; just keep the socket open (and drain
            # any pings) until it drops.
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        ROOMS.remove(pid, websocket)
