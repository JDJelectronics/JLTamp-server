"""Admin: manage users (invite-only), per-user library access (Plex-style
sharing). Only admins may call these."""
from __future__ import annotations

import time

from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy import select, delete

from ..db import SessionLocal
from ..models import (User, UserLibraryAccess, Library, Session as AuthSession,
                      Playlist, PlaylistItem, LikedTrack, Track, UserTrackState,
                      PlayEvent)
from ..security import new_invite, norm_email
from .. import config, mailer
from ..deps import require_admin

router = APIRouter()


class InviteBody(BaseModel):
    email: str
    display_name: str | None = None
    is_admin: bool = False
    library_ids: list[int] = []


class AccessBody(BaseModel):
    library_ids: list[int]


class FlagBody(BaseModel):
    value: bool


def _is_owner(user: User) -> bool:
    """The account that OWNS this server. It is admin forever — no admin, not even
    a second one that got invited or compromised, may touch its rights."""
    return norm_email(user.email) == norm_email(config.ADMIN_EMAIL)


def _active_admin_count(db) -> int:
    return db.query(User).filter(
        User.is_admin == True, User.is_active == True  # noqa: E712
    ).count()


def _guard_owner(user: User) -> None:
    if _is_owner(user):
        raise HTTPException(status_code=403, detail="De eigenaar van de server kan niet worden gewijzigd")


def _guard_last_admin(db, user: User) -> None:
    """Block removing the final working admin — even a non-owner one — so the
    server can never end up with nobody able to administer it."""
    if user.is_admin and user.is_active and _active_admin_count(db) <= 1:
        raise HTTPException(status_code=400, detail="Dit is de laatste beheerder — die kan niet worden verwijderd of gedegradeerd")


def _user_dict(db, u: User) -> dict:
    grants = [r.library_id for r in db.execute(
        select(UserLibraryAccess).where(UserLibraryAccess.user_id == u.id)).scalars()]
    return {
        "id": u.id,
        "email": u.email,
        "displayName": u.display_name,
        "isAdmin": u.is_admin,
        "isOwner": _is_owner(u),
        "isActive": u.is_active,
        "pending": u.password_hash is None,
        "inviteToken": u.invite_token,
        "libraryIds": grants,
        "createdAt": u.created_at,
    }


def _set_access(db, user_id: int, library_ids: list[int]):
    db.execute(delete(UserLibraryAccess).where(UserLibraryAccess.user_id == user_id))
    valid = {l.id for l in db.execute(select(Library)).scalars()}
    for lid in set(library_ids):
        if lid in valid:
            db.add(UserLibraryAccess(user_id=user_id, library_id=lid))


@router.get("/admin/users")
def list_users(_: User = Depends(require_admin)):
    db = SessionLocal()
    try:
        return {"users": [_user_dict(db, u) for u in
                          db.execute(select(User).order_by(User.id)).scalars()]}
    finally:
        db.close()


@router.post("/admin/users/invite")
def invite_user(body: InviteBody, admin: User = Depends(require_admin)):
    email = norm_email(body.email)
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Ferkeard e-mailadres")
    db = SessionLocal()
    try:
        if db.query(User).filter(User.email == email).first():
            raise HTTPException(status_code=409, detail="E-mail al yn gebrûk")
        token = new_invite()
        # Single-admin server by the owner's rule: invited users are NEVER admin.
        # There is exactly one admin — the owner — and no invite can create a
        # second one. (body.is_admin is intentionally ignored.)
        user = User(email=email, password_hash=None,
                    display_name=(body.display_name or email.split("@")[0]).strip(),
                    is_admin=False, is_active=True,
                    invite_token=token, created_at=int(time.time()))
        db.add(user)
        db.flush()
        _set_access(db, user.id, body.library_ids)
        db.commit()

        # Mail the link ourselves when SMTP is set up. `mailed` tells the admin
        # UI whether it still has to hand the link over manually.
        mailed = mailer.send_invite(
            email, token,
            inviter=(admin.display_name or admin.email),
            server_name=config.SERVER_NAME,
        )
        return {"user": _user_dict(db, user), "inviteToken": token,
                "invitePath": f"/invite/{token}", "mailed": mailed}
    finally:
        db.close()


@router.post("/admin/users/{user_id}/reinvite")
def reinvite(user_id: int, admin: User = Depends(require_admin)):
    db = SessionLocal()
    try:
        user = db.get(User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="Brûker net fûn")
        user.invite_token = new_invite()
        user.password_hash = None  # force them to set a new password
        db.commit()
        mailed = mailer.send_invite(
            user.email, user.invite_token,
            inviter=(admin.display_name or admin.email),
            server_name=config.SERVER_NAME,
        )
        return {"inviteToken": user.invite_token,
                "invitePath": f"/invite/{user.invite_token}", "mailed": mailed}
    finally:
        db.close()


@router.post("/admin/users/{user_id}/access")
def set_access(user_id: int, body: AccessBody, _: User = Depends(require_admin)):
    db = SessionLocal()
    try:
        if not db.get(User, user_id):
            raise HTTPException(status_code=404, detail="Brûker net fûn")
        _set_access(db, user_id, body.library_ids)
        db.commit()
        return {"ok": True, "libraryIds": body.library_ids}
    finally:
        db.close()


@router.post("/admin/users/{user_id}/admin")
def set_admin(user_id: int, body: FlagBody, admin: User = Depends(require_admin)):
    db = SessionLocal()
    try:
        user = db.get(User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="Brûker net fûn")
        if body.value:
            # Promotion: any account may be made admin, but only once it has
            # actually joined (a pending invite has no password yet) and is
            # active — you can't hand admin rights to an account nobody controls.
            if user.password_hash is None or not user.is_active:
                raise HTTPException(status_code=400, detail="Alleen een actieve, aangemelde gebruiker kan beheerder worden")
        if not body.value:
            # Demotion: never the owner, never yourself, never the last admin.
            _guard_owner(user)
            if user.id == admin.id:
                raise HTTPException(status_code=400, detail="Kinsto dysels net degradearje")
            _guard_last_admin(db, user)
        user.is_admin = body.value
        db.commit()
        return {"ok": True}
    finally:
        db.close()


@router.post("/admin/users/{user_id}/active")
def set_active(user_id: int, body: FlagBody, admin: User = Depends(require_admin)):
    db = SessionLocal()
    try:
        user = db.get(User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="Brûker net fûn")
        if not body.value:
            _guard_owner(user)
            if user.id == admin.id:
                raise HTTPException(status_code=400, detail="Kinsto dysels net útskeakelje")
            _guard_last_admin(db, user)
        user.is_active = body.value
        db.commit()
        return {"ok": True}
    finally:
        db.close()


@router.delete("/admin/users/{user_id}")
def delete_user(user_id: int, admin: User = Depends(require_admin)):
    db = SessionLocal()
    try:
        user = db.get(User, user_id)
        if not user:
            raise HTTPException(status_code=404, detail="Brûker net fûn")
        _guard_owner(user)
        if user.id == admin.id:
            raise HTTPException(status_code=400, detail="Kinsto dysels net wiskje")
        _guard_last_admin(db, user)
        db.execute(delete(AuthSession).where(AuthSession.user_id == user_id))
        db.execute(delete(UserLibraryAccess).where(UserLibraryAccess.user_id == user_id))
        db.execute(delete(LikedTrack).where(LikedTrack.user_id == user_id))
        # A playlist's items and every per-user row must go too, or they orphan.
        pl_ids = [p.id for p in db.execute(
            select(Playlist).where(Playlist.user_id == user_id)).scalars()]
        if pl_ids:
            db.execute(delete(PlaylistItem).where(PlaylistItem.playlist_id.in_(pl_ids)))
        db.execute(delete(Playlist).where(Playlist.user_id == user_id))
        db.execute(delete(UserTrackState).where(UserTrackState.user_id == user_id))
        db.execute(delete(PlayEvent).where(PlayEvent.user_id == user_id))
        db.delete(user)
        db.commit()
        return {"ok": True}
    finally:
        db.close()


# ── import liked/rated tracks from an existing Plex server (READ-ONLY) ────────
class PlexImportBody(BaseModel):
    plex_url: str = "http://localhost:32400"
    plex_token: str
    min_rating: float = 1.0


@router.post("/admin/import-plex-likes")
def import_plex_likes(body: PlexImportBody, admin: User = Depends(require_admin)):
    """Import the calling admin's Plex track ratings as JLTamp 'likes'. Only ever
    issues GET requests to Plex — it never writes to or modifies Plex."""
    import json as _json
    import urllib.parse
    import urllib.request

    base = body.plex_url.rstrip("/")

    def pget(path: str):
        req = urllib.request.Request(
            f"{base}{path}",
            headers={"X-Plex-Token": body.plex_token, "Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=20) as r:
            return _json.loads(r.read().decode("utf-8", "replace"))

    try:
        secs = pget("/library/sections").get("MediaContainer", {}).get("Directory", [])
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Kin Plex net berikke: {e}")
    music_keys = [d["key"] for d in secs if d.get("type") == "artist"]

    liked: list[tuple[str, str]] = []
    rq = urllib.parse.quote(f"userRating>={body.min_rating}", safe=">=")
    for k in music_keys:
        start = 0
        while True:
            try:
                data = pget(f"/library/sections/{k}/all?type=10&{rq}"
                            f"&X-Plex-Container-Start={start}&X-Plex-Container-Size=200")
            except Exception:
                break
            mc = data.get("MediaContainer", {})
            tracks = mc.get("Metadata", []) or []
            for t in tracks:
                liked.append((t.get("title", ""), t.get("grandparentTitle", "") or t.get("originalTitle", "")))
            start += len(tracks)
            if len(tracks) < 200 or start >= int(mc.get("totalSize", start) or start):
                break

    def norm(s: str) -> str:
        return (s or "").strip().lower()

    db = SessionLocal()
    try:
        index: dict[tuple, int] = {}
        title_only: dict[str, int] = {}
        for tr in db.execute(select(Track)).scalars():
            nt = norm(tr.title)
            title_only[nt] = tr.id
            for a in {tr.artist_name, tr.orig_artist, tr.album_artist}:
                if a:
                    index[(nt, norm(a))] = tr.id

        matched: set[int] = set()
        for title, artist in liked:
            nt = norm(title)
            tid = index.get((nt, norm(artist))) or title_only.get(nt)
            if tid:
                matched.add(tid)

        existing = {lt.track_id for lt in db.execute(
            select(LikedTrack).where(LikedTrack.user_id == admin.id)).scalars()}
        now = int(time.time())
        added = 0
        for tid in matched:
            if tid not in existing:
                db.add(LikedTrack(user_id=admin.id, track_id=tid, created_at=now))
                added += 1
        db.commit()
        return {"plexRated": len(liked), "matched": len(matched), "imported": added}
    finally:
        db.close()
