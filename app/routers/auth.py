"""Login + invites + server discovery (multi-user, invite-only).

Login accepts either an email or the legacy username ("admin") in the
`username` field, so the existing Android app (which posts {username,password})
keeps working while the web app logs in by email. A successful login issues a
per-user bearer token (the `X-Plex-Token`).
"""
from __future__ import annotations

import time

from fastapi import APIRouter, Request, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy import select, delete

from .. import config
from ..db import SessionLocal
from ..models import (User, UserLibraryAccess, Session as AuthSession, Playlist,
                      PlaylistItem, LikedTrack, UserTrackState, PlayEvent)
from ..security import verify_password, hash_password, norm_email
from ..deps import create_session, require_user
from ..security import new_invite
from .. import mailer

router = APIRouter()


# ── brute-force protection ───────────────────────────────────────────────────
# The server is reachable from the public internet, so the login
# endpoint is exposed to anyone who wants to guess passwords at it. Unlimited
# tries turn any weak password into a matter of time.
#
# Failures are counted per client IP *and* per account, so neither a single host
# hammering many accounts nor a botnet hammering one account gets a free run.
# In-memory on purpose: this is throttling state, not data — losing it on restart
# costs an attacker a restart's worth of delay and costs us no correctness.
FAIL_WINDOW = 900        # failures are counted over 15 minutes
# The IP limit is the real defence: an attacker guesses from somewhere, and 5
# tries per quarter of an hour makes guessing hopeless. The per-ACCOUNT limit is
# deliberately much higher, because a strict one is a weapon *against* the user:
# anyone could lock a user out of their own server by typing their password wrong
# 5 times from a café. It exists only to stop a botnet spreading tries over many
# IPs against one account.
FAIL_LIMIT_IP = 5
FAIL_LIMIT_ACCOUNT = 30
_FAILS: dict[str, list[float]] = {}


def _client_ip(request: Request) -> str:
    # Behind our Caddy proxy the true client is the LAST hop in X-Forwarded-For:
    # a proxy APPENDS the peer it saw, so a client that pre-seeds the header can
    # only add entries to the LEFT — the rightmost is the one Caddy wrote and is
    # the only value it cannot control. (Caddy also drops inbound XFF by default,
    # so normally there is exactly one entry.) Taking the leftmost, as before,
    # let an attacker rotate a fake IP to defeat the per-IP throttle.
    fwd = request.headers.get("X-Forwarded-For", "")
    if fwd:
        return fwd.split(",")[-1].strip()
    return request.client.host if request.client else "?"


def _recent(key: str) -> list[float]:
    now = time.time()
    hits = [t for t in _FAILS.get(key, []) if now - t < FAIL_WINDOW]
    if hits:
        _FAILS[key] = hits
    else:
        _FAILS.pop(key, None)
    return hits


def _check_not_locked(keys: dict[str, int]) -> None:
    for key, limit in keys.items():
        hits = _recent(key)
        if len(hits) >= limit:
            retry = int(FAIL_WINDOW - (time.time() - hits[0]))
            raise HTTPException(
                status_code=429,
                detail="Te veel mislukte inlogpogingen. Probeer het later opnieuw.",
                headers={"Retry-After": str(max(1, retry))},
            )


def _record_failure(keys: dict[str, int]) -> None:
    now = time.time()
    for key in keys:
        _FAILS.setdefault(key, []).append(now)


def _clear_failures(keys: dict[str, int]) -> None:
    for key in keys:
        _FAILS.pop(key, None)


# ── request bodies ───────────────────────────────────────────────────────────
class LoginBody(BaseModel):
    username: str | None = None   # email OR legacy username (Android app)
    email: str | None = None
    password: str


class AcceptInviteBody(BaseModel):
    token: str
    password: str
    display_name: str | None = None
    lang: str | None = None  # app locale (native) or chosen language (web) → mail


class ForgotBody(BaseModel):
    email: str


class ResetBody(BaseModel):
    token: str
    password: str


class RegisterBody(BaseModel):
    email: str
    password: str
    display_name: str | None = None
    lang: str | None = None  # app locale (native) or chosen language (web) → mail


# ── helpers ──────────────────────────────────────────────────────────────────
def _server_descriptor(request: Request, token: str) -> dict:
    host = request.url.hostname or "127.0.0.1"
    port = request.url.port or config.PORT
    scheme = request.headers.get("x-forwarded-proto") or request.url.scheme or "http"
    return {
        "name": config.SERVER_NAME,
        "provides": "server",
        "clientIdentifier": config.SERVER_ID,
        "accessToken": token,
        # The LAN address (if configured) so the app can stream locally at home
        # and fall back to the public URL when away. Blank when not set.
        "localUrl": config.LOCAL_URL,
        "connections": [{
            "protocol": scheme,
            "address": host,
            "port": port,
            "uri": f"{scheme}://{host}:{port}",
            "local": True,
            "relay": False,
        }],
    }


def _user_dict(u: User) -> dict:
    return {
        "id": u.id,
        "email": u.email,
        "username": u.display_name or u.email,
        "title": u.display_name or u.email,
        "friendlyName": u.display_name or u.email,
        "isAdmin": u.is_admin,
        # The server owner cannot delete their own account — the app hides the
        # delete-account row for them (the server also refuses it, 403).
        "isOwner": norm_email(u.email) == norm_email(config.ADMIN_EMAIL),
        "thumb": f"/users/{u.id}/thumb" if u.thumb_path else None,
        # Whether this user gets the "new music" email digest after a scan.
        "notifyNewMusic": bool(u.notify_new_music),
    }


def _find_login_user(db, identifier: str) -> User | None:
    ident = (identifier or "").strip()
    if not ident:
        return None
    # email match (case-insensitive)
    u = db.query(User).filter(User.email == norm_email(ident)).first()
    if u:
        return u
    # legacy username → the admin account (so the legacy username still works)
    if ident.lower() == config.USERNAME.lower():
        return db.query(User).filter(User.is_admin == True).order_by(User.id).first()  # noqa: E712
    # display name match (fallback)
    return db.query(User).filter(User.display_name == ident).first()


# ── endpoints ────────────────────────────────────────────────────────────────
@router.post("/auth/login")
def login(body: LoginBody, request: Request):
    identifier = body.email or body.username or ""
    keys = {
        f"ip:{_client_ip(request)}": FAIL_LIMIT_IP,
        f"user:{identifier.strip().lower()}": FAIL_LIMIT_ACCOUNT,
    }
    _check_not_locked(keys)

    db = SessionLocal()
    try:
        user = _find_login_user(db, identifier)
        if not user or not user.is_active or not verify_password(body.password, user.password_hash):
            _record_failure(keys)
            raise HTTPException(status_code=401, detail="Ferkearde e-mail/brûkersnamme of wachtwurd")
        _clear_failures(keys)
        label = "web" if body.email else "app"
        token = create_session(db, user, label=label)
        return {
            "token": token,
            "user": _user_dict(user),
            "server": _server_descriptor(request, token),
        }
    finally:
        db.close()


# ── forgot / reset password ──────────────────────────────────────────────────
RESET_TTL = 3600  # a reset link is valid for one hour


@router.post("/auth/forgot")
def forgot_password(body: ForgotBody, request: Request):
    """Mail a reset link. ALWAYS answers the same, whether or not the address has
    an account: a different answer would turn this endpoint into a way to find out
    who has one. Throttled per IP so it cannot be used to bomb someone's inbox."""
    email = norm_email(body.email or "")
    _check_not_locked({f"forgot:{_client_ip(request)}": FAIL_LIMIT_IP})
    _record_failure({f"forgot:{_client_ip(request)}": FAIL_LIMIT_IP})

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.email == email).first()
        if user and user.is_active:
            user.reset_token = new_invite()
            user.reset_expires_at = int(time.time()) + RESET_TTL
            db.commit()
            mailer.send_reset(user.email, user.reset_token,
                              server_name=config.SERVER_NAME,
                              valid_minutes=RESET_TTL // 60)
        return {"ok": True}
    finally:
        db.close()


@router.post("/auth/reset")
def reset_password(body: ResetBody, request: Request):
    """Consume a reset token and set the new password. The user is logged straight
    in, so they never bounce back to a login screen they just failed at."""
    if len(body.password) < 6:
        raise HTTPException(status_code=400, detail="Wachtwoord moet minstens 6 tekens zijn")

    db = SessionLocal()
    try:
        user = db.query(User).filter(User.reset_token == body.token).first()
        if not user or not user.reset_token:
            raise HTTPException(status_code=404, detail="Deze link is niet (meer) geldig")
        if user.reset_expires_at and user.reset_expires_at < int(time.time()):
            # Expired: clear it so a leaked old link stays dead.
            user.reset_token = None
            db.commit()
            raise HTTPException(status_code=410, detail="Deze link is verlopen. Vraag een nieuwe aan.")

        user.password_hash = hash_password(body.password)
        user.reset_token = None       # one shot
        user.reset_expires_at = 0
        user.is_active = True
        db.commit()

        _clear_failures({f"ip:{_client_ip(request)}": 0, f"user:{user.email}": 0})
        token = create_session(db, user, label="web")
        return {"token": token, "user": _user_dict(user),
                "server": _server_descriptor(request, token)}
    finally:
        db.close()


@router.post("/auth/accept-invite")
def accept_invite(body: AcceptInviteBody, request: Request):
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.invite_token == body.token).first()
        if not user:
            raise HTTPException(status_code=404, detail="Utnûging net fûn of al brûkt")
        if len(body.password) < 6:
            raise HTTPException(status_code=400, detail="Wachtwurd moat op syn minst 6 tekens wêze")
        user.password_hash = hash_password(body.password)
        user.invite_token = None
        user.is_active = True
        if body.display_name:
            user.display_name = body.display_name.strip()
        if body.lang:
            user.lang = body.lang.strip()
        db.commit()
        # First join — welcome them (background thread, never blocks the request).
        mailer.send_welcome(user.email, user.display_name or user.email,
                            config.SERVER_NAME, base_url=str(request.base_url),
                            lang=user.lang)
        token = create_session(db, user, label="web")
        return {"token": token, "user": _user_dict(user),
                "server": _server_descriptor(request, token)}
    finally:
        db.close()


@router.get("/auth/invite/{token}")
def invite_info(token: str):
    """Look up a pending invite so the web form can greet the user by email."""
    db = SessionLocal()
    try:
        user = db.query(User).filter(User.invite_token == token).first()
        if not user:
            raise HTTPException(status_code=404, detail="Utnûging net fûn")
        return {"email": user.email, "display_name": user.display_name}
    finally:
        db.close()


@router.post("/auth/register")
def register(body: RegisterBody, request: Request):
    if not config.OPEN_REGISTRATION:
        raise HTTPException(status_code=403, detail="Registraasje is allinne op útnûging")
    email = norm_email(body.email)
    if not email or "@" not in email:
        raise HTTPException(status_code=400, detail="Ferkeard e-mailadres")
    db = SessionLocal()
    try:
        if db.query(User).filter(User.email == email).first():
            raise HTTPException(status_code=409, detail="E-mail al yn gebrûk")
        user = User(email=email, password_hash=hash_password(body.password),
                    display_name=(body.display_name or email.split("@")[0]).strip(),
                    is_admin=False, is_active=True, created_at=int(time.time()),
                    lang=(body.lang or None))
        db.add(user)
        db.commit()
        mailer.send_welcome(user.email, user.display_name or user.email,
                            config.SERVER_NAME, base_url=str(request.base_url),
                            lang=user.lang)
        token = create_session(db, user, label="web")
        return {"token": token, "user": _user_dict(user),
                "server": _server_descriptor(request, token)}
    finally:
        db.close()


@router.get("/auth/me")
def me(user: User = Depends(require_user)):
    return _user_dict(user)


class NotifyBody(BaseModel):
    enabled: bool


@router.post("/users/me/notify")
def set_notify(body: NotifyBody, user: User = Depends(require_user)):
    """Toggle the per-user 'new music' email digest (Settings on the web)."""
    db = SessionLocal()
    try:
        u = db.get(User, user.id)
        if not u:
            raise HTTPException(status_code=404, detail="User not found")
        u.notify_new_music = bool(body.enabled)
        db.commit()
        return {"ok": True, "notifyNewMusic": bool(u.notify_new_music)}
    finally:
        db.close()


@router.delete("/users/me")
def delete_me(user: User = Depends(require_user)):
    """Self-service account deletion. The server owner cannot delete their own
    account (they own the server); everyone else may remove themselves and ALL
    their data — sessions, library grants, likes, playlists, per-track state and
    play history. Mirrors the admin delete cascade in admin_users.delete_user."""
    if norm_email(user.email) == norm_email(config.ADMIN_EMAIL):
        raise HTTPException(
            status_code=403,
            detail="De eigenaar van de server kan het eigen account niet verwijderen")
    db = SessionLocal()
    try:
        u = db.get(User, user.id)
        if not u:
            raise HTTPException(status_code=404, detail="User not found")
        uid = u.id
        db.execute(delete(AuthSession).where(AuthSession.user_id == uid))
        db.execute(delete(UserLibraryAccess).where(UserLibraryAccess.user_id == uid))
        db.execute(delete(LikedTrack).where(LikedTrack.user_id == uid))
        pl_ids = [p.id for p in db.execute(
            select(Playlist).where(Playlist.user_id == uid)).scalars()]
        if pl_ids:
            db.execute(delete(PlaylistItem).where(PlaylistItem.playlist_id.in_(pl_ids)))
        db.execute(delete(Playlist).where(Playlist.user_id == uid))
        db.execute(delete(UserTrackState).where(UserTrackState.user_id == uid))
        db.execute(delete(PlayEvent).where(PlayEvent.user_id == uid))
        db.delete(u)
        db.commit()
        return {"ok": True}
    finally:
        db.close()


# ── plex.tv-shaped helpers (used if the app's auth base is pointed here) ──
@router.get("/user")
def user_ep(user: User = Depends(require_user)):
    return _user_dict(user)


@router.get("/resources")
def resources(request: Request, user: User = Depends(require_user)):
    from ..deps import token_from_request
    return [_server_descriptor(request, token_from_request(request) or "")]
