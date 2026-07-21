"""Auth: per-user bearer tokens.

A token is issued on login and sent as `X-Plex-Token` (header or query param —
Android Auto / Chromecast can only send the query form). Each token maps to a
User via the `sessions` table. Admins implicitly see every library; other users
only see libraries granted to them (see accessible_library_ids).
"""
from __future__ import annotations

import time

from fastapi import Request, HTTPException

from . import config
from .db import SessionLocal
from .models import User, Session as AuthSession, UserLibraryAccess
from .security import hash_password, new_token, verify_password


# ── token plumbing ───────────────────────────────────────────────────────────
def token_from_request(request: Request) -> str | None:
    return request.headers.get("X-Plex-Token") or request.query_params.get("X-Plex-Token")


def create_session(db, user: User, label: str = "") -> str:
    tok = new_token()
    now = int(time.time())
    db.add(AuthSession(token=tok, user_id=user.id, created_at=now, last_seen_at=now, label=label))
    db.commit()
    return tok


# A token unused for this long is dead. Generous — an active listener refreshes
# last_seen on every request, so real sessions never expire; but a token that
# leaked (e.g. into a proxy log via the query-string form) stops working once
# nobody has used it for six months.
TOKEN_IDLE_TTL = 180 * 86400


def _user_for_token(token: str | None) -> User | None:
    if not token:
        return None
    db = SessionLocal()
    try:
        sess = db.get(AuthSession, token)
        if not sess:
            return None
        now = int(time.time())
        if sess.last_seen_at and now - sess.last_seen_at > TOKEN_IDLE_TTL:
            # Idle too long — burn it so a stale leaked token cannot be replayed.
            try:
                db.delete(sess)
                db.commit()
            except Exception:
                db.rollback()
            return None
        user = db.get(User, sess.user_id)
        if not user or not user.is_active:
            return None
        # Throttle the last_seen write (once per 60s) and never let a transient
        # write lock during a scan turn an authenticated request into a 500.
        if now - (sess.last_seen_at or 0) > 60:
            try:
                sess.last_seen_at = now
                db.commit()
            except Exception:
                db.rollback()
        db.expunge(user)  # detach so callers can use it after the session closes
        return user
    finally:
        db.close()


# ── FastAPI dependencies ─────────────────────────────────────────────────────
def optional_user(request: Request) -> User | None:
    return _user_for_token(token_from_request(request))


def require_user(request: Request) -> User:
    user = _user_for_token(token_from_request(request))
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or missing X-Plex-Token")
    return user


def require_admin(request: Request) -> User:
    user = require_user(request)
    if not user.is_admin:
        raise HTTPException(status_code=403, detail="Admin only")
    return user


def require_token(request: Request) -> None:
    """Back-compat dependency for endpoints that only need *a* valid user."""
    require_user(request)


# ── access helpers ───────────────────────────────────────────────────────────
def accessible_library_ids(db, user: User) -> list[int] | None:
    """Library ids a user may see. None = all (admin)."""
    if user.is_admin:
        return None
    rows = db.execute(
        UserLibraryAccess.__table__.select().where(UserLibraryAccess.user_id == user.id)
    ).all()
    return [r.library_id for r in rows]


# ── admin seeding ────────────────────────────────────────────────────────────
def seed_admin() -> None:
    """Guarantee the owner account exists AND is a working admin.

    Idempotent and self-healing: keyed on the owner's EMAIL, not on "any admin
    exists". If the owner row is present but was somehow demoted or deactivated
    (see the mutual-demotion race the admin guards now prevent), this re-promotes
    it in place instead of trying to INSERT a duplicate — which would hit the
    email UNIQUE constraint and crash startup. The owner can never be locked out.
    """
    db = SessionLocal()
    try:
        owner = db.query(User).filter(User.email == config.ADMIN_EMAIL).first()
        if owner:
            changed = False
            if not owner.is_admin:
                owner.is_admin = True
                changed = True
            if not owner.is_active:
                owner.is_active = True
                changed = True
            # SECURITY: never allow the account to keep the built-in default
            # password on a public server. If the owner still has "changeme",
            # null the hash so it can NEVER be used to log in, and force recovery
            # through the email reset flow (which is configured). No credential is
            # invented or stored anywhere in the process.
            if owner.password_hash and verify_password("changeme", owner.password_hash):
                owner.password_hash = None
                changed = True
                import logging
                logging.getLogger("jltamp").warning(
                    "SECURITY: default admin password was active — it has been "
                    "disabled. Use 'forgot password' to set a real one.")
            # Only (re)set the password from config when the row has none yet AND
            # the configured password is not the insecure default.
            if not owner.password_hash and config.PASSWORD and config.PASSWORD != "changeme":
                owner.password_hash = hash_password(config.PASSWORD)
                changed = True
            if changed:
                db.commit()
            return
        # Fresh install: seed the owner, but never with the insecure default —
        # leave the hash empty so the first thing required is an email reset.
        seed_pw = config.PASSWORD if config.PASSWORD and config.PASSWORD != "changeme" else None
        admin = User(
            email=config.ADMIN_EMAIL,
            password_hash=hash_password(seed_pw) if seed_pw else None,
            display_name=config.ADMIN_NAME,
            is_admin=True,
            is_active=True,
            created_at=int(time.time()),
        )
        db.add(admin)
        db.commit()
    finally:
        db.close()
