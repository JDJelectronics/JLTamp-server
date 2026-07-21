"""Password hashing + token helpers. Stdlib only (pbkdf2_hmac) so the server
stays dependency-light and easy to self-host.
"""
from __future__ import annotations

import hashlib
import hmac
import secrets

_ALGO = "sha256"
_ITER = 240_000


def hash_password(password: str) -> str:
    salt = secrets.token_bytes(16)
    dk = hashlib.pbkdf2_hmac(_ALGO, password.encode("utf-8"), salt, _ITER)
    return f"pbkdf2_{_ALGO}${_ITER}${salt.hex()}${dk.hex()}"


def verify_password(password: str, stored: str | None) -> bool:
    if not stored:
        return False
    try:
        scheme, iter_s, salt_hex, hash_hex = stored.split("$")
        assert scheme == f"pbkdf2_{_ALGO}"
        iters = int(iter_s)
        salt = bytes.fromhex(salt_hex)
        expected = bytes.fromhex(hash_hex)
    except Exception:
        return False
    dk = hashlib.pbkdf2_hmac(_ALGO, password.encode("utf-8"), salt, iters)
    return hmac.compare_digest(dk, expected)


def new_token() -> str:
    return secrets.token_hex(24)


def new_invite() -> str:
    return secrets.token_urlsafe(24)


def norm_email(email: str) -> str:
    return (email or "").strip().lower()
