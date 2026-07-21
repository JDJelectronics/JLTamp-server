"""Runtime configuration, read from environment variables (see .env.example).

Kept deliberately tiny and dependency-free so the server is trivial to reason
about and self-host.
"""
from __future__ import annotations

import os
import secrets
from pathlib import Path


def _bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


MUSIC_DIR = Path(os.environ.get("MUSIC_DIR", "/music"))
DATA_DIR = Path(os.environ.get("DATA_DIR", "/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

ARTWORK_DIR = DATA_DIR / "artwork"
ARTWORK_DIR.mkdir(parents=True, exist_ok=True)
AVATAR_DIR = DATA_DIR / "avatars"
AVATAR_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR = DATA_DIR / "cache"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

DB_PATH = DATA_DIR / "library.db"

# Legacy single-user creds — kept so the admin can still log in as "admin"
# from the Android app. The admin account is seeded from these on first boot.
USERNAME = os.environ.get("JLTAMP_USERNAME", "admin")
PASSWORD = os.environ.get("JLTAMP_PASSWORD", "changeme")

# Seed admin (multi-user). Login works with either this email or the legacy
# username above.
ADMIN_EMAIL = os.environ.get("JLTAMP_ADMIN_EMAIL", "admin@example.com").strip().lower()
ADMIN_NAME = os.environ.get("JLTAMP_ADMIN_NAME", "Admin")

# Whether anyone can self-register (False = invite-only, the user's choice).
OPEN_REGISTRATION = _bool("JLTAMP_OPEN_REGISTRATION", False)

SERVER_NAME = os.environ.get("SERVER_NAME", "JLTamp")
RESCAN_INTERVAL_MIN = int(os.environ.get("RESCAN_INTERVAL_MIN", "0") or 0)
PORT = int(os.environ.get("PORT", "32400") or 32400)

# Optional LAN address (e.g. http://192.168.1.10:8090) the app can reach when
# it's on the same network. Advertised to the client so it streams locally
# (faster, no internet round-trip) at home and falls back to the public URL
# when away. Blank = only the public/request address is offered.
LOCAL_URL = os.environ.get("LOCAL_URL", "").strip().rstrip("/")

# Audio file extensions we index.
AUDIO_EXTS = {
    ".mp3", ".flac", ".m4a", ".aac", ".ogg", ".opus",
    ".wav", ".aiff", ".aif", ".wma", ".alac", ".dsf", ".dff",
}
# Hi-res / lossless codecs the default Chromecast receiver can't decode →
# transcoded to mp3 on the cast/transcode path.
HIRES_CONTAINERS = {"flac", "alac", "wav", "aiff", "aif", "dsf", "dff", "wma"}

# Cover-art filenames to look for next to tracks when there's no embedded art.
COVER_NAMES = ["cover", "folder", "front", "albumart", "album", "thumb"]
COVER_EXTS = [".jpg", ".jpeg", ".png", ".webp"]


def _persisted_or_new_server_id() -> str:
    """A stable machineIdentifier. Env override wins; else persist a random one."""
    env = os.environ.get("SERVER_ID", "").strip()
    if env:
        return env
    id_file = DATA_DIR / "server_id"
    if id_file.exists():
        return id_file.read_text().strip()
    new_id = secrets.token_hex(20)
    id_file.write_text(new_id)
    return new_id


SERVER_ID = _persisted_or_new_server_id()


def available_music_roots() -> list[dict]:
    """Folders the admin may turn into libraries — MUSIC_DIR and its immediate
    subdirectories (which is where the read-only NAS mounts land, e.g.
    /music/mp3, /music/flac). Read-only listing only; we never write here."""
    roots: list[dict] = []
    base = MUSIC_DIR
    try:
        if base.exists():
            roots.append({"path": str(base), "name": base.name or "music"})
            for child in sorted(base.iterdir()):
                try:
                    if child.is_dir() and not child.name.startswith("@"):
                        roots.append({"path": str(child), "name": child.name})
                except OSError:
                    continue
    except OSError:
        pass
    return roots


def path_is_allowed(path: str) -> bool:
    """A library folder must live under MUSIC_DIR (defence-in-depth: never let an
    admin point a library outside the read-only music mounts)."""
    try:
        p = Path(path).resolve()
        base = MUSIC_DIR.resolve()
        return p == base or base in p.parents
    except Exception:
        return False
