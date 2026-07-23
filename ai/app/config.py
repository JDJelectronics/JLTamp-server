"""Configuration — everything comes from the environment, nothing is hardcoded.

The previous Plex-era engine carried a live server token in its source. That is
the one thing this module exists to prevent: no secret ever lands in git. Copy
`.env.example` to `.env` and fill it in; `.env` is gitignored.
"""
from __future__ import annotations

import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv() -> None:
    """Read `.env` into the environment without pulling in a dependency.

    Real environment variables always win, so a systemd unit or docker-compose
    can override the file without editing it.
    """
    env_file = BASE_DIR / ".env"
    if not env_file.exists():
        return
    for raw in env_file.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


_load_dotenv()


def _env(name: str, default: str = "") -> str:
    """An empty value counts as unset.

    `.env.example` ships every key with a blank value so it doubles as
    documentation. Without this, copying it and filling in only what you need
    would blank out every default — AI_DATA_DIR="" became Path("."), and the
    engine tried to read the working directory as a JSON file.
    """
    return (os.environ.get(name) or "").strip() or default


def _env_int(name: str, default: int) -> int:
    try:
        return int(_env(name) or default)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(_env(name) or default)
    except ValueError:
        return default


# ── JLTamp server ────────────────────────────────────────────────────────────
JLTAMP_URL = _env("JLTAMP_URL", "http://192.168.1.10:8090").rstrip("/")
# Either a long-lived session token, or email+password to obtain one at startup.
JLTAMP_TOKEN = _env("JLTAMP_TOKEN")
JLTAMP_EMAIL = _env("JLTAMP_EMAIL")
JLTAMP_PASSWORD = _env("JLTAMP_PASSWORD")

# ── Embedding server (llama.cpp on the Jetson's GPU) ──────────────────────────
EMBED_URL = _env("EMBED_URL", "http://127.0.0.1:3100").rstrip("/")
EMBED_ENDPOINT = f"{EMBED_URL}/embedding"
EMBED_BATCH = _env_int("EMBED_BATCH", 64)
EMBED_TIMEOUT = _env_int("EMBED_TIMEOUT", 120)

# ── This service ─────────────────────────────────────────────────────────────
HOST = _env("AI_HOST", "0.0.0.0")
PORT = _env_int("AI_PORT", 5000)
# Shared secret the app must present. Empty = open, which is only sane on a
# trusted tailnet; the service logs a warning at startup when it is unset.
API_KEY = _env("AI_API_KEY")
# Browsers enforce CORS, so a wildcard here undoes the API key for web clients.
CORS_ORIGINS = [o.strip() for o in _env("AI_CORS_ORIGINS", "").split(",") if o.strip()]

# Require every caller to present their own JLTamp session token, which the
# service validates against JLTamp. Turn this ON whenever the service is
# reachable beyond a trusted network: without it the AI acts as the single
# account in JLTAMP_EMAIL, so one person's playlists land in another's library.
REQUIRE_USER_TOKEN = _env("AI_REQUIRE_USER_TOKEN", "0").lower() in (
    "1", "true", "yes", "on")

DATA_DIR = Path(_env("AI_DATA_DIR", str(BASE_DIR / "data")))

# Infer a genre for tracks that carry none, from their embedding neighbours, so
# the ~half of the library with a blank/placeholder tag still matches genre
# prompts. A guessed genre is written to a separate overlay, never over a real
# tag. Threshold measured: lower labels more and scores better here, 0.28 keeps
# a floor of neighbour agreement so it is not a blind guess.
INFER_GENRES = _env("AI_INFER_GENRES", "1").lower() in ("1", "true", "yes", "on")
INFER_THRESHOLD = _env_float("AI_INFER_THRESHOLD", 0.28)
INFERRED_GENRES_FILE = Path(_env("AI_INFERRED_GENRES_FILE",
                                 str(DATA_DIR / "inferred_genres.json")))
# Audio features (BPM/energy/brightness) produced by scripts/analyze_audio.py.
FEATURES_FILE = Path(_env("AI_FEATURES_FILE", str(DATA_DIR / "track_features.json")))

# How often the background worker re-syncs the track list from JLTamp.
LIBRARY_REFRESH_SEC = _env_int("AI_LIBRARY_REFRESH_SEC", 3600)
JOB_TIMEOUT_SEC = _env_int("AI_JOB_TIMEOUT_SEC", 120)

# Weekly per-user playlists only run once we can actually know a taste. A user
# who has barely listened gets no "personal" playlist rather than a random one.
# Both must be met: enough of an account history, and enough listening in it.
MIN_ACCOUNT_AGE_SEC = _env_int("AI_MIN_ACCOUNT_AGE_DAYS", 0) * 86400
MIN_TASTE_SEED = _env_int("AI_MIN_TASTE_SEED", 25)
MAX_JOBS = _env_int("AI_MAX_JOBS", 20)

# ── Scoring weights ──────────────────────────────────────────────────────────
# In practice bge-m3 puts related text at cosine ~0.5-0.75 and unrelated at
# ~0.3-0.5, so the whole meaningful range is about 0.25 wide. The first version
# of this carried the old engine's weights (+0.5 per keyword, +2.0 per context
# tag, +5.0 for a named artist), which dwarfed that range completely: a prompt
# for "instrumentale focus muziek" returned Ariana Grande's "Focus" because a
# literal word match outweighed every semantic signal.
#
# Everything here is now small relative to 0.25. Boosts break ties between
# semantically similar tracks; they no longer decide the ranking.
SCORING = {
    "MIN_SCORE": _env_float("AI_MIN_SCORE", 0.42),
    "BOOST_EXPLICIT_ARTIST": 0.60,   # naming an artist is a strong intent
    "BOOST_CONTEXT_MATCH": 0.06,     # per matching tag, capped in scoring.py
    "BOOST_KEYWORD_MATCH": 0.04,     # per prompt word found in the metadata
    "BOOST_KIDS_MATCH": 0.15,
    # Naming a genre the library actually has is a strong, explicit
    # statement — stronger than any similarity the text can express.
    "BOOST_NAMED_GENRE": 0.25,
    "BOOST_AUDIO_FEATURE": 0.14,
    "BOOST_LIKED": 0.05,
    "PENALTY_ARTIST_REPEAT": 0.03,
    "PENALTY_SKIPPED": 0.04,
    "BOOST_UNPLAYED": 0.02,
    "MAX_TRACKS": _env_int("AI_MAX_TRACKS", 50),
}
