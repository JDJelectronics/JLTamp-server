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

# Whether the playlist importers are available at all (False = off, the default).
# Off means: the endpoints answer 403 AND the clients hide the UI for them, so a
# deployment that wants no outbound lookups has no dead buttons either.
# Advertised in the /api descriptor, next to OPEN_REGISTRATION, so an
# unauthenticated login screen can already know.
IMPORTERS = _bool("JLTAMP_IMPORTERS", False)

# Public demo with one shared account: nobody may delete that account or give
# it a picture. The app hides the delete row (canDeleteAccount), the server
# refuses both (403).
DEMO = _bool("JLTAMP_DEMO", False)

SERVER_NAME = os.environ.get("SERVER_NAME", "JLTamp")
RESCAN_INTERVAL_MIN = int(os.environ.get("RESCAN_INTERVAL_MIN", "0") or 0)
PORT = int(os.environ.get("PORT", "32400") or 32400)

# Hoeveel nummers er tegelijk omgezet mogen worden (ffmpeg, zie stream.py).
#
# Omzetten gebeurt alleen bij casten of als de speler het formaat niet aankan;
# gewoon afspelen stuurt het bestand door en kost hier niets. Elke omzetting is
# wél een eigen ffmpeg-proces ván zo'n 20 à 40 MB, binnen een container die op
# 1500 MB staat. Zonder grens loopt die bij een stuk of dertig vol en grijpt de
# OOM-killer in — en die kiest niet ffmpeg, die kiest de server, waarna iedereen
# stilvalt. Twaalf is ruim: het drukste moment ooit gemeten was zes luisteraars
# in totaal, omzetting of niet.
MAX_TRANSCODES = int(os.environ.get("JLTAMP_MAX_TRANSCODES", "12") or 12)


# ── JLTamp Radio (app/radio) ─────────────────────────────────────────────────
#
# Eén 24/7-zender voor iedereen, aan/uit met de schakelaar `radio` in meta
# (standaard UIT). Staat hij aan, dan lopen er vast een paar ffmpeg's naast de
# omzettingen hierboven, en die tellen NIET mee in MAX_TRANSCODES: één encoder,
# hooguit twee decoders (alleen tijdens een overvloei tegelijk) en af en toe
# één korte voor het meten van de luidheid of het decoderen van een stem.
# Samen zo'n 100 à 150 MB op z'n drukst; het mengen zelf werkt in blokjes van
# een tiende seconde en houdt per nummer hooguit tien seconden voorraad.
def _int_lijst(naam: str) -> list[int]:
    uit = []
    for stuk in os.environ.get(naam, "").replace(" ", "").split(","):
        if stuk.isdigit():
            uit.append(int(stuk))
    return uit


def _getal(naam: str, standaard: float) -> float:
    try:
        return float(os.environ.get(naam, "") or standaard)
    except ValueError:
        return standaard


RADIO_DIR = DATA_DIR / "radio"
RADIO_BITRATE = int(_getal("JLTAMP_RADIO_BITRATE", 128))           # kbps AAC
RADIO_SEGMENT_S = int(_getal("JLTAMP_RADIO_SEGMENT", 6))           # HLS-stuk
RADIO_LIJST = int(_getal("JLTAMP_RADIO_LIJST", 10))                # stukken in de playlist
RADIO_OVERVLOEI_S = _getal("JLTAMP_RADIO_OVERVLOEI", 4.0)
RADIO_DUCK_DB = _getal("JLTAMP_RADIO_DUCK_DB", -12.0)              # muziek onder de stem
# Of er gepraat wordt. Uit = alleen muziek, ook als de presentator er is.
RADIO_PRESENTATOR = _bool("JLTAMP_RADIO_PRESENTATOR", True)
# Uit welke bibliotheken (id's, komma's). Leeg = alle.
RADIO_BIBLIOTHEKEN = _int_lijst("JLTAMP_RADIO_BIBLIOTHEKEN")
# Hoeveel later een luisteraar iets hoort dan het gemaakt wordt: een HLS-speler
# begint zo'n drie stukken achter de rand. /radio/nu rekent daarmee, zodat de
# hoes in de app tegelijk wisselt met het geluid.
RADIO_VERTRAGING_S = _getal("JLTAMP_RADIO_VERTRAGING", RADIO_SEGMENT_S * 3)

# Spotify Web API (client-credentials) for the playlist importer — read PUBLIC
# playlists. Create a free app at developer.spotify.com and set these two env
# vars; the importer's Spotify path is disabled until both are present.
# IMPORT_CLIENT_ID / IMPORT_CLIENT_SECRET still work: that is what these were
# called before, and an existing install should not lose its import.
SPOTIFY_CLIENT_ID = (os.environ.get("SPOTIFY_CLIENT_ID") or os.environ.get("IMPORT_CLIENT_ID") or "").strip()
SPOTIFY_CLIENT_SECRET = (os.environ.get("SPOTIFY_CLIENT_SECRET") or os.environ.get("IMPORT_CLIENT_SECRET") or "").strip()

# Optional LAN address (e.g. http://192.168.1.10:8090) the app can reach when
# it's on the same network. Advertised to the client so it streams locally
# (faster, no internet round-trip) at home and falls back to the public URL
# when away. Blank = only the public/request address is offered.
LOCAL_URL = os.environ.get("LOCAL_URL", "").strip().rstrip("/")

# Home Assistant, voor "het huis doet mee" (zie routers/huis.py): lampen in de
# kleur van de hoes en afspelen op een speaker in huis. Alles optioneel; zonder
# HA_URL en HA_TOKEN zegt de server gewoon "niet ingesteld" en doet hij niets.
#   HA_URL         waar Home Assistant voor DEZE container te bereiken is. Vanuit
#                  een container op de Studio is dat de host, niet 127.0.0.1:
#                  http://host.docker.internal:8123
#   HA_TOKEN       een long-lived access token uit het HA-profiel.
#   HA_STREAM_URL  het adres van deze server zoals de SPEAKERS het zien — die
#                  halen het nummer zelf op. Leeg = het eerste LOCAL_URL-adres.
HA_URL = os.environ.get("HA_URL", "").strip().rstrip("/")
HA_TOKEN = os.environ.get("HA_TOKEN", "").strip()
HA_STREAM_URL = os.environ.get("HA_STREAM_URL", "").strip().rstrip("/")

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
