"""Audio streaming: direct file (with HTTP Range / seeking) + on-the-fly
transcode to MP3 (ffmpeg) for the cast/transcode fallback path.

Aan het omzetten zit een grens (config.MAX_TRANSCODES). Waarom, zie daar: het
is de enige plek in deze server waar een luisteraar meer kost dan wat bytes
doorgeven, en de container heeft maar 1500 MB.
"""
from __future__ import annotations

import logging
import shutil
import subprocess
import threading
from urllib.parse import unquote

from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

from .. import config
from ..deps import require_user, accessible_library_ids
from ..db import SessionLocal
from ..ids import parse_key
from ..models import Track, User

router = APIRouter()

_CONTENT_TYPES = {
    "mp3": "audio/mpeg", "flac": "audio/flac", "aac": "audio/aac",
    "m4a": "audio/mp4", "mp4": "audio/mp4", "alac": "audio/mp4",
    "ogg": "audio/ogg", "opus": "audio/opus", "wav": "audio/wav",
    "aiff": "audio/aiff", "aif": "audio/aiff", "wma": "audio/x-ms-wma",
}


log = logging.getLogger("jltamp.stream")

# ── de grens op het omzetten ─────────────────────────────────────────────────
#
# Eén slot per lopende ffmpeg. Een sync endpoint draait in de threadpool, dus
# het tellen moet achter een slot: twee verzoeken tegelijk mogen niet allebei
# het laatste plekje krijgen.
_bezig: set[subprocess.Popen] = set()
_slot = threading.Lock()


def _neem_plek() -> bool:
    """Een plek in de rij, of False als het vol is.

    Ruimt eerst op wat al klaar is. Dat is niet alleen netheid: raakt er ooit
    een proces zoek doordat een speler halverwege wegvalt zonder dat de
    generator netjes wordt afgesloten, dan loopt de teller zonder dit langzaam
    vol en kan er uiteindelijk niets meer omgezet worden.
    """
    with _slot:
        for p in [p for p in _bezig if p.poll() is not None]:
            _bezig.discard(p)
        return len(_bezig) < config.MAX_TRANSCODES


def _bewaar(proc: subprocess.Popen) -> None:
    with _slot:
        _bezig.add(proc)


def _geef_terug(proc: subprocess.Popen) -> None:
    with _slot:
        _bezig.discard(proc)


def bezet() -> int:
    """Voor het diagnosescherm: hoeveel er nu omgezet worden."""
    with _slot:
        return len(_bezig)


def _track_from_key(rk: str, user: User) -> Track:
    parsed = parse_key(rk)
    if not parsed or parsed[0] != "track":
        raise HTTPException(404, "Not a track")
    db = SessionLocal()
    try:
        t = db.get(Track, parsed[1])
        if not t:
            raise HTTPException(404, "Track not found")
        allowed = accessible_library_ids(db, user)
        if allowed is not None and t.library_id not in allowed:
            raise HTTPException(403, "Gjin tagong ta dit nûmer")
        db.expunge(t)
        return t
    finally:
        db.close()


# Direct play: /library/parts/{rk}/file.ext  (FileResponse handles Range).
@router.get("/library/parts/{rk}/{filename:path}")
def part(rk: str, user: User = Depends(require_user)):
    t = _track_from_key(rk, user)
    ctype = _CONTENT_TYPES.get((t.ext or "").lstrip("."), "application/octet-stream")
    return FileResponse(t.path, media_type=ctype, filename=None)


# Transcode fallback: /music/:/transcode/universal/start.mp3?path=/library/metadata/{rk}&...
@router.get("/music/:/transcode/universal/start.mp3")
def transcode(request: Request, user: User = Depends(require_user),
              path: str = "", audioBitrate: int = 320, maxAudioBitrate: int = 320):
    rk = ""
    p = unquote(path or "")
    if "/library/metadata/" in p:
        rk = p.rsplit("/library/metadata/", 1)[-1].split("/")[0]
    t = _track_from_key(rk, user)
    return omgezet(t, min(int(maxAudioBitrate or 320), int(audioBitrate or 320), 320))


def omgezet(t: Track, bitrate: int = 320):
    """Het nummer als mp3, met de grens op het omzetten erbij.

    Los van het verzoek zelf, zodat ook de speakers in huis (zie huis.py, die
    met een eigen ondertekend adres komen in plaats van een token) via precies
    dezelfde weg gaan."""
    ext = (t.ext or "").lstrip(".").lower()
    ff = shutil.which("ffmpeg")
    if not ff:
        # No ffmpeg — fall back to the raw file (works for already-mp3 sources).
        return FileResponse(t.path, media_type=_CONTENT_TYPES.get(ext, "audio/mpeg"))

    if not _neem_plek():
        # Vol. Is de bron al mp3, dan kan hij gewoon rechtstreeks: de speler
        # vroeg om een omzetting, maar krijgt hetzelfde formaat en hoort het
        # verschil niet. Alleen bij een formaat dat er écht doorheen moet
        # (flac, wma) houdt het op — en dan is een nette 503 beter dan een
        # container die omvalt en iedereen tegelijk stil zet.
        log.warning("omzetten zit vol: %d van de %d bezig — %s valt terug op het bestand zelf",
                    bezet(), config.MAX_TRANSCODES, ext or "?")
        if ext in ("mp3", "aac", "m4a", "mp4"):
            return FileResponse(t.path, media_type=_CONTENT_TYPES.get(ext, "audio/mpeg"))
        raise HTTPException(503, "Even te druk met omzetten — probeer het zo weer")

    bitrate = min(int(bitrate or 320), 320)
    proc = subprocess.Popen(
        [ff, "-hide_banner", "-loglevel", "error", "-i", t.path,
         "-vn", "-map", "0:a:0", "-f", "mp3", "-b:a", f"{bitrate}k", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )

    _bewaar(proc)

    def gen():
        try:
            while True:
                chunk = proc.stdout.read(64 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            # Ook als de speler er halverwege mee ophoudt: het proces gaat weg
            # en de plek komt vrij.
            try:
                proc.stdout.close()
                proc.terminate()
            except Exception:
                pass
            _geef_terug(proc)

    return StreamingResponse(gen(), media_type="audio/mpeg")
