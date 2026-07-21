"""Audio streaming: direct file (with HTTP Range / seeking) + on-the-fly
transcode to MP3 (ffmpeg) for the cast/transcode fallback path."""
from __future__ import annotations

import shutil
import subprocess
from urllib.parse import unquote

from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import FileResponse, StreamingResponse

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

    ff = shutil.which("ffmpeg")
    if not ff:
        # No ffmpeg — fall back to the raw file (works for already-mp3 sources).
        return FileResponse(t.path, media_type=_CONTENT_TYPES.get((t.ext or "").lstrip("."), "audio/mpeg"))

    bitrate = min(int(maxAudioBitrate or 320), int(audioBitrate or 320), 320)
    proc = subprocess.Popen(
        [ff, "-hide_banner", "-loglevel", "error", "-i", t.path,
         "-vn", "-map", "0:a:0", "-f", "mp3", "-b:a", f"{bitrate}k", "pipe:1"],
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
    )

    def gen():
        try:
            while True:
                chunk = proc.stdout.read(64 * 1024)
                if not chunk:
                    break
                yield chunk
        finally:
            try:
                proc.stdout.close()
                proc.terminate()
            except Exception:
                pass

    return StreamingResponse(gen(), media_type="audio/mpeg")
