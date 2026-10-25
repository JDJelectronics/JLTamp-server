"""Server-side crossfade "mix" engine.

Streams a queue of tracks as ONE continuous, crossfaded audio stream (built with
ffmpeg's `acrossfade`). Because every transition is baked into a single stream,
the phone just plays one long track — the client never runs a JS-timer crossfade,
so it keeps working on the Android lock screen / in doze (where JS timers freeze).

MVP endpoint:
    GET /mix/stream?tracks=rk1,rk2,rk3&crossfade=6   ->  audio/mpeg

The client re-requests with the next batch when it nears the end, or on skip.
"""
from __future__ import annotations

import shutil
import subprocess

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse

from ..deps import require_user, accessible_library_ids
from ..db import SessionLocal
from ..models import Track, User

router = APIRouter()

# Bound one request so ffmpeg never opens hundreds of inputs at once.
MAX_TRACKS = 25


def _resolve_paths(keys: list[str], user: User) -> list[str]:
    """File paths for the accessible, existing tracks named in `keys`."""
    out: list[str] = []
    with SessionLocal() as db:
        allowed = accessible_library_ids(db, user)
        for k in keys:
            digits = "".join(ch for ch in k.strip() if ch.isdigit())
            if not digits:
                continue
            t = db.get(Track, int(digits))
            if not t or not t.path:
                continue
            if allowed is not None and t.library_id not in allowed:
                continue
            out.append(t.path)
    return out


def _build_ffmpeg(paths: list[str], xfade: float, bitrate: int) -> list[str]:
    ff = shutil.which("ffmpeg")
    cmd = [ff, "-hide_banner", "-loglevel", "error"]
    for p in paths:
        cmd += ["-i", p]
    n = len(paths)
    # Normalise every input to one common format so acrossfade can join them
    # (tracks may differ in sample rate / channels / codec).
    norm = [f"[{i}:a]aresample=44100,aformat=sample_fmts=fltp:channel_layouts=stereo[a{i}]"
            for i in range(n)]
    if n == 1:
        filt = ";".join(norm)
        out_label = "[a0]"
    else:
        chain = []
        cur = "[a0]"
        for i in range(1, n):
            out = f"[x{i}]" if i < n - 1 else "[mix]"
            chain.append(f"{cur}[a{i}]acrossfade=d={xfade}:c1=tri:c2=tri{out}")
            cur = out
        filt = ";".join(norm + chain)
        out_label = "[mix]"
    cmd += ["-filter_complex", filt, "-map", out_label,
            "-f", "mp3", "-b:a", f"{bitrate}k", "pipe:1"]
    return cmd


@router.get("/mix/stream")
def mix_stream(tracks: str = "", crossfade: float = 6.0, bitrate: int = 256,
               user: User = Depends(require_user)):
    """Continuous crossfaded stream of the given tracks."""
    keys = [k for k in (tracks or "").split(",") if k.strip()][:MAX_TRACKS]
    if not keys:
        raise HTTPException(400, "no tracks given")
    paths = _resolve_paths(keys, user)
    if not paths:
        raise HTTPException(404, "no playable tracks")
    if not shutil.which("ffmpeg"):
        raise HTTPException(500, "ffmpeg not available on the server")

    xfade = max(0.5, min(float(crossfade or 6), 12.0))
    br = max(96, min(int(bitrate or 256), 320))
    proc = subprocess.Popen(_build_ffmpeg(paths, xfade, br),
                            stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

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
