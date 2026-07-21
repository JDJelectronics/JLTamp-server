"""Cover art. Serves raw images and a Plex-shaped /photo/:/transcode resizer
(JPEG), so the app's getThumbnailUrl / cast / Android-Auto art URLs all work."""
from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response
from PIL import Image, ImageOps

from .. import config
from ..deps import require_token
from ..db import SessionLocal
from ..ids import parse_key
from ..models import Track, Album, Artist, PlaylistItem

router = APIRouter(dependencies=[Depends(require_token)])


def resolve_art_path(rk: str) -> str | None:
    parsed = parse_key(rk)
    if not parsed:
        return None
    kind, i = parsed
    db = SessionLocal()
    try:
        if kind == "album":
            a = db.get(Album, i)
            return (a.art_path or a.online_art_path) if a else None
        if kind == "artist":
            a = db.get(Artist, i)
            # artist: prefer the online photo (nicer than a reused album cover)
            return (a.online_art_path or a.art_path) if a else None
        if kind == "track":
            t = db.get(Track, i)
            if not t:
                return None
            if t.art_path:
                return t.art_path
            al = db.get(Album, t.album_id)
            return (al.art_path or al.online_art_path) if al else None
        if kind == "playlist":
            # Playlist cover = first item's album art (simple; collage later).
            pi = db.query(PlaylistItem).filter(PlaylistItem.playlist_id == i) \
                   .order_by(PlaylistItem.position).first()
            if pi:
                t = db.get(Track, pi.track_id)
                if t:
                    al = db.get(Album, t.album_id)
                    return (t.art_path or (al.art_path if al else None))
        return None
    finally:
        db.close()


def _extract_rk_from_arturl(url: str) -> str:
    # url is like "/art/al45" (optionally with query) → "al45"
    u = unquote(url or "").split("?")[0]
    if "/art/" in u:
        return u.rsplit("/art/", 1)[-1].strip("/")
    return u.strip("/")


@router.get("/art/{rk}")
def raw_art(rk: str):
    p = resolve_art_path(rk)
    if not p or not Path(p).exists():
        raise HTTPException(404, "No art")
    return FileResponse(p)


@router.get("/photo/:/transcode")
def photo_transcode(url: str = "", width: int = 300, height: int = 0,
                    format: str = "jpeg", quality: int = 80,
                    minSize: int = 1, upscale: int = 1):
    rk = _extract_rk_from_arturl(url)
    src = resolve_art_path(rk)
    if not src or not Path(src).exists():
        raise HTTPException(404, "No art")

    w = max(16, min(int(width or 300), 2000))
    h = max(16, min(int(height or width or 300), 2000))
    q = max(40, min(int(quality or 80), 95))

    cache_key = hashlib.sha1(f"{src}|{Path(src).stat().st_mtime}|{w}x{h}|{q}".encode()).hexdigest()
    cache_file = config.CACHE_DIR / f"{cache_key}.jpg"
    if cache_file.exists():
        return FileResponse(cache_file, media_type="image/jpeg")

    try:
        img = Image.open(src).convert("RGB")
        # minSize/upscale → cover-fit (fill the box, centre-crop). For square
        # art (w==h) this is identical to a normal fit.
        img = ImageOps.fit(img, (w, h), Image.LANCZOS)
        img.save(cache_file, "JPEG", quality=q, optimize=True)
    except Exception as e:
        import logging; logging.getLogger("jltamp").warning("art transcode failed: %s", e)
        raise HTTPException(500, "Image processing failed")
    return FileResponse(cache_file, media_type="image/jpeg")
