"""Cover art. Serves raw images and a Plex-shaped /photo/:/transcode resizer
(JPEG), so the app's getThumbnailUrl / cast / Android-Auto art URLs all work."""
from __future__ import annotations

import hashlib
from pathlib import Path
from urllib.parse import unquote

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse, Response
from sqlalchemy import select
from PIL import Image, ImageOps

from .. import config
from ..deps import require_user, accessible_library_ids
from ..db import SessionLocal
from ..ids import parse_key
from ..models import Track, Album, Artist, Playlist, PlaylistItem, PlaylistMember, User

router = APIRouter()


def resolve_art_path(rk: str, user: User) -> str | None:
    """Resolve a ratingKey to an on-disk art path, ENFORCING the caller's access:
    library entities must live in a library the user was granted; a playlist cover
    is only served to its owner or a member (else the collage would leak another
    user's playlist contents to anyone who guesses the sequential id)."""
    parsed = parse_key(rk)
    if not parsed:
        return None
    kind, i = parsed
    db = SessionLocal()
    try:
        allowed = accessible_library_ids(db, user)   # None = admin (all libraries)

        def _ok(lib_id: int) -> bool:
            return allowed is None or lib_id in allowed

        if kind == "album":
            a = db.get(Album, i)
            return (a.art_path or a.online_art_path) if (a and _ok(a.library_id)) else None
        if kind == "artist":
            a = db.get(Artist, i)
            # artist: prefer the online photo (nicer than a reused album cover)
            return (a.online_art_path or a.art_path) if (a and _ok(a.library_id)) else None
        if kind == "track":
            t = db.get(Track, i)
            if not t or not _ok(t.library_id):
                return None
            if t.art_path:
                return t.art_path
            al = db.get(Album, t.album_id)
            return (al.art_path or al.online_art_path) if al else None
        if kind == "playlist":
            pl = db.get(Playlist, i)
            if not pl:
                return None
            is_member = db.execute(
                select(PlaylistMember.id)
                .where(PlaylistMember.playlist_id == i, PlaylistMember.user_id == user.id)
            ).first()
            if pl.user_id != user.id and not is_member:
                return None
            # Playlist cover = a 2×2 collage of the first up-to-4 UNIQUE album arts.
            return _playlist_collage_path(i, db)
        return None
    finally:
        db.close()


def _playlist_art_sources(playlist_id: int, db, limit: int = 4) -> list[str]:
    """Up-to-`limit` UNIQUE, on-disk album-art paths for a playlist, in order."""
    rows = (db.query(PlaylistItem)
              .filter(PlaylistItem.playlist_id == playlist_id)
              .order_by(PlaylistItem.position).all())
    out: list[str] = []
    seen: set[str] = set()
    for pi in rows:
        t = db.get(Track, pi.track_id)
        if not t:
            continue
        al = db.get(Album, t.album_id)
        p = t.art_path or (al.art_path if al else None) or (al.online_art_path if al else None)
        if not p or p in seen or not Path(p).exists():
            continue
        seen.add(p)
        out.append(p)
        if len(out) >= limit:
            break
    return out


def _playlist_collage_path(playlist_id: int, db) -> str | None:
    """Render (and cache) a square collage from a playlist's first unique arts.
    1 art → that art as-is; 2 → side-by-side; 3 → two-top + one-bottom; 4 → 2×2."""
    paths = _playlist_art_sources(playlist_id, db, 4)
    if not paths:
        return None
    if len(paths) == 1:
        return paths[0]

    key = "|".join(f"{p}:{Path(p).stat().st_mtime_ns}" for p in paths)
    cache_file = config.CACHE_DIR / f"plcollage_{hashlib.sha1(key.encode()).hexdigest()}.jpg"
    if cache_file.exists():
        return str(cache_file)

    S = 600
    h = S // 2
    canvas = Image.new("RGB", (S, S), (18, 18, 22))

    def tile(path: str, x: int, y: int, w: int, ht: int) -> None:
        im = ImageOps.fit(Image.open(path).convert("RGB"), (w, ht), Image.LANCZOS)
        canvas.paste(im, (x, y))

    try:
        if len(paths) == 2:
            tile(paths[0], 0, 0, h, S); tile(paths[1], h, 0, h, S)
        elif len(paths) == 3:
            tile(paths[0], 0, 0, h, h); tile(paths[1], h, 0, h, h)
            tile(paths[2], 0, h, S, h)
        else:
            tile(paths[0], 0, 0, h, h); tile(paths[1], h, 0, h, h)
            tile(paths[2], 0, h, h, h); tile(paths[3], h, h, h, h)
        canvas.save(cache_file, "JPEG", quality=85, optimize=True)
    except Exception as e:
        import logging; logging.getLogger("jltamp").warning("playlist collage failed: %s", e)
        return paths[0]   # fall back to the first art
    return str(cache_file)


def _extract_rk_from_arturl(url: str) -> str:
    # url is like "/art/al45" (optionally with query) → "al45"
    u = unquote(url or "").split("?")[0]
    if "/art/" in u:
        return u.rsplit("/art/", 1)[-1].strip("/")
    return u.strip("/")


@router.get("/art/{rk}")
def raw_art(rk: str, user: User = Depends(require_user)):
    p = resolve_art_path(rk, user)
    if not p or not Path(p).exists():
        raise HTTPException(404, "No art")
    return FileResponse(p)


@router.get("/photo/:/transcode")
def photo_transcode(url: str = "", width: int = 300, height: int = 0,
                    format: str = "jpeg", quality: int = 80,
                    minSize: int = 1, upscale: int = 1,
                    user: User = Depends(require_user)):
    rk = _extract_rk_from_arturl(url)
    src = resolve_art_path(rk, user)
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
