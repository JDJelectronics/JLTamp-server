"""Image uploads: per-user avatars + custom album/artist artwork (used when no
embedded/folder/online art was found). All uploads are stored in the writable
data dir — NEVER on the read-only NAS. Accepts a base64 data URL in JSON so it
works from both web (canvas/FileReader) and the mobile image picker.
"""
from __future__ import annotations

import base64
import io
import time
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import FileResponse
from pydantic import BaseModel
from PIL import Image, ImageOps

from .. import config
from ..db import SessionLocal
from ..models import User, Album, Artist
from ..deps import require_user, require_admin
from ..ids import parse_key

router = APIRouter()


class ImageBody(BaseModel):
    image: str  # "data:image/...;base64,...." or raw base64


def _decode(data_url: str) -> Image.Image:
    b64 = data_url.split(",", 1)[-1] if data_url.startswith("data:") else data_url
    raw = base64.b64decode(b64)
    return Image.open(io.BytesIO(raw)).convert("RGB")


def _save_square(img: Image.Image, dest: Path, size: int) -> None:
    img = ImageOps.fit(img, (size, size), Image.LANCZOS)
    dest.parent.mkdir(parents=True, exist_ok=True)
    img.save(dest, "JPEG", quality=88, optimize=True)


# ── avatar ───────────────────────────────────────────────────────────────────
@router.post("/users/me/avatar")
def set_avatar(body: ImageBody, user: User = Depends(require_user)):
    dest = config.AVATAR_DIR / f"user_{user.id}.jpg"
    try:
        _save_square(_decode(body.image), dest, 512)
    except Exception:
        raise HTTPException(400, "Kin de ôfbylding net ferwurkje")
    db = SessionLocal()
    try:
        u = db.get(User, user.id)
        if u:
            u.thumb_path = str(dest)
            db.commit()
    finally:
        db.close()
    return {"thumb": f"/users/{user.id}/thumb?t={int(time.time())}"}


@router.get("/users/{uid}/thumb")
def user_thumb(uid: int):
    db = SessionLocal()
    try:
        u = db.get(User, uid)
        if not u or not u.thumb_path or not Path(u.thumb_path).exists():
            raise HTTPException(404, "Gjin ôfbylding")
        return FileResponse(u.thumb_path, media_type="image/jpeg")
    finally:
        db.close()


# ── custom album / artist artwork (admin curation) ──────────────────────────
@router.post("/library/metadata/{rk}/art")
def set_art(rk: str, body: ImageBody, _: User = Depends(require_admin)):
    parsed = parse_key(rk)
    if not parsed:
        raise HTTPException(404, "Ûnbekende kaai")
    kind, i = parsed
    try:
        img = _decode(body.image)
    except Exception:
        raise HTTPException(400, "Kin de ôfbylding net ferwurkje")
    db = SessionLocal()
    try:
        if kind == "album":
            row = db.get(Album, i)
            if not row:
                raise HTTPException(404, "Album net fûn")
            dest = config.ARTWORK_DIR / f"album_custom_{i}.jpg"
            _save_square(img, dest, 900)
            row.art_path = str(dest)
            db.commit()
        elif kind == "artist":
            row = db.get(Artist, i)
            if not row:
                raise HTTPException(404, "Artyst net fûn")
            dest = config.ARTWORK_DIR / f"artist_custom_{i}.jpg"
            _save_square(img, dest, 900)
            row.online_art_path = str(dest)
            db.commit()
        else:
            raise HTTPException(400, "Allinne album/artyst")
    finally:
        db.close()
    return {"ok": True, "art": f"/art/{rk}?t={int(time.time())}"}
