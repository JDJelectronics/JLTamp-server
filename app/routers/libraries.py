"""Libraries — Plex-style manual library management.

The admin adds a library (name + one or more folders under the read-only music
mounts) and triggers scans manually. Regular users see only the libraries they
have been granted access to. Deleting a library removes only OUR database rows
— it never touches the read-only NAS files.
"""
from __future__ import annotations

import threading
import time

from fastapi import APIRouter, Request, HTTPException, Depends
from pydantic import BaseModel
from sqlalchemy import select, delete, func

from .. import config
from ..db import SessionLocal
from ..models import (Library, LibraryFolder, Artist, Album, Track,
                      UserLibraryAccess, User)
from ..deps import require_admin, require_user, accessible_library_ids
from ..scanner import scan_library, scan_state

router = APIRouter()


class CreateLibraryBody(BaseModel):
    name: str
    folders: list[str]
    kind: str = "music"


def _launch_scan(library_id: int, full: bool = False):
    threading.Thread(target=scan_library, args=(library_id,),
                     kwargs={"full": full}, daemon=True).start()


def _library_dict(db, lib: Library) -> dict:
    folders = [f.path for f in db.execute(
        select(LibraryFolder).where(LibraryFolder.library_id == lib.id)).scalars()]
    return {
        "id": lib.id,
        "key": str(lib.id),
        "name": lib.name,
        "kind": lib.kind,
        "folders": folders,
        "trackCount": lib.track_count,
        "lastScanAt": lib.last_scan_at,
        "lastScanMs": lib.last_scan_ms,
    }


# ── admin: available folders to build libraries from ─────────────────────────
@router.get("/libraries/available-folders")
def available_folders(_: User = Depends(require_admin)):
    return {"roots": config.available_music_roots()}


# ── list libraries (filtered per user) ───────────────────────────────────────
@router.get("/libraries")
def list_libraries(user: User = Depends(require_user)):
    db = SessionLocal()
    try:
        allowed = accessible_library_ids(db, user)  # None = all (admin)
        q = select(Library)
        libs = list(db.execute(q).scalars())
        out = []
        for lib in libs:
            if allowed is not None and lib.id not in allowed:
                continue
            out.append(_library_dict(db, lib))
        return {"libraries": out}
    finally:
        db.close()


@router.post("/libraries")
def create_library(body: CreateLibraryBody, _: User = Depends(require_admin)):
    folders = [f for f in (body.folders or []) if f.strip()]
    if not body.name.strip() or not folders:
        raise HTTPException(status_code=400, detail="Namme en op syn minst ien map fereaske")
    for f in folders:
        if not config.path_is_allowed(f):
            raise HTTPException(status_code=400,
                                detail=f"Map {f} falt bûten de muzykmap (net tastien)")
    db = SessionLocal()
    try:
        lib = Library(name=body.name.strip(), kind=body.kind or "music",
                      created_at=int(time.time()))
        db.add(lib)
        db.flush()
        for f in folders:
            db.add(LibraryFolder(library_id=lib.id, path=f))
        db.commit()
        lid = lib.id
        result = _library_dict(db, lib)
    finally:
        db.close()
    _launch_scan(lid)  # start the first scan immediately (like Plex "add + scan")
    return {"library": result, "scanning": True}


@router.post("/libraries/{library_id}/scan")
def scan_one(library_id: int, _: User = Depends(require_admin), full: bool = False):
    db = SessionLocal()
    try:
        if not db.get(Library, library_id):
            raise HTTPException(status_code=404, detail="Bibleteek net fûn")
    finally:
        db.close()
    _launch_scan(library_id, full=full)
    return {"ok": True, "state": scan_state()}


@router.get("/libraries/{library_id}")
def get_library(library_id: int, user: User = Depends(require_user)):
    db = SessionLocal()
    try:
        allowed = accessible_library_ids(db, user)
        if allowed is not None and library_id not in allowed:
            raise HTTPException(status_code=403, detail="Gjin tagong ta dizze bibleteek")
        lib = db.get(Library, library_id)
        if not lib:
            raise HTTPException(status_code=404, detail="Bibleteek net fûn")
        return {"library": _library_dict(db, lib)}
    finally:
        db.close()


@router.delete("/libraries/{library_id}")
def delete_library(library_id: int, _: User = Depends(require_admin)):
    """Remove a library and its indexed rows FROM OUR DB ONLY. The read-only
    NAS music files are never touched."""
    db = SessionLocal()
    try:
        lib = db.get(Library, library_id)
        if not lib:
            raise HTTPException(status_code=404, detail="Bibleteek net fûn")
        db.execute(delete(Track).where(Track.library_id == library_id))
        db.execute(delete(Album).where(Album.library_id == library_id))
        db.execute(delete(Artist).where(Artist.library_id == library_id))
        db.execute(delete(LibraryFolder).where(LibraryFolder.library_id == library_id))
        db.execute(delete(UserLibraryAccess).where(UserLibraryAccess.library_id == library_id))
        db.delete(lib)
        db.commit()
        return {"ok": True}
    finally:
        db.close()
