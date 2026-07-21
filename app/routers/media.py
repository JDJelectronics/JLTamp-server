"""Plex-compatible library browsing, metadata, genres and search — now
multi-library and per-user access filtered. Each JLTamp library is exposed as a
Plex 'section'; a user only sees the sections (and their content) they've been
granted. Admins see everything.
"""
from __future__ import annotations

from fastapi import APIRouter, Request, Depends, Query, HTTPException
from sqlalchemy import select, func

from .. import config
from ..db import SessionLocal
from ..deps import require_user, accessible_library_ids
from ..ids import parse_key
from ..models import Artist, Album, Track, Library, User
from ..serializers import artist_dict, album_dict, track_dict, track_states, container

router = APIRouter()


# ── access helpers ───────────────────────────────────────────────────────────
def _allowed(db, user: User) -> list[int] | None:
    """Library ids the user may see; None = all (admin)."""
    return accessible_library_ids(db, user)


def _scope(stmt, model, allowed: list[int] | None, section: str | None = None):
    """Restrict a query to the user's accessible libraries (and an optional
    single section/library key)."""
    if allowed is not None:
        stmt = stmt.where(model.library_id.in_(allowed or [-1]))
    if section and section.isdigit():
        stmt = stmt.where(model.library_id == int(section))
    return stmt


def _can_see(db, user: User, library_id: int) -> bool:
    allowed = _allowed(db, user)
    return allowed is None or library_id in allowed


# ── identity (also the reachability probe) ───────────────────────────────────
@router.get("/identity")
def identity(_: User = Depends(require_user)):
    return {"MediaContainer": {
        "machineIdentifier": config.SERVER_ID,
        "friendlyName": config.SERVER_NAME,
        "version": "0.2.0",
    }}


# ── sections = the user's accessible libraries ───────────────────────────────
@router.get("/library/sections")
def sections(user: User = Depends(require_user)):
    db = SessionLocal()
    try:
        allowed = _allowed(db, user)
        libs = list(db.execute(select(Library).order_by(Library.id)).scalars())
        directory = []
        for lib in libs:
            if allowed is not None and lib.id not in allowed:
                continue
            directory.append({
                "key": str(lib.id),
                "type": "artist",       # a music library is type 'artist' in Plex
                "title": lib.name,
                "agent": "jltamp",
                "scanner": "JLTamp Music",
            })
        return {"MediaContainer": {"size": len(directory), "Directory": directory}}
    finally:
        db.close()


def _apply_genre(stmt, model, genre: str | None):
    if genre:
        stmt = stmt.where(func.lower(model.genre) == genre.strip().lower())
    return stmt


# ── sorting (Plex `sort=field:dir`) ──────────────────────────────────────────
# Plex clients sort a section by passing e.g. sort=addedAt:desc. Unknown fields
# fall back to the natural order rather than erroring, which is what Plex does.
_SORT_FIELDS = {
    8: {  # artists
        "titleSort": (Artist.sort_name, Artist.name),
        "title": (Artist.name,),
        "addedAt": (Artist.id,),
    },
    9: {  # albums
        "titleSort": (Album.sort_title,),
        "title": (Album.title,),
        "addedAt": (Album.added_at,),
        "year": (Album.year,),
        "artist": (Album.artist_name, Album.sort_title),
    },
    10: {  # tracks
        "titleSort": (Track.sort_title, Track.title),
        "title": (Track.title,),
        "addedAt": (Track.added_at,),
        "year": (Track.year,),
        "artist": (Track.artist_name, Track.album_title, Track.disc_no, Track.track_no),
        "album": (Track.album_title, Track.disc_no, Track.track_no),
        "duration": (Track.duration_ms,),
    },
}
_DEFAULT_ORDER = {
    8: (Artist.sort_name, Artist.name),
    9: (Album.artist_name, Album.sort_title),
    10: (Track.artist_name, Track.album_title, Track.disc_no, Track.track_no, Track.title),
}


def _apply_sort(stmt, type: int, sort: str | None):
    fields = _SORT_FIELDS.get(type, {})
    if sort:
        name, _, direction = sort.partition(":")
        cols = fields.get(name.strip())
        if cols:
            desc = direction.strip().lower().startswith("desc")
            order = []
            for c in cols:
                # Untagged rows (no year) sort LAST either way — a year-sorted
                # list that opens with 200 "unknown" albums is useless. DESC puts
                # NULLs last already in SQLite; ASC needs the nudge.
                if not desc and c.nullable:
                    order.append(c.is_(None))
                order.append(c.desc() if desc else c.asc())
            return stmt.order_by(*order)
    return stmt.order_by(*_DEFAULT_ORDER[type])


def _apply_year(stmt, model, year: int | None, decade: int | None):
    if year:
        stmt = stmt.where(model.year == year)
    if decade:
        stmt = stmt.where(model.year >= decade, model.year < decade + 10)
    return stmt


def _page(db, stmt, start: int, size: int | None):
    if start:
        stmt = stmt.offset(start)
    if size is not None:
        stmt = stmt.limit(size)
    return list(db.execute(stmt).scalars())


# ── /library/sections/{key}/all — artists(8) / albums(9) / tracks(10) ────────
@router.get("/library/sections/{key}/all")
def section_all(request: Request, key: str, user: User = Depends(require_user),
                type: int = 8, genre: str | None = None, limit: int | None = None,
                sort: str | None = None, year: int | None = None,
                decade: int | None = None, artist: str | None = None):
    start = int(request.query_params.get("X-Plex-Container-Start", 0) or 0)
    size = request.query_params.get("X-Plex-Container-Size")
    size = int(size) if size is not None else (limit or None)

    db = SessionLocal()
    try:
        allowed = _allowed(db, user)
        if key and key.isdigit() and not _can_see(db, user, int(key)):
            raise HTTPException(403, "Gjin tagong")
        if type == 8:  # artists
            base = _scope(select(Artist), Artist, allowed, key)
            count = _scope(select(func.count(Artist.id)), Artist, allowed, key)
            total = db.execute(count).scalar() or 0
            items = [artist_dict(a) for a in _page(db, _apply_sort(base, 8, sort), start, size)]
        elif type == 9:  # albums
            def narrow(s):
                s = _apply_year(_apply_genre(_scope(s, Album, allowed, key), Album, genre),
                                Album, year, decade)
                if artist:
                    s = s.where(func.lower(Album.artist_name) == artist.strip().lower())
                return s
            total = db.execute(narrow(select(func.count(Album.id)))).scalar() or 0
            base = _apply_sort(narrow(select(Album)), 9, sort)
            items = [album_dict(a) for a in _page(db, base, start, size)]
        else:  # tracks (10)
            def narrow(s):
                s = _apply_year(_apply_genre(_scope(s, Track, allowed, key), Track, genre),
                                Track, year, decade)
                if artist:
                    s = s.where(func.lower(Track.artist_name) == artist.strip().lower())
                return s
            total = db.execute(narrow(select(func.count(Track.id)))).scalar() or 0
            base = _apply_sort(narrow(select(Track)), 10, sort)
            rows = _page(db, base, start, size)
            states = track_states(db, user.id, rows)
            items = [track_dict(t, state=states.get(t.id)) for t in rows]
        return container(items, total=total)
    finally:
        db.close()


# ── recently added ───────────────────────────────────────────────────────────
@router.get("/library/sections/{key}/recentlyAdded")
def recently_added(key: str, user: User = Depends(require_user), type: int = 10, limit: int = 30):
    db = SessionLocal()
    try:
        allowed = _allowed(db, user)
        section = key if key and key.isdigit() else None
        if type == 9:
            base = _scope(select(Album), Album, allowed, section).order_by(Album.added_at.desc()).limit(limit)
            items = [album_dict(a) for a in db.execute(base).scalars()]
        else:
            base = _scope(select(Track), Track, allowed, section).order_by(Track.added_at.desc()).limit(limit)
            rows = list(db.execute(base).scalars())
            states = track_states(db, user.id, rows)
            items = [track_dict(t, state=states.get(t.id)) for t in rows]
        return container(items)
    finally:
        db.close()


# ── genres ───────────────────────────────────────────────────────────────────
@router.get("/library/sections/{key}/genre")
def genres(key: str, user: User = Depends(require_user)):
    db = SessionLocal()
    try:
        allowed = _allowed(db, user)
        section = key if key and key.isdigit() else None
        stmt = _scope(select(Track.genre, func.count(Track.id)), Track, allowed, section) \
            .where(Track.genre != "").group_by(func.lower(Track.genre)).order_by(func.lower(Track.genre))
        rows = db.execute(stmt).all()
        directory = [{"key": g, "title": g, "count": c} for g, c in rows if g and len(g) >= 2]
        return {"MediaContainer": {"size": len(directory), "Directory": directory}}
    finally:
        db.close()


# ── folder view — expose artists as browsable folders ────────────────────────
@router.get("/library/sections/{key}/folder")
def folder(key: str, user: User = Depends(require_user)):
    db = SessionLocal()
    try:
        allowed = _allowed(db, user)
        section = key if key and key.isdigit() else None
        rows = db.execute(_scope(select(Artist), Artist, allowed, section)
                          .order_by(Artist.sort_name)).scalars()
        return container([artist_dict(a) for a in rows])
    finally:
        db.close()


# ── single metadata + children (access-checked) ──────────────────────────────
@router.get("/library/metadata/{rk}")
def metadata(rk: str, user: User = Depends(require_user)):
    parsed = parse_key(rk)
    db = SessionLocal()
    try:
        if parsed:
            kind, i = parsed
            if kind == "track":
                t = db.get(Track, i)
                if t and _can_see(db, user, t.library_id):
                    states = track_states(db, user.id, [t])
                    return container([track_dict(t, state=states.get(t.id))])
            elif kind == "album":
                a = db.get(Album, i)
                if a and _can_see(db, user, a.library_id):
                    return container([album_dict(a)])
            elif kind == "artist":
                a = db.get(Artist, i)
                if a and _can_see(db, user, a.library_id):
                    return container([artist_dict(a)])
        return container([])
    finally:
        db.close()


@router.get("/library/metadata/{rk}/children")
def metadata_children(rk: str, user: User = Depends(require_user)):
    parsed = parse_key(rk)
    db = SessionLocal()
    try:
        if parsed:
            kind, i = parsed
            if kind == "artist":
                ar = db.get(Artist, i)
                if not ar or not _can_see(db, user, ar.library_id):
                    return container([])
                rows = db.execute(select(Album).where(Album.artist_id == i)
                                  .order_by(Album.year, Album.sort_title)).scalars()
                return container([album_dict(a) for a in rows])
            if kind == "album":
                al = db.get(Album, i)
                if not al or not _can_see(db, user, al.library_id):
                    return container([])
                rows = list(db.execute(select(Track).where(Track.album_id == i)
                                       .order_by(Track.disc_no, Track.track_no, Track.title)).scalars())
                states = track_states(db, user.id, rows)
                return container([track_dict(t, state=states.get(t.id)) for t in rows])
        return container([])
    finally:
        db.close()


# ── search (scoped to accessible libraries) ──────────────────────────────────
@router.get("/hubs/search")
def hub_search(query: str = Query(""), user: User = Depends(require_user),
               limit: int = 20, sectionId: str | None = None):
    q = f"%{query.strip().lower()}%"
    db = SessionLocal()
    try:
        allowed = _allowed(db, user)
        tracks = list(db.execute(_scope(select(Track), Track, allowed, sectionId)
                                 .where(func.lower(Track.title).like(q)).limit(limit)).scalars())
        tstates = track_states(db, user.id, tracks)
        artists = db.execute(_scope(select(Artist), Artist, allowed, sectionId)
                             .where(func.lower(Artist.name).like(q)).limit(limit)).scalars()
        albums = db.execute(_scope(select(Album), Album, allowed, sectionId)
                            .where(func.lower(Album.title).like(q)).limit(limit)).scalars()
        hubs = [
            {"type": "track", "Metadata": [track_dict(t, state=tstates.get(t.id)) for t in tracks]},
            {"type": "artist", "Metadata": [artist_dict(a) for a in artists]},
            {"type": "album", "Metadata": [album_dict(a) for a in albums]},
        ]
        return {"MediaContainer": {"size": len(hubs), "Hub": hubs}}
    finally:
        db.close()


@router.get("/library/sections/{key}/search")
def section_search(key: str, query: str = Query(""), user: User = Depends(require_user)):
    q = f"%{query.strip().lower()}%"
    db = SessionLocal()
    try:
        allowed = _allowed(db, user)
        section = key if key and key.isdigit() else None
        tracks = list(db.execute(_scope(select(Track), Track, allowed, section)
                                 .where(func.lower(Track.title).like(q)).limit(30)).scalars())
        artists = db.execute(_scope(select(Artist), Artist, allowed, section)
                             .where(func.lower(Artist.name).like(q)).limit(15)).scalars()
        albums = db.execute(_scope(select(Album), Album, allowed, section)
                            .where(func.lower(Album.title).like(q)).limit(15)).scalars()
        states = track_states(db, user.id, tracks)
        items = ([track_dict(t, state=states.get(t.id)) for t in tracks]
                 + [artist_dict(a) for a in artists] + [album_dict(a) for a in albums])
        return container(items)
    finally:
        db.close()
