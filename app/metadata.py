"""Online metadata enrichment (Plex-style) — fetches nice artist photos and
album covers so the app looks rich even when local files have no embedded art.

Primary source: **Deezer** public API (no API key, good artist pictures + album
covers). All fetched images are cached under the writable data dir
(`/data/artwork`) — NEVER written back to the read-only NAS.

Stdlib only (urllib) to keep the server dependency-light. Best-effort and
rate-limited; failures are logged and skipped, never fatal to a scan.
"""
from __future__ import annotations

import json
import logging
import time
import urllib.parse
import urllib.request

from sqlalchemy import select

from . import config
from .db import SessionLocal
from .models import Artist, Album

log = logging.getLogger("metadata")

_UA = "JLTamp/1.0 (+self-hosted music server)"
_DEEZER = "https://api.deezer.com"
_RATE_SLEEP = 0.25  # be polite to Deezer


def _get_json(url: str) -> dict | None:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception as e:
        log.debug("deezer GET failed %s: %s", url, e)
        return None


def _download(url: str, dest) -> bool:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=15) as r:
            data = r.read()
        if not data:
            return False
        dest.write_bytes(data)
        return True
    except Exception as e:
        log.debug("art download failed %s: %s", url, e)
        return False


def _deezer_artist_image(name: str) -> str | None:
    q = urllib.parse.quote(name)
    data = _get_json(f"{_DEEZER}/search/artist?q={q}&limit=1")
    time.sleep(_RATE_SLEEP)
    items = (data or {}).get("data") or []
    if not items:
        return None
    a = items[0]
    # prefer the largest; Deezer returns picture_xl (1000px) down to picture_small
    for k in ("picture_xl", "picture_big", "picture_medium", "picture"):
        url = a.get(k)
        if url and "/artist//" not in url:  # skip empty placeholder
            return url
    return None


def _deezer_album_cover(artist: str, album: str) -> str | None:
    q = urllib.parse.quote(f"{artist} {album}".strip())
    data = _get_json(f"{_DEEZER}/search/album?q={q}&limit=1")
    time.sleep(_RATE_SLEEP)
    items = (data or {}).get("data") or []
    if not items:
        return None
    a = items[0]
    for k in ("cover_xl", "cover_big", "cover_medium", "cover"):
        url = a.get(k)
        if url and "/cover//" not in url:
            return url
    return None


def enrich_library(library_id: int) -> None:
    """Fetch artist photos + album covers for anything not yet enriched.
    Idempotent: only touches rows with enriched=False."""
    db = SessionLocal()
    try:
        artists = list(db.execute(select(Artist).where(
            Artist.library_id == library_id, Artist.enriched == False)).scalars())  # noqa: E712
        albums = list(db.execute(select(Album).where(
            Album.library_id == library_id, Album.enriched == False)).scalars())  # noqa: E712
    finally:
        db.close()

    if artists or albums:
        log.info("enriching library %d: %d artists, %d albums",
                 library_id, len(artists), len(albums))

    # ── artists: online photo (local files rarely have one) ──
    for ar in artists:
        img = _deezer_artist_image(ar.name)
        online_path = None
        if img:
            dest = config.ARTWORK_DIR / f"artist_online_{ar.id}.jpg"
            if _download(img, dest):
                online_path = str(dest)
        db = SessionLocal()
        try:
            row = db.get(Artist, ar.id)
            if row:
                if online_path:
                    row.online_art_path = online_path
                row.enriched = True
                db.commit()
        finally:
            db.close()

    # ── albums: only fetch when there is no local cover ──
    for al in albums:
        online_path = None
        if not al.art_path:
            img = _deezer_album_cover(al.artist_name, al.title)
            if img:
                dest = config.ARTWORK_DIR / f"album_online_{al.id}.jpg"
                if _download(img, dest):
                    online_path = str(dest)
        db = SessionLocal()
        try:
            row = db.get(Album, al.id)
            if row:
                if online_path:
                    row.online_art_path = online_path
                row.enriched = True
                db.commit()
        finally:
            db.close()

    if artists or albums:
        log.info("enrichment done for library %d", library_id)
