"""Playlist importer — rebuild a playlist from a shared link by matching each
entry (title + artist) against the user's OWN library.

**No media is fetched, decrypted or downloaded.** All this reads is a public
track LISTING — names and artists, the same text a browser shows anyone who opens
the link. Those names are then matched against files the operator already owns and
has mounted; the music mount itself stays read-only. Entries with no match are
reported back untouched, and nothing is created for them.

The listing is read through an official API using credentials the operator
registers themselves (IMPORT_CLIENT_ID / IMPORT_CLIENT_SECRET). Without those,
this import path stays switched off.
"""
from __future__ import annotations

import base64
import json
import re
import time
import urllib.parse
import urllib.request
from collections import defaultdict

from fastapi import APIRouter, Depends, HTTPException

from .. import config
from ..deps import require_user, accessible_library_ids
from ..db import SessionLocal
from ..ids import playlist_key
from ..models import Playlist, PlaylistItem, Track, User

router = APIRouter()

_UA = "JLTamp/1.0"


# ── normalisation + matching ─────────────────────────────────────────────────
def _norm(s: str) -> str:
    """Aggressive normalise for fuzzy matching: lowercase, drop parenthetical
    noise ("(Remastered)", "[Official Video]"), feat., and all non-alphanumerics."""
    s = (s or "").lower()
    s = re.sub(r"\(.*?\)|\[.*?\]", " ", s)
    s = re.sub(r"\bfeat\.?\b.*", " ", s)
    s = re.sub(r"\bft\.?\b.*", " ", s)
    return re.sub(r"[^a-z0-9]+", "", s)


def _match(db, user: User, items: list[dict]) -> tuple[list[int], list[dict]]:
    """Match [{title, artist}] to library track ids. Returns (matched_track_ids
    in order, unmatched items). Exact normalised-title match first, then the
    artist decides between candidates; a lone candidate is taken as-is."""
    allowed = accessible_library_ids(db, user)
    q = db.query(Track)
    if allowed is not None:
        q = q.filter(Track.library_id.in_(allowed or [-1]))
    tracks = q.all()

    # Normalise every library title ONCE (regex-heavy) and reuse it for both the
    # exact-match index and the substring fallback — otherwise the fallback would
    # re-normalise the whole library for every unmatched item (N×M regex calls).
    norm_pairs: list[tuple[str, Track]] = [(_norm(t.title), t) for t in tracks]
    by_title: dict[str, list[Track]] = defaultdict(list)
    for n, t in norm_pairs:
        by_title[n].append(t)

    matched: list[int] = []
    unmatched: list[dict] = []
    seen: set[int] = set()

    for it in items:
        nt = _norm(it.get("title", ""))
        na = _norm(it.get("artist", ""))
        if not nt:
            continue
        cands = list(by_title.get(nt, []))
        if not cands:
            # substring fallback (one title contains the other), bounded to 8
            cands = []
            for n, t in norm_pairs:
                if n and (n in nt or nt in n):
                    cands.append(t)
                    if len(cands) >= 8:
                        break
        if not cands:
            unmatched.append({"title": it.get("title", ""), "artist": it.get("artist", "")})
            continue

        pick = None
        if na:
            for t in cands:
                blob = _norm(t.artist_name) + _norm(t.orig_artist) + _norm(t.album_artist)
                if na in blob or (blob and blob in na):
                    pick = t
                    break
        if pick is None:
            pick = cands[0] if len(cands) == 1 else (cands[0] if not na else None)
        if pick is None:
            unmatched.append({"title": it.get("title", ""), "artist": it.get("artist", "")})
            continue
        if pick.id not in seen:
            seen.add(pick.id)
            matched.append(pick.id)

    return matched, unmatched


# ── Spotify (client-credentials) ─────────────────────────────────────────────
_spotify_tok = {"token": "", "exp": 0.0}


def _spotify_token() -> str:
    if not (config.IMPORT_CLIENT_ID and config.IMPORT_CLIENT_SECRET):
        raise HTTPException(400, "Spotify import is not configured on this server")
    if _spotify_tok["token"] and _spotify_tok["exp"] > time.time() + 30:
        return _spotify_tok["token"]
    basic = base64.b64encode(
        f"{config.IMPORT_CLIENT_ID}:{config.IMPORT_CLIENT_SECRET}".encode()
    ).decode()
    body = urllib.parse.urlencode({"grant_type": "client_credentials"}).encode()
    req = urllib.request.Request(
        "https://accounts.spotify.com/api/token", data=body,
        headers={"Authorization": f"Basic {basic}", "Content-Type": "application/x-www-form-urlencoded"},
    )
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            data = json.loads(r.read().decode())
    except Exception:
        raise HTTPException(502, "Spotify auth failed (check the server credentials)")
    tok = data.get("access_token") or ""
    if not tok:
        raise HTTPException(502, "Spotify auth returned no token")
    _spotify_tok["token"] = tok
    _spotify_tok["exp"] = time.time() + int(data.get("expires_in", 3600))
    return tok


_SPOTIFY_ID = re.compile(r"playlist[/:]([A-Za-z0-9]+)")


def _fetch_spotify(url: str) -> tuple[str, list[dict]]:
    m = _SPOTIFY_ID.search(url)
    if not m:
        raise HTTPException(400, "That doesn't look like a Spotify playlist link")
    pid = m.group(1)
    tok = _spotify_token()
    hdr = {"Authorization": f"Bearer {tok}", "User-Agent": _UA}

    # playlist name
    name = "Spotify playlist"
    try:
        req = urllib.request.Request(
            f"https://api.spotify.com/v1/playlists/{pid}?fields=name", headers=hdr)
        with urllib.request.urlopen(req, timeout=15) as r:
            name = json.loads(r.read().decode()).get("name") or name
    except Exception:
        pass

    items: list[dict] = []
    next_url = (f"https://api.spotify.com/v1/playlists/{pid}/tracks"
                f"?fields=items(track(name,artists(name))),next&limit=100")
    while next_url:
        req = urllib.request.Request(next_url, headers=hdr)
        try:
            with urllib.request.urlopen(req, timeout=20) as r:
                page = json.loads(r.read().decode())
        except Exception as e:
            code = getattr(e, "code", 0)
            if code == 404:
                raise HTTPException(404, "Spotify playlist not found (is it public?)")
            raise HTTPException(502, "Couldn't read the Spotify playlist")
        for row in page.get("items", []):
            tr = (row or {}).get("track") or {}
            title = tr.get("name")
            if not title:
                continue
            artists = ", ".join(a.get("name", "") for a in (tr.get("artists") or []) if a.get("name"))
            items.append({"title": title, "artist": artists})
        next_url = page.get("next")
    return name, items


# ── endpoint ─────────────────────────────────────────────────────────────────
# Only the official-API source is supported. A second source that scraped entry
# titles out of a video site was removed on 2026-07-27: reading a listing through
# a documented API with the operator's own credentials is a different thing from
# extracting it out of a site that does not offer one, and only the first belongs
# in a repo anyone can run.
def _detect(url: str) -> str:
    u = (url or "").lower()
    if "spotify.com" in u or u.startswith("spotify:"):
        return "spotify"
    return ""


@router.post("/import/playlist")
def import_playlist(url: str = "", title: str = "", user: User = Depends(require_user)):
    """Read a public playlist's track listing, match it to the library, and create
    a JLTamp playlist from the matches. Returns a report (matched / unmatched)."""
    url = (url or "").strip()
    source = _detect(url)
    if not source:
        raise HTTPException(400, "Paste a supported playlist link")

    name, items = _fetch_spotify(url)
    if not items:
        raise HTTPException(404, "No tracks found in that playlist")

    db = SessionLocal()
    try:
        matched_ids, unmatched = _match(db, user, items)
        if not matched_ids:
            return {
                "source": source, "title": name, "total": len(items),
                "matched": 0, "ratingKey": None,
                "unmatched": unmatched[:100],
            }
        now = int(time.time())
        pl = Playlist(user_id=user.id, title=(title.strip() or name), created_at=now, updated_at=now)
        db.add(pl)
        db.flush()
        for pos, tid in enumerate(matched_ids):
            db.add(PlaylistItem(playlist_id=pl.id, track_id=tid, position=pos, added_by=user.id))
        db.commit()
        return {
            "source": source, "title": pl.title, "ratingKey": playlist_key(pl.id),
            "total": len(items), "matched": len(matched_ids),
            "unmatched": unmatched[:100],
        }
    finally:
        db.close()
