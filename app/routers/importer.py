"""Playlist importer — turn a Spotify or YouTube playlist link into a JLTamp
playlist by matching each track (title + artist) against the user's OWN library.

Spotify: official Web API, client-credentials flow (public playlists only) — needs
SPOTIFY_CLIENT_ID / SPOTIFY_CLIENT_SECRET. YouTube: yt-dlp extracts the entry
titles (no API key), which are parsed into artist/title heuristically.

Nothing is downloaded — we only read the track list and map it onto music the
server already has (the music mount stays read-only). Unmatched tracks are
reported back.
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
    if not (config.SPOTIFY_CLIENT_ID and config.SPOTIFY_CLIENT_SECRET):
        raise HTTPException(400, "Spotify import is not configured on this server")
    if _spotify_tok["token"] and _spotify_tok["exp"] > time.time() + 30:
        return _spotify_tok["token"]
    basic = base64.b64encode(
        f"{config.SPOTIFY_CLIENT_ID}:{config.SPOTIFY_CLIENT_SECRET}".encode()
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


# ── YouTube (yt-dlp) ─────────────────────────────────────────────────────────
_YT_NOISE = re.compile(r"\b(official\s*(music\s*)?video|official\s*audio|lyrics?|audio|hd|hq|mv|visualizer|4k)\b", re.I)


def _parse_yt_title(raw: str, uploader: str) -> dict:
    t = re.sub(r"\(.*?\)|\[.*?\]", " ", raw or "")
    t = _YT_NOISE.sub(" ", t).strip(" -|·")
    # Split on the FIRST "artist – title" separator: a dash/en-dash/em-dash that
    # has a SPACE before it (so "Jay-Z", "T-Pain", "Hip-Hop" are NOT split).
    # Catches "Artist - Song", "Artist -Song", "Artist – Song".
    parts = re.split(r"\s+[-–—]\s*", t, maxsplit=1)
    if len(parts) == 2 and parts[0].strip() and parts[1].strip():
        return {"artist": parts[0].strip(), "title": parts[1].strip()}
    up = re.sub(r"\s*-\s*topic$", "", uploader or "", flags=re.I).strip()
    return {"artist": up, "title": t.strip()}


def _fetch_youtube(url: str) -> tuple[str, list[dict]]:
    try:
        import yt_dlp  # heavy import — only when actually importing from YouTube
    except Exception:
        raise HTTPException(500, "YouTube import isn't available on this server")
    opts = {"quiet": True, "no_warnings": True, "extract_flat": "in_playlist", "skip_download": True}
    try:
        with yt_dlp.YoutubeDL(opts) as ydl:
            info = ydl.extract_info(url, download=False)
    except Exception:
        raise HTTPException(502, "Couldn't read that YouTube playlist")
    entries = info.get("entries") or []
    if not entries and info.get("title"):
        entries = [info]  # single video link
    name = info.get("title") or "YouTube playlist"
    items = []
    for e in entries:
        if not e:
            continue
        parsed = _parse_yt_title(e.get("title", ""), e.get("uploader") or e.get("channel") or "")
        if parsed["title"]:
            items.append(parsed)
    return name, items


# ── endpoint ─────────────────────────────────────────────────────────────────
def _detect(url: str) -> str:
    u = (url or "").lower()
    if "spotify.com" in u or u.startswith("spotify:"):
        return "spotify"
    if "youtube.com" in u or "youtu.be" in u:
        return "youtube"
    return ""


@router.post("/import/playlist")
def import_playlist(url: str = "", title: str = "", user: User = Depends(require_user)):
    """Fetch a Spotify/YouTube playlist, match it to the library, and create a
    JLTamp playlist from the matches. Returns a report (matched / unmatched)."""
    url = (url or "").strip()
    source = _detect(url)
    if not source:
        raise HTTPException(400, "Paste a Spotify or YouTube playlist link")

    name, items = _fetch_spotify(url) if source == "spotify" else _fetch_youtube(url)
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
