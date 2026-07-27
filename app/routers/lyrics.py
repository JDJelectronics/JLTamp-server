"""Lyrics for a track — local sources first, an online lookup as fallback.

Measured on the real library before building this: **zero** `.lrc` sidecars, and
embedded lyrics in roughly a quarter of the MP3s and none of the FLACs. A
local-only feature would therefore show nothing for most of the library, so
there is an online fallback (LRCLIB — free, no API key, and the only source that
returns *synced* lines).

Order, cheapest and most private first:
  1. `.lrc` sidecar next to the audio file  → synced
  2. `.txt` sidecar                          → plain
  3. embedded tag (USLT / LYRICS / ©lyr)     → plain
  4. LRCLIB over the network                 → synced when it has it

⚠️ Step 4 is **OFF unless switched on** (`JLTAMP_LYRICS_ONLINE=true`). The owner
of this deployment chose local-only on 2026-07-27, and off-by-default is the
right shape for self-hosted software anyway: reaching out to a third party
should be a decision, not a default. When enabled it sends only the artist and
title — nothing else, and never the file.

With it off, expect most of the library to have no lyrics: this collection has
zero `.lrc` sidecars and embedded lyrics in only ~a quarter of the MP3s.

Results are cached in the writable data dir — never on the read-only music
mount. Negative results are cached too, but tagged with whether the online step
was allowed at the time, so enabling it later re-checks them instead of serving
a stale "not found".

"""
from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from fastapi import APIRouter, Depends, HTTPException

from .. import config
from ..db import SessionLocal
from ..deps import require_user
from ..models import Track, User

router = APIRouter(tags=["lyrics"])
log = logging.getLogger("lyrics")

# OFF by default — the online step is opt-in, never a surprise. A default of
# "true" means a rebuild or a fresh install starts talking to a third party
# without anyone deciding to; for a self-hosted server that is the wrong way
# round. Set JLTAMP_LYRICS_ONLINE=true to allow it.
ONLINE = os.getenv("JLTAMP_LYRICS_ONLINE", "false").lower() in ("1", "true", "yes")
CACHE = config.DATA_DIR / "lyrics"
CACHE.mkdir(parents=True, exist_ok=True)

# "[01:23.45] a line" — the LRC timestamp form.
LRC_LINE = re.compile(r"\[(\d+):(\d+)(?:[.:](\d+))?\]\s*(.*)")


def _parse_lrc(text: str) -> list[dict] | None:
    """LRC text → [{timeMs, line}]. None when there is not a single timestamp,
    so caller can fall back to treating it as plain text."""
    out: list[dict] = []
    for raw in text.splitlines():
        m = LRC_LINE.match(raw.strip())
        if not m:
            continue
        mins, secs, frac, line = m.groups()
        ms = int(mins) * 60000 + int(secs) * 1000
        if frac:
            ms += int(frac.ljust(3, "0")[:3])
        out.append({"timeMs": ms, "line": line.strip()})
    return out or None


def _from_sidecar(audio_path: str) -> tuple[str, list[dict] | None, str] | None:
    p = Path(audio_path)
    for ext, kind in ((".lrc", "synced"), (".txt", "plain")):
        f = p.with_suffix(ext)
        try:
            if not f.is_file():
                continue
            text = f.read_text(encoding="utf-8", errors="ignore").strip()
            if not text:
                continue
            return (text, _parse_lrc(text) if kind == "synced" else None, f"sidecar{ext}")
        except OSError:
            continue
    return None


def _from_tags(audio_path: str) -> tuple[str, list[dict] | None, str] | None:
    try:
        import mutagen
        f = mutagen.File(audio_path)
        if not f or not f.tags:
            return None
        for key in f.tags.keys():
            k = str(key).upper()
            if "USLT" in k or "LYRIC" in k or k == "\xa9LYR":
                val = f.tags[key]
                text = getattr(val, "text", val)
                if isinstance(text, list):
                    text = text[0] if text else ""
                text = str(text).strip()
                if text:
                    return (text, _parse_lrc(text), "embedded")
    except Exception as e:
        log.debug("tag read failed for %s: %s", audio_path, e)
    return None


def _from_lrclib(artist: str, title: str, album: str, dur_s: int):
    """LRCLIB lookup.

    Deliberately does NOT send album/duration to /api/get. That endpoint demands
    an EXACT match on every field it is given, and this library's album titles
    are stored as "Artist - Album" (and singles land in a synthetic "[Singles]"
    collection), so including them made every lookup miss — verified against
    "Miley Cyrus — Flowers", which LRCLIB definitely has. Artist + title alone
    hits; /api/search is the fuzzy fallback for the rest.
    """
    if not ONLINE or not (artist and title):
        return None
    import urllib.parse
    import urllib.request

    def _fetch(url: str):
        req = urllib.request.Request(
            url, headers={"User-Agent": "JLTamp (self-hosted music server)"})
        with urllib.request.urlopen(req, timeout=6) as r:
            return json.loads(r.read().decode("utf-8"))

    def _pick(d: dict):
        synced = (d.get("syncedLyrics") or "").strip()
        plain = (d.get("plainLyrics") or "").strip()
        if synced:
            return (synced, _parse_lrc(synced), "lrclib")
        if plain:
            return (plain, None, "lrclib")
        return None

    base = urllib.parse.urlencode({"artist_name": artist, "track_name": title})
    try:
        got = _pick(_fetch(f"https://lrclib.net/api/get?{base}"))
        if got:
            return got
    except Exception as e:
        log.debug("lrclib get failed for %s — %s: %s", artist, title, e)

    try:
        results = _fetch(f"https://lrclib.net/api/search?{base}")
        if isinstance(results, list):
            # Prefer a hit whose length is close to ours — same title by the same
            # artist can be a radio edit, a remix or a live version.
            def score(d):
                dd = d.get("duration") or 0
                return abs(dd - dur_s) if dur_s and dd else 999
            for d in sorted(results, key=score)[:5]:
                got = _pick(d)
                if got:
                    return got
    except Exception as e:
        log.debug("lrclib search failed for %s — %s: %s", artist, title, e)
    return None


@router.get("/lyrics/{track_id}")
def get_lyrics(track_id: str, user: User = Depends(require_user)):
    # Takes a STRING, not an int: the clients hand back the ratingKey they were
    # given, and this API hands out `t7937`, not `7937`. Typed as int, every
    # real request from the app came back 422 and the app quietly showed "no
    # lyrics" — a broken route that looked like a working feature.
    raw = str(track_id).strip()
    if raw[:1] in ("t", "T"):
        raw = raw[1:]
    try:
        track_id = int(raw)
    except ValueError:
        raise HTTPException(400, "Bad track id")

    db = SessionLocal()
    try:
        t = db.get(Track, track_id)
        if not t:
            raise HTTPException(404, "No such track")

        cache_file = CACHE / f"{track_id}.json"
        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                # A "nothing found" answer is only final if every source was
                # actually tried. Cached while offline and now allowed online →
                # look again, otherwise switching the flag on would appear to do
                # nothing for every track played in the meantime.
                stale = (not cached.get("source")
                         and ONLINE and not cached.get("onlineTried"))
                if not stale:
                    return cached
            except (OSError, ValueError):
                pass   # unreadable cache → just look it up again

        artist = (t.artist_name or "") if hasattr(t, "artist_name") else ""
        found = (
            _from_sidecar(t.path)
            or _from_tags(t.path)
            or _from_lrclib(artist or _artist_of(db, t), t.title or "",
                            _album_of(db, t), int((t.duration_ms or 0) / 1000))
        )

        if found:
            text, synced, source = found
            payload = {"trackId": track_id, "source": source,
                       "synced": synced or [], "plain": text,
                       "onlineTried": ONLINE}
        else:
            payload = {"trackId": track_id, "source": None, "synced": [],
                       "plain": "", "onlineTried": ONLINE}

        # Cache negatives too — otherwise every replay of a track without lyrics
        # hits the network again.
        try:
            cache_file.write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            pass
        return payload
    finally:
        db.close()


def _artist_of(db, t: Track) -> str:
    try:
        from ..models import Artist
        a = db.get(Artist, t.artist_id) if getattr(t, "artist_id", None) else None
        return a.name if a else ""
    except Exception:
        return ""


def _album_of(db, t: Track) -> str:
    try:
        from ..models import Album
        a = db.get(Album, t.album_id) if getattr(t, "album_id", None) else None
        return a.title if a else ""
    except Exception:
        return ""
