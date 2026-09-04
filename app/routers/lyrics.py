"""Lyrics for a track — local sources first, an online lookup as fallback.

Measured on the real library: **zero** `.lrc` sidecars, and almost no embedded
lyrics either — well under 1% of the tracks, all of them MP3s.

An earlier note here claimed "roughly a quarter of the MP3s", which counted the
wrong thing: ~9.5% of the MP3s carry a `USLT` frame, but 36 of every 38 of those
frames are EMPTY — a tagger wrote the frame and no words. Sampled 400 files at
random: 38 had the tag, 2 had text. So a local-only feature shows nothing for
99% of the library, which is why there is an online fallback (LRCLIB — free, no
API key, and the only source that returns *synced* lines).

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

With it off, expect almost the whole library to have no lyrics.

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
from pydantic import BaseModel

from .. import config
from ..db import SessionLocal
from ..deps import require_user, require_admin
from ..models import Track, User

router = APIRouter(tags=["lyrics"])
log = logging.getLogger("lyrics")

# OFF by default — the online step is opt-in, never a surprise. A default of
# "true" means a rebuild or a fresh install starts talking to a third party
# without anyone deciding to; for a self-hosted server that is the wrong way
# round.
#
# The env var is only the DEFAULT now. Once an admin flips the switch in the
# app the answer lives in the database, because changing an env var means
# rebuilding the container — which drops every live Listen Together session and
# any running index pass. A privacy switch you can only use by disrupting
# everyone is a switch nobody uses.
ONLINE_DEFAULT = os.getenv("JLTAMP_LYRICS_ONLINE", "false").lower() in ("1", "true", "yes")
_ONLINE_KEY = "lyrics_online"


def online_enabled() -> bool:
    """Is the LRCLIB fallback allowed right now? Database first, env as the
    default for a server nobody has configured yet."""
    from ..models import Meta
    db = SessionLocal()
    try:
        row = db.get(Meta, _ONLINE_KEY)
        if row is None:
            return ONLINE_DEFAULT
        return (row.value or "").lower() in ("1", "true", "yes")
    finally:
        db.close()


def set_online_enabled(enabled: bool) -> bool:
    from ..models import Meta
    db = SessionLocal()
    try:
        row = db.get(Meta, _ONLINE_KEY)
        if row is None:
            row = Meta(key=_ONLINE_KEY)
            db.add(row)
        row.value = "true" if enabled else "false"
        db.commit()
        return enabled
    finally:
        db.close()
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
    if not online_enabled() or not (artist and title):
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


# Artist names that identify nobody. A compilation stores "Various Artists" as
# the album artist, and a track with no artist of its own inherits it — so the
# lookup went out as "Various Artists — <title>", which cannot match anything.
_NO_ARTIST = {"", "various artists", "various", "va", "verzamelaars",
              "diverse artiesten", "unknown artist", "unknown"}


def _is_placeholder_artist(name: str) -> bool:
    return re.sub(r"[^a-z ]+", "", (name or "").lower()).strip() in _NO_ARTIST


def _from_lrclib_by_title(title: str, dur_s: int):
    """Search LRCLIB on the TITLE alone, for tracks whose artist is a
    placeholder. Two guards, because a bare title is a weak key:

      • the returned title must match ours exactly once normalised, so "Intro"
        cannot quietly collect the lyrics of a different "Intro (Live)";
      • the duration must be within 5 seconds, which is what separates two
        recordings of the same song from each other.

    Without a duration of our own there is nothing to check against, so we
    decline rather than guess.
    """
    if not online_enabled() or not title or not dur_s:
        return None
    import urllib.parse
    import urllib.request

    want = re.sub(r"[^a-z0-9]+", "", title.lower())
    url = "https://lrclib.net/api/search?" + urllib.parse.urlencode({"track_name": title})
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "JLTamp (self-hosted music server)"})
        with urllib.request.urlopen(req, timeout=8) as r:
            results = json.loads(r.read().decode("utf-8"))
    except Exception as e:
        log.debug("lrclib title search failed for %s: %s", title, e)
        return None
    if not isinstance(results, list):
        return None

    for d in sorted(results, key=lambda d: abs((d.get("duration") or 0) - dur_s)):
        if re.sub(r"[^a-z0-9]+", "", (d.get("trackName") or "").lower()) != want:
            continue
        if abs((d.get("duration") or 0) - dur_s) > 5:
            break          # sorted by distance, so everything after is worse
        synced = (d.get("syncedLyrics") or "").strip()
        plain = (d.get("plainLyrics") or "").strip()
        if synced:
            return (synced, _parse_lrc(synced), "lrclib")
        if plain:
            return (plain, None, "lrclib")
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

        # The INDEX first. It already holds the answer for most tracks — the
        # local text and, since the online pass, what LRCLIB had. Re-deriving it
        # here was not just wasteful: this path asks with artist_name, which is
        # the album artist ("Various Artists" on any compilation), while the
        # index asks with the track's OWN artist and falls back to a title
        # search. So a track with perfectly good indexed lyrics showed "none",
        # and the miss was then cached. Measured on the real server: of 20
        # tracks anyone had opened, 8 said "no lyrics" and all 8 were indexed.
        from ..models import TrackLyrics
        row = db.get(TrackLyrics, track_id)
        if row is not None and row.found and row.text:
            return {"trackId": track_id, "source": row.source or "index",
                    "synced": _parse_lrc(row.text) or [], "plain": row.text,
                    "onlineTried": online_enabled()}

        cache_file = CACHE / f"{track_id}.json"
        if cache_file.exists():
            try:
                cached = json.loads(cache_file.read_text(encoding="utf-8"))
                # A "nothing found" answer is only final if every source was
                # actually tried. Cached while offline and now allowed online →
                # look again, otherwise switching the flag on would appear to do
                # nothing for every track played in the meantime.
                stale = (not cached.get("source")
                         and online_enabled() and not cached.get("onlineTried"))
                if not stale:
                    return cached
            except (OSError, ValueError):
                pass   # unreadable cache → just look it up again

        # The track's OWN artist first: artist_name is the album/grouping
        # artist, so on a compilation it is the phrase "Various Artists" — a
        # question LRCLIB cannot answer. Same rule the indexer uses.
        artist = (t.orig_artist or t.artist_name or "").strip() or _artist_of(db, t)
        dur_s = int((t.duration_ms or 0) / 1000)
        found = (
            _from_sidecar(t.path)
            or _from_tags(t.path)
            or (_from_lrclib_by_title(t.title or "", dur_s)
                if _is_placeholder_artist(artist)
                else _from_lrclib(artist, t.title or "", _album_of(db, t), dur_s))
        )

        if found:
            text, synced, source = found
            payload = {"trackId": track_id, "source": source,
                       "synced": synced or [], "plain": text,
                       "onlineTried": online_enabled()}
        else:
            payload = {"trackId": track_id, "source": None, "synced": [],
                       "plain": "", "onlineTried": online_enabled()}

        # Cache negatives too — otherwise every replay of a track without lyrics
        # hits the network again.
        try:
            cache_file.write_text(json.dumps(payload), encoding="utf-8")
        except OSError:
            pass

        # Opening the lyrics of a track also indexes it, so search coverage
        # grows as the library gets played even if the admin never runs a pass.
        # Online hits included since 2026-08-01 — the owner asked for search to
        # cover the whole library, which means the index holds LRCLIB text too.
        if found:
            try:
                from .. import lyrics_index
                text, synced_lines, source = found
                lyrics_index.index_text(db, t, text, source, bool(synced_lines))
                db.commit()
            except Exception as e:
                log.debug("index-on-view failed for %s: %s", track_id, e)

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


# ── search ───────────────────────────────────────────────────────────────────
# Getting a line stuck in your head and not knowing the title is the oldest
# problem in music. The index that makes this answerable is built separately
# (see lyrics_index.py) because it has to read every file once; searching it
# afterwards is a single query.

@router.get("/search/lyrics")
def search_lyrics(q: str = "", limit: int = 50, user: User = Depends(require_user)):
    """Tracks whose lyrics contain `q`, each with the line it was found on."""
    from .. import lyrics_index
    from ..deps import accessible_library_ids
    from ..serializers import track_dict, container

    q = (q or "").strip()
    if len(q) < 3:
        # Two letters match half the library and tell the searcher nothing.
        return container([], extra={"query": q, "tooShort": True})

    db = SessionLocal()
    try:
        allowed = accessible_library_ids(db, user)
        hits = lyrics_index.search(db, q, allowed, limit=max(1, min(200, limit)))
        items = []
        for track, snippet in hits:
            d = track_dict(track)
            d["lyricSnippet"] = snippet
            items.append(d)
        return container(items, extra={"query": q})
    finally:
        db.close()


@router.get("/lyrics/index/status")
def lyrics_index_status(user: User = Depends(require_user)):
    """Progress of the index pass, plus how much of the library it covers.
    Visible to any user: it explains why a search comes back empty."""
    from .. import lyrics_index
    from ..models import TrackLyrics

    db = SessionLocal()
    try:
        st = lyrics_index.status()
        st["indexed"] = db.query(TrackLyrics).count()
        st["withLyrics"] = db.query(TrackLyrics).filter(TrackLyrics.found.is_(True)).count()
        st["tracks"] = db.query(Track).count()
        # So the admin screen can show the switch in its real position rather
        # than guessing from whatever it last sent.
        st["online"] = online_enabled()
        return st
    finally:
        db.close()


class OnlineBody(BaseModel):
    enabled: bool


@router.post("/lyrics/online")
def set_lyrics_online(body: OnlineBody, user: User = Depends(require_admin)):
    """Allow or forbid the LRCLIB fallback. Takes effect immediately — no
    restart, so flipping it never costs anyone their session.

    When on, a lookup sends the artist and the title of one track. Nothing
    else: not the file, not the library, nothing about who is listening.
    """
    return {"ok": True, "online": set_online_enabled(bool(body.enabled))}


@router.post("/lyrics/index")
def lyrics_index_start(force: bool = False, online: bool = False,
                       user: User = Depends(require_admin)):
    """Walk the library and index every local lyric. Admin only — it reads the
    whole music mount. Resumable and mtime-skipping, so re-running after a scan
    costs seconds; `force=true` re-reads everything anyway.

    `online=true` adds a second pass that asks LRCLIB about every track still
    without lyrics and indexes what it finds. Hours, not seconds: one request per
    unique song, a few per second. Needs JLTAMP_LYRICS_ONLINE on, and only
    artist + title ever leave this server.
    """
    from .. import lyrics_index
    if online and not online_enabled():
        raise HTTPException(400, "Online lyrics are switched off on this server")
    return lyrics_index.start(force=force, online=online)


@router.delete("/lyrics/index")
def lyrics_index_stop(user: User = Depends(require_admin)):
    """Ask a running pass to stop after the current batch."""
    from .. import lyrics_index
    lyrics_index.request_stop()
    return lyrics_index.status()
