"""Build and query the lyrics search index.

Displaying lyrics and searching them are different problems. Display reads one
file when you open one track — cheap, and that is what routers/lyrics.py does.
Search has to look at everything at once, which means a tag read per track over
NFS: minutes per query, on a read-only mount, for one search box.
So the local text is copied into the database once and queried from there.

Local sources first — `.lrc`/`.txt` sidecars and embedded tags. Those are free
and private, and on a typical library they cover well under 1% of the tracks.

The online pass (LRCLIB) is SEPARATE and opt-in twice over: the server must have
JLTAMP_LYRICS_ONLINE on, and the caller must ask for `online=true`. It writes
what it finds into the index, so search covers those tracks too — the owner
asked for that on 2026-08-01, knowing it makes part of the index a copy of
LRCLIB's.

Being a guest on someone else's free service is the whole design of that pass:
requests are deduplicated by artist+title (many tracks share a pair),
throttled to a few per second, backed off on errors, and every answer including
a miss is written down so a re-run never asks twice.

The indexer is resumable and skips files whose mtime has not moved, so running
it again after adding an album costs seconds rather than a full pass.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
import unicodedata

from sqlalchemy import select

from .db import SessionLocal
from .models import Track, TrackLyrics

log = logging.getLogger("lyrics-index")

# Progress for the admin UI. Plain dict + a lock: one indexer runs at a time.
_state = {"running": False, "done": 0, "total": 0, "found": 0,
          "started_at": 0, "finished_at": 0, "error": "",
          # The online pass reports separately: it is the slow half, and "3000
          # of 63000 looked up" is the number someone watching actually wants.
          "phase": "", "online_done": 0, "online_total": 0, "online_found": 0}
_lock = threading.Lock()
_stop = threading.Event()


def status() -> dict:
    with _lock:
        return dict(_state)


def request_stop() -> None:
    _stop.set()


# ── normalisation ────────────────────────────────────────────────────────────
_LRC_STAMP = re.compile(r"\[\d+:\d+(?:[.:]\d+)?\]")


def normalise(s: str) -> str:
    """Fold a string to the form the index is searched in: no accents, no case,
    no punctuation, single spaces. Both the stored text and the query go through
    this, which is what lets "dont stop" match "Don't stop"."""
    s = _LRC_STAMP.sub(" ", s or "")
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = s.lower()
    # Apostrophes JOIN a word — they are deleted, not spaced. Turning them into
    # spaces gave "don t stop", so searching "dont stop" found nothing, which is
    # exactly the phrase a person types when they only know how it sounds.
    s = re.sub(r"['‘’ʼ`]", "", s)
    s = re.sub(r"[^a-z0-9\s]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


# ~3 requests a second. LRCLIB is free and community-run; this pass asks it
# one question per track, so it goes at the pace of a person browsing, not a crawler.
ONLINE_INTERVAL_S = 0.35
# Give up on the online pass if the service keeps failing — either it is down or
# it has had enough of us. Either way, stop asking.
ONLINE_MAX_CONSECUTIVE_ERRORS = 25

# Marks a track we DID ask LRCLIB about and it had nothing. Distinct from the
# empty source of "no local lyrics, never asked", so a re-run skips it.
MISS = "lrclib-miss"


def _extract(path: str) -> tuple[str, str, bool] | None:
    """(text, source, synced) from the local sources, or None. Imported lazily:
    routers.lyrics pulls in FastAPI, and the scanner has no business doing so."""
    from .routers.lyrics import _from_sidecar, _from_tags
    for fn in (_from_sidecar, _from_tags):
        try:
            hit = fn(path)
        except Exception as e:      # a malformed tag must not stop the pass
            log.debug("extract failed for %s: %s", path, e)
            continue
        if hit:
            text, synced_lines, source = hit
            return text, source, bool(synced_lines)
    return None


def index_track(db, track: Track, *, force: bool = False) -> bool:
    """Index one track. Returns True when lyrics were found. Cheap to call
    repeatedly: an unchanged file is skipped on its mtime."""
    row = db.get(TrackLyrics, track.id)
    try:
        mtime = os.path.getmtime(track.path)
    except OSError:
        mtime = 0.0

    if row and not force and row.mtime == mtime and mtime:
        return row.found

    hit = _extract(track.path)
    text, source, synced = hit if hit else ("", "", False)

    if row is None:
        row = TrackLyrics(track_id=track.id)
        db.add(row)
    row.library_id = track.library_id
    row.text = text
    row.search_text = normalise(text)
    row.source = source
    row.synced = synced
    row.found = bool(text)
    row.mtime = mtime
    row.indexed_at = int(time.time())
    return row.found


def _artist_of(track: Track) -> str:
    """The track's OWN artist first: artist_name is the album/grouping artist,
    which is "Various Artists" on every compilation and would send LRCLIB a
    question it cannot answer."""
    return (track.orig_artist or track.artist_name or "").strip()


def _online_pass(db) -> None:
    """Ask LRCLIB about every track that has no lyrics yet.

    Deduplicated by normalised artist+title, so a library costs one request per
    unique song rather than one per track, and a compilation that repeats a song
    ten times costs one. Every answer is written down, misses included, so a
    second run picks up where this one stopped instead of asking again.
    """
    from .routers.lyrics import (online_enabled, _from_lrclib, _from_lrclib_by_title,
                                 _is_placeholder_artist)
    if not online_enabled():
        log.warning("online pass requested but JLTAMP_LYRICS_ONLINE is off")
        return

    # Everything without lyrics that we have not already asked about — plus the
    # ones we asked the WRONG QUESTION about. A miss recorded while the artist
    # was a placeholder came from "Various Artists — <title>", which could never
    # have matched; now that those are asked by title, that answer is stale and
    # the track deserves another go. Self-healing: after this pass they are
    # marked again, on the strength of a question that was actually fair.
    candidates = db.execute(
        select(Track).join(TrackLyrics, TrackLyrics.track_id == Track.id, isouter=True)
        .where((TrackLyrics.found.is_(None)) | (TrackLyrics.found.is_(False)))
    ).scalars().all()

    rows = []
    stale = 0
    for t in candidates:
        row = db.get(TrackLyrics, t.id)
        if row is not None and row.source == MISS:
            if not _is_placeholder_artist(_artist_of(t)):
                continue
            stale += 1
        rows.append(t)
    if stale:
        log.info("re-asking %d tracks whose artist was a placeholder", stale)

    with _lock:
        _state.update(phase="online", online_total=len(rows), online_done=0, online_found=0)

    seen: dict[tuple[str, str], tuple[str, bool] | None] = {}
    done = found = errors = 0
    last_call = 0.0

    for t in rows:
        if _stop.is_set():
            break
        artist, title = _artist_of(t), (t.title or "").strip()
        # For a placeholder artist the question IS the title, so that is the
        # key. Keying on ("various artists", title) would have been fine too,
        # but this keeps the two kinds of question apart in the cache.
        key = (("", normalise(title)) if _is_placeholder_artist(artist)
               else (normalise(artist), normalise(title)))

        if not key[1] or (not key[0] and not _is_placeholder_artist(artist)):
            done += 1
            continue

        if key in seen:
            hit = seen[key]          # another copy of the same song, already asked
        else:
            # Throttle on the CALL, not the loop: cache hits should not be slowed
            # down to the speed of the network.
            wait = ONLINE_INTERVAL_S - (time.monotonic() - last_call)
            if wait > 0:
                time.sleep(wait)
            last_call = time.monotonic()
            try:
                dur_s = int((t.duration_ms or 0) / 1000)
                # A compilation track with no artist of its own would be asked
                # for as "Various Artists — <title>", which matches nothing.
                # Ask on the title instead, with the duration as the referee.
                got = (_from_lrclib_by_title(title, dur_s)
                       if _is_placeholder_artist(artist)
                       else _from_lrclib(artist, title, "", dur_s))
                errors = 0
            except Exception as e:
                errors += 1
                log.debug("lrclib failed for %s — %s: %s", artist, title, e)
                if errors >= ONLINE_MAX_CONSECUTIVE_ERRORS:
                    log.warning("lrclib failed %d times in a row — stopping the online pass", errors)
                    break
                continue
            hit = (got[0], bool(got[1])) if got else None
            seen[key] = hit

        row = db.get(TrackLyrics, t.id)
        if row is None:
            row = TrackLyrics(track_id=t.id, library_id=t.library_id)
            db.add(row)
        if hit:
            row.text, row.synced = hit[0], hit[1]
            row.search_text = normalise(hit[0])
            row.source = "lrclib"
            row.found = True
            found += 1
        else:
            # Write the miss down. Without this the next run asks LRCLIB every
            # question it already answered with "no".
            row.source = MISS
            row.found = False
        row.indexed_at = int(time.time())

        done += 1
        if done % 50 == 0:
            db.commit()
            with _lock:
                _state.update(online_done=done, online_found=found)

    db.commit()
    with _lock:
        _state.update(online_done=done, online_found=found)
    log.info("online lyrics pass: %d looked up, %d found (%d unique questions)",
             done, found, len(seen))


def index_text(db, track: Track, text: str, source: str, synced: bool) -> None:
    """Store lyrics we were HANDED, rather than re-reading the file.

    The display path already has the text — including an online hit, which
    index_track() would never find because it only looks at local sources. Using
    index_track there wrote an empty row over a perfectly good LRCLIB result.
    """
    row = db.get(TrackLyrics, track.id)
    if row is None:
        row = TrackLyrics(track_id=track.id)
        db.add(row)
    row.library_id = track.library_id
    row.text = text or ""
    row.search_text = normalise(text or "")
    row.source = source or ""
    row.synced = bool(synced)
    row.found = bool(text)
    row.indexed_at = int(time.time())
    try:
        row.mtime = os.path.getmtime(track.path)
    except OSError:
        row.mtime = 0.0


def _run(force: bool, online: bool = False) -> None:
    db = SessionLocal()
    try:
        total = db.query(Track).count()
        with _lock:
            _state.update(running=True, done=0, total=total, found=0,
                          started_at=int(time.time()), finished_at=0, error="",
                          phase="local", online_done=0, online_total=0, online_found=0)

        done = found = 0
        # Stream in batches: the library does not fit in memory as ORM objects,
        # and committing often keeps SQLite's single write lock from being held
        # across slow NFS reads (the same reason the scanner commits every 25).
        last_id = 0
        while not _stop.is_set():
            batch = db.execute(
                select(Track).where(Track.id > last_id).order_by(Track.id).limit(200)
            ).scalars().all()
            if not batch:
                break
            for t in batch:
                last_id = t.id
                if index_track(db, t, force=force):
                    found += 1
                done += 1
            db.commit()
            with _lock:
                _state.update(done=done, found=found)

        db.commit()
        with _lock:
            _state.update(done=done, found=found)
        log.info("lyrics index finished: %d/%d tracks have local lyrics", found, done)

        # The local pass is fast and always runs. The online one is the long
        # tail, and only when explicitly asked for.
        if online and not _stop.is_set():
            _online_pass(db)

        with _lock:
            _state.update(running=False, finished_at=int(time.time()), phase="")
    except Exception as e:
        log.exception("lyrics index failed")
        with _lock:
            _state.update(running=False, error=str(e), finished_at=int(time.time()))
    finally:
        db.close()
        _stop.clear()


def start(force: bool = False, online: bool = False) -> dict:
    """Kick off a background pass. A second call while one runs is a no-op."""
    with _lock:
        if _state["running"]:
            return dict(_state)
    threading.Thread(target=_run, args=(force, online), daemon=True,
                     name="lyrics-index").start()
    # Give the thread a moment to publish "running", so the caller's first
    # status poll does not report an idle indexer that is about to start.
    time.sleep(0.05)
    return status()


def search(db, query: str, allowed_library_ids, limit: int = 50):
    """Tracks whose lyrics contain `query`, with a snippet around the hit.

    LIKE on a normalised column rather than FTS5: the table is one row per
    track with lyrics (a few thousand here, not millions), and this keeps the
    schema to something `create_all` can make on an existing database with no
    migration step.
    """
    q = normalise(query)
    if len(q) < 3:
        return []

    stmt = (select(TrackLyrics, Track)
            .join(Track, Track.id == TrackLyrics.track_id)
            .where(TrackLyrics.found.is_(True),
                   TrackLyrics.search_text.like(f"%{q}%")))
    if allowed_library_ids is not None:
        stmt = stmt.where(Track.library_id.in_(allowed_library_ids or [-1]))

    out = []
    for row, track in db.execute(stmt.limit(limit)).all():
        out.append((track, _snippet(row.text, q)))
    return out


def _snippet(text: str, needle: str, width: int = 90) -> str:
    """The line the match is on, plus its neighbour — a lyric hit is only
    recognisable in context, and a raw character window cuts words in half."""
    lines = [l.strip() for l in _LRC_STAMP.sub("", text or "").splitlines()]
    lines = [l for l in lines if l]
    for i, line in enumerate(lines):
        if needle in normalise(line):
            nxt = lines[i + 1] if i + 1 < len(lines) else ""
            joined = f"{line} / {nxt}" if nxt else line
            return joined[:width * 2]
    # The match spans a line break — fall back to a window around it.
    flat = " ".join(lines)
    pos = normalise(flat).find(needle)
    if pos < 0:
        return flat[:width]
    start = max(0, pos - width // 2)
    return ("…" if start else "") + flat[start:start + width] + "…"
