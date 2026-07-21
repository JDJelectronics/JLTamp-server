"""Library scanner. Walks a *library's* folders (Plex-style: the admin defines
libraries + folders and triggers scans), reads tags with mutagen, groups into
artists/albums/tracks, and extracts cover art. Incremental: unchanged files
(same path + mtime) are skipped, so rescans are cheap.

Artists and albums are scoped per-library (the same artist in the MP3 library
and the FLAC library are separate rows) so per-user library access can filter
cleanly. Runs in a background thread (see main.py) so it never blocks the API.

The music folders are mounted READ-ONLY — the scanner only ever reads them.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time
from fnmatch import fnmatch
from pathlib import Path

import mutagen
from sqlalchemy import select, delete, func

# Expose the ReplayGain TXXX frames in mutagen's "easy" mode for MP3 too (FLAC/
# Ogg/Opus already surface `replaygain_track_gain` in easy mode). Best-effort.
try:
    from mutagen.easyid3 import EasyID3
    for _rg in ("replaygain_track_gain", "replaygain_track_peak",
                "replaygain_album_gain", "replaygain_album_peak"):
        try:
            EasyID3.RegisterTXXXKey(_rg, _rg)
        except Exception:
            pass
except Exception:
    pass

from . import config
from .db import SessionLocal
from .models import Artist, Album, Track, PlaylistItem, Library, LibraryFolder

log = logging.getLogger("scanner")

# One scan at a time. `library` names the library currently scanning (or "").
#   current        : "Artist — Album" being read right now (Plex-style progress).
#   new_album_count/new_track_count : totals of freshly-added items this scan.
#   new_albums/new_tracks           : the most-recent N new items, for the UI feed.
_scan_state = {"scanning": False, "library": "", "library_id": 0,
               "last_ms": 0, "tracks": 0, "started": 0, "cancelled": False,
               "current": "", "new_album_count": 0, "new_track_count": 0,
               "new_albums": [], "new_tracks": []}

# Keep only the most-recent N new items in the live feed (like Plex's activity
# panel) so a huge first-ever scan can't grow the state unbounded; the *counts*
# stay exact.
_NEW_KEEP = 60

# Separate accumulator for the post-scan "new music" email digest: EXACT per-
# library counts (so each recipient's mail reflects only libraries they can see)
# plus a capped sample of new albums (with their library) for the list body.
_MAIL_SAMPLE_MAX = 400
_scan_mail: dict = {"albums_by_lib": {}, "tracks_by_lib": {}, "album_sample": []}

# Guards the mutable new-item lists: the scan thread appends while the API thread
# copies them in scan_state(). Cheap; only held for the tiny list op.
_scan_lock = threading.Lock()

# Set by request_stop(); the running scan checks it and bails cleanly, keeping
# whatever it already committed. Cleared at the start of every scan.
_scan_cancel = {"flag": False}


def request_stop() -> dict:
    """Ask a running scan to stop. No-op if nothing is scanning."""
    if _scan_state["scanning"]:
        _scan_cancel["flag"] = True
        log.info("scan stop requested")
    return scan_state()


def scan_state() -> dict:
    with _scan_lock:
        s = dict(_scan_state)
        # Copy the lists so a concurrent append can't mutate them mid-serialize.
        s["new_albums"] = list(_scan_state["new_albums"])
        s["new_tracks"] = list(_scan_state["new_tracks"])
    return s


def _reset_new() -> None:
    """Clear the new-item feed + counters — once per user-initiated scan run."""
    with _scan_lock:
        _scan_state["new_album_count"] = 0
        _scan_state["new_track_count"] = 0
        _scan_state["new_albums"].clear()
        _scan_state["new_tracks"].clear()
        _scan_mail["albums_by_lib"] = {}
        _scan_mail["tracks_by_lib"] = {}
        _scan_mail["album_sample"] = []


def _note_new_album(library_id: int, artist: str, title: str, year) -> None:
    with _scan_lock:
        _scan_state["new_album_count"] += 1
        lst = _scan_state["new_albums"]
        lst.append({"artist": artist, "title": title, "year": year})
        if len(lst) > _NEW_KEEP:
            del lst[0]
        _scan_mail["albums_by_lib"][library_id] = _scan_mail["albums_by_lib"].get(library_id, 0) + 1
        if len(_scan_mail["album_sample"]) < _MAIL_SAMPLE_MAX:
            _scan_mail["album_sample"].append(
                {"library_id": library_id, "artist": artist, "title": title, "year": year})


def _note_new_track(library_id: int, artist: str, title: str, album: str) -> None:
    with _scan_lock:
        _scan_state["new_track_count"] += 1
        lst = _scan_state["new_tracks"]
        lst.append({"artist": artist, "title": title, "album": album})
        if len(lst) > _NEW_KEEP:
            del lst[0]
        _scan_mail["tracks_by_lib"][library_id] = _scan_mail["tracks_by_lib"].get(library_id, 0) + 1


def send_new_music_digest() -> None:
    """Email the per-user 'new music' digest after a scan run. Each opted-in,
    active user gets only the libraries they can access; pending invites and
    users with the toggle off are skipped. Best-effort — never raises."""
    from . import mailer
    from .models import User, UserLibraryAccess
    if not mailer.configured():
        return
    with _scan_lock:
        albums_by_lib = dict(_scan_mail["albums_by_lib"])
        tracks_by_lib = dict(_scan_mail["tracks_by_lib"])
        sample = list(_scan_mail["album_sample"])
    if sum(tracks_by_lib.values()) <= 0:
        return

    db = SessionLocal()
    try:
        users = db.execute(select(User).where(
            User.is_active == True, User.notify_new_music == True)).scalars().all()  # noqa: E712
        for u in users:
            if not u.email or not u.password_hash:
                continue  # skip pending invites (not joined yet)
            if u.is_admin:
                acc = set(albums_by_lib) | set(tracks_by_lib)
            else:
                acc = set(db.execute(select(UserLibraryAccess.library_id).where(
                    UserLibraryAccess.user_id == u.id)).scalars().all())
            t_count = sum(tracks_by_lib.get(l, 0) for l in acc)
            if t_count <= 0:
                continue
            a_count = sum(albums_by_lib.get(l, 0) for l in acc)
            albums = [a for a in sample if a["library_id"] in acc]
            try:
                mailer.send_new_music(u.email, config.SERVER_NAME, a_count, t_count, albums)
            except Exception:
                log.exception("new-music mail to %s failed", u.email)
    finally:
        db.close()


# ── tag helpers ──────────────────────────────────────────────────────────────
def _first(tags, *keys) -> str:
    for k in keys:
        v = tags.get(k)
        if v:
            if isinstance(v, list):
                v = v[0]
            s = str(v).strip()
            if s:
                return s
    return ""


def _parse_gain(s: str) -> float | None:
    """Parse a ReplayGain value like "-6.48 dB" / "+3.2" into a float (dB)."""
    if not s:
        return None
    m = re.search(r"[-+]?\d+(?:\.\d+)?", str(s))
    if not m:
        return None
    try:
        v = float(m.group())
    except ValueError:
        return None
    # Sanity clamp — a tag that says ±60 dB is corrupt, not music.
    return v if -40.0 <= v <= 40.0 else None


def _int(s: str) -> int | None:
    if not s:
        return None
    s = s.split("/")[0].split("-")[0].strip()
    try:
        return int(s)
    except ValueError:
        return None


def _year(s: str) -> int | None:
    """A release year, from whatever the tag actually holds.

    Date tags come in every shape: "2018", "2018-07-21", "20180721", "2018/07".
    _int() only strips '-' and '/', so a compact "20180721" used to be stored as
    the year 20180721 — which then sorts and filters as if it were the year
    twenty-million. Take the leading 4 digits and only accept a plausible year.
    """
    if not s:
        return None
    digits = ""
    for ch in str(s).strip():
        if ch.isdigit():
            digits += ch
            if len(digits) == 4:
                break
        elif digits:
            break
    if len(digits) != 4:
        return None
    y = int(digits)
    return y if 1000 <= y <= 2999 else None


def _sort_key(s: str) -> str:
    s = (s or "").strip().lower()
    for art in ("the ", "a ", "an ", "de ", "het ", "een "):
        if s.startswith(art):
            return s[len(art):]
    return s


def _read(path: Path):
    try:
        mf = mutagen.File(str(path), easy=True)
    except Exception as e:
        log.warning("tag read failed %s: %s", path, e)
        return None, None
    if mf is None:
        return None, None
    return (mf.tags or {}), mf.info


def _container_codec(path: Path, info) -> tuple[str, str, int]:
    ext = path.suffix.lower().lstrip(".")
    container = ext
    codec = ext
    bits = getattr(info, "bits_per_sample", 0) or 0
    cls = type(info).__name__.lower()
    if "mp3" in cls:
        container, codec = "mp3", "mp3"
    elif "flac" in cls:
        container, codec = "flac", "flac"
    elif "mp4" in cls or "m4a" in cls or "aac" in cls:
        container = "mp4"
        codec = "alac" if ext in ("alac",) else "aac"
    elif "wave" in cls or "wav" in cls:
        container, codec = "wav", "pcm"
    return container, codec, bits


# ── album art extraction ─────────────────────────────────────────────────────
def _folder_cover(folder: Path) -> Path | None:
    for name in config.COVER_NAMES:
        for ext in config.COVER_EXTS:
            p = folder / f"{name}{ext}"
            if p.exists():
                return p
    for p in sorted(folder.glob("*")):
        if p.suffix.lower() in config.COVER_EXTS:
            return p
    return None


def _embedded_art_bytes(path: Path) -> bytes | None:
    try:
        mf = mutagen.File(str(path))
    except Exception:
        return None
    if mf is None:
        return None
    try:
        for k in mf.tags.keys() if mf.tags else []:
            if k.startswith("APIC"):
                return mf.tags[k].data
    except Exception:
        pass
    pics = getattr(mf, "pictures", None)
    if pics:
        return pics[0].data
    try:
        covr = mf.tags.get("covr") if mf.tags else None
        if covr:
            return bytes(covr[0])
    except Exception:
        pass
    return None


def _resolve_album_art(album_id: int, sample_track: Path) -> str | None:
    """Folder cover if present, else embedded art extracted once to the data dir
    (NEVER back to the read-only music folder)."""
    cover = _folder_cover(sample_track.parent)
    if cover:
        return str(cover)
    data = _embedded_art_bytes(sample_track)
    if data:
        out = config.ARTWORK_DIR / f"album_{album_id}.jpg"
        try:
            out.write_bytes(data)
            return str(out)
        except Exception as e:
            log.warning("art write failed: %s", e)
    return None


# ── ignore files (Plex-style) ────────────────────────────────────────────────
# Drop a `.plexignore` (or `.jltampignore`) file in a folder to keep the scanner
# out of it — exactly like Plex. An EMPTY file, or one containing `*`, ignores the
# whole folder and everything under it. Otherwise each non-comment line is a glob
# pattern (e.g. `Demos`, `*.tmp`, `Various Artists`) matched against the files and
# subfolders in that directory. Reading works fine on the read-only NAS mount;
# create the files from the NAS side.
IGNORE_FILES = (".plexignore", ".jltampignore")


def _load_ignore(dir_path: Path) -> list[str] | None:
    """Patterns for a directory, or None if it has no ignore file. `['*']` means
    'ignore the whole folder'."""
    for name in IGNORE_FILES:
        f = dir_path / name
        if f.is_file():
            try:
                lines = [ln.strip() for ln in f.read_text(errors="ignore").splitlines()]
                pats = [ln for ln in lines if ln and not ln.startswith("#")]
                return pats if pats else ["*"]  # empty file → ignore everything
            except OSError:
                return ["*"]
    return None


def _iter_audio(root: Path):
    if not root.exists():
        log.warning("library folder %s does not exist", root)
        return
    for dirpath, dirnames, filenames in os.walk(root):
        d = Path(dirpath)
        pats = _load_ignore(d)
        if pats == ["*"]:
            log.info("ignoring folder (ignore file): %s", d)
            dirnames[:] = []  # prune everything below
            continue
        if pats:
            # Skip matching subfolders and matching files in this directory.
            dirnames[:] = [dn for dn in dirnames if not any(fnmatch(dn, p) for p in pats)]
        for fn in filenames:
            if pats and any(fnmatch(fn, p) for p in pats):
                continue
            p = d / fn
            if p.suffix.lower() in config.AUDIO_EXTS:
                yield p


# ── main scan (per library) ──────────────────────────────────────────────────
def scan_library(library_id: int, full: bool = False, keep_new: bool = False) -> dict:
    if _scan_state["scanning"]:
        return scan_state()
    # Standalone scan: start the new-item feed fresh. scan_all() passes
    # keep_new=True so the feed accumulates across all libraries in one run.
    if not keep_new:
        _reset_new()
    db = SessionLocal()
    try:
        lib = db.get(Library, library_id)
        if not lib:
            return scan_state()
        folders = [f.path for f in db.execute(
            select(LibraryFolder).where(LibraryFolder.library_id == library_id)).scalars()]
        _scan_cancel["flag"] = False
        _scan_state.update(scanning=True, started=int(time.time()),
                           library=lib.name, library_id=library_id, tracks=0,
                           cancelled=False, current="")
        t0 = time.time()

        # Existing tracks in THIS library, keyed by path (incremental + prune).
        existing = {t.path: t for t in db.execute(
            select(Track).where(Track.library_id == library_id)).scalars()}
        seen_paths: set[str] = set()

        artist_cache: dict[str, Artist] = {}
        album_cache: dict[tuple, Album] = {}

        def get_artist(name: str) -> Artist:
            key = name.lower()
            a = artist_cache.get(key)
            if a:
                return a
            a = db.execute(select(Artist).where(
                Artist.library_id == library_id, Artist.name == name)).scalar_one_or_none()
            if not a:
                a = Artist(library_id=library_id, name=name, sort_name=_sort_key(name),
                           added_at=int(time.time()))
                db.add(a)
                db.flush()
            artist_cache[key] = a
            return a

        def get_album(title: str, artist: Artist, year, genre: str, sample: Path) -> Album:
            key = (artist.id, title.lower())
            al = album_cache.get(key)
            if al:
                return al
            al = db.execute(select(Album).where(
                Album.library_id == library_id, Album.artist_id == artist.id,
                Album.title == title)).scalar_one_or_none()
            if not al:
                al = Album(library_id=library_id, title=title, sort_title=_sort_key(title),
                           artist_id=artist.id, artist_name=artist.name, year=year,
                           genre=genre, added_at=int(time.time()))
                db.add(al)
                db.flush()
                al.art_path = _resolve_album_art(al.id, sample)
                if al.art_path and not artist.art_path:
                    artist.art_path = al.art_path
                _note_new_album(library_id, artist.name, title, year)
            album_cache[key] = al
            return al

        count = 0
        cancelled = False
        for folder in folders:
            if cancelled:
                break
            for path in _iter_audio(Path(folder)):
                # Stop cleanly on request — keep everything committed so far, but
                # DON'T prune (missing files below would look "vanished").
                if _scan_cancel["flag"]:
                    cancelled = True
                    log.info("[%s] scan cancelled after %d tracks", lib.name, count)
                    break
                sp = str(path)
                seen_paths.add(sp)
                try:
                    st = path.stat()
                except OSError:
                    continue
                prev = existing.get(sp)
                if prev and not full and abs(prev.mtime - st.st_mtime) < 1.0:
                    continue

                tags, info = _read(path)
                if info is None:
                    continue

                title = _first(tags, "title") or path.stem
                track_artist = _first(tags, "artist") or "Unknown Artist"
                album_artist = _first(tags, "albumartist") or track_artist
                album_title = _first(tags, "album") or "Unknown Album"
                genre = _first(tags, "genre")
                year = _year(_first(tags, "date", "originaldate", "year"))
                track_no = _int(_first(tags, "tracknumber"))
                disc_no = _int(_first(tags, "discnumber"))

                container, codec, bits = _container_codec(path, info)
                # Loudness normalisation: prefer the track ReplayGain tag, fall
                # back to album gain. NULL when neither is present (phase 2 will
                # measure those on demand). Cheap — no audio decode.
                gain_db = _parse_gain(_first(tags, "replaygain_track_gain",
                                             "replaygain_album_gain"))
                duration_ms = int((getattr(info, "length", 0) or 0) * 1000)
                bitrate = int((getattr(info, "bitrate", 0) or 0) / 1000)
                channels = getattr(info, "channels", 2) or 2
                samplerate = getattr(info, "sample_rate", 44100) or 44100

                artist = get_artist(album_artist)
                album = get_album(album_title, artist, year, genre, path)
                # What we're reading right now — shown live in the admin UI.
                _scan_state["current"] = f"{album_artist} — {album_title}"
                orig_artist = track_artist if track_artist and track_artist != album_artist else ""

                fields = dict(
                    library_id=library_id,
                    title=title, sort_title=_sort_key(title),
                    album_id=album.id, album_title=album.title,
                    artist_id=artist.id, artist_name=artist.name, album_artist=album_artist,
                    orig_artist=orig_artist,
                    track_no=track_no, disc_no=disc_no, duration_ms=duration_ms, genre=genre, year=year,
                    path=sp, ext=path.suffix.lower(), container=container, codec=codec,
                    bitrate=bitrate, channels=channels, samplerate=samplerate, bits=bits,
                    size=st.st_size, mtime=st.st_mtime, art_path=album.art_path,
                    added_at=int(st.st_ctime), gain_db=gain_db,
                )
                if prev:
                    for k, v in fields.items():
                        setattr(prev, k, v)
                else:
                    db.add(Track(**fields))
                    _note_new_track(library_id, artist.name, title, album.title)
                count += 1
                # Commit FREQUENTLY: get_artist/get_album flush() opens a write
                # transaction (holding SQLite's single write lock) that stays
                # open until commit. With slow NFS tag reads in between, a large
                # batch would hold the lock for minutes and make concurrent
                # writes (login, likes) fail with "database is locked". Small
                # batches keep the lock window to a few seconds.
                if count % 25 == 0:
                    db.commit()
                    _scan_state["tracks"] = count
                if count % 500 == 0:
                    log.info("[%s] scanned %d tracks…", lib.name, count)

        # Prune tracks whose files vanished — but NOT after a cancel: the scan
        # never reached the rest of the library, so those files aren't gone.
        if not cancelled:
            removed = [t for p, t in existing.items() if p not in seen_paths]
            for t in removed:
                db.execute(delete(PlaylistItem).where(PlaylistItem.track_id == t.id))
                db.delete(t)
            db.commit()

        _recompute_aggregates(db, library_id)
        db.commit()

        lib.track_count = db.execute(select(func.count(Track.id)).where(
            Track.library_id == library_id)).scalar() or 0
        lib.last_scan_at = int(time.time())
        lib.last_scan_ms = int((time.time() - t0) * 1000)
        db.commit()
        _scan_state["tracks"] = lib.track_count
    finally:
        db.close()
        dt = int((time.time() - _scan_state.get("started", time.time())))
        # Do NOT clear the flag here — scan_all() checks it AFTER we return to
        # decide whether to move on to the next library. Clearing it now let a
        # cancel stop only the current library while the rest kept scanning.
        was_cancelled = _scan_cancel["flag"]
        # Clear the live "current" label but KEEP the new-item feed + counts so
        # the admin UI can show what was added after the scan finishes.
        _scan_state.update(scanning=False, last_ms=dt * 1000, library="",
                           library_id=0, cancelled=was_cancelled, current="")
        log.info("scan %s", "cancelled" if was_cancelled else "done")
    # Skip enrichment (more network work) if the admin stopped the scan.
    if not was_cancelled:
        try:
            from .metadata import enrich_library
            enrich_library(library_id)
        except Exception:
            log.exception("enrichment failed")
        # Standalone single-library scan → send the digest now. During a
        # "scan all" run (keep_new=True) the feed accumulates and scan_all()
        # sends one digest for the whole run instead.
        if not keep_new:
            try:
                send_new_music_digest()
            except Exception:
                log.exception("new-music digest failed")
    return scan_state()


def _recompute_aggregates(db, library_id: int) -> None:
    rows = db.execute(
        select(Track.album_id, func.count(Track.id), func.coalesce(func.sum(Track.duration_ms), 0))
        .where(Track.library_id == library_id)
        .group_by(Track.album_id)
    ).all()
    counts = {aid: (c, d) for aid, c, d in rows}
    for al in db.execute(select(Album).where(Album.library_id == library_id)).scalars():
        c, d = counts.get(al.id, (0, 0))
        al.track_count, al.duration_ms = c, d
        if c == 0:
            db.delete(al)
    used = {aid for (aid,) in db.execute(
        select(Album.artist_id).where(Album.library_id == library_id).distinct()).all()}
    for ar in db.execute(select(Artist).where(Artist.library_id == library_id)).scalars():
        if ar.id not in used:
            db.delete(ar)


def scan_all(full: bool = False) -> None:
    """Scan every configured library in turn (used by manual 'scan all')."""
    db = SessionLocal()
    try:
        ids = [l.id for l in db.execute(select(Library)).scalars()]
    finally:
        db.close()
    _reset_new()  # one shared new-item feed for the whole multi-library run
    cancelled = False
    for lib_id in ids:
        scan_library(lib_id, full=full, keep_new=True)
        if _scan_cancel["flag"]:
            cancelled = True
            log.info("scan-all stopped — remaining libraries skipped")
            break
    _scan_cancel["flag"] = False  # clear once the whole run is over
    # One digest for the whole run — only if it actually finished.
    if not cancelled:
        try:
            send_new_music_digest()
        except Exception:
            log.exception("new-music digest failed")
