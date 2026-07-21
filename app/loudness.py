"""Loudness analysis for volume normalisation.

Fills `Track.gain_db` for tracks that have no ReplayGain tag (the scanner reads
those cheaply). Here we MEASURE the integrated loudness with ffmpeg's EBU R128
`loudnorm` analysis and store the gain needed to reach the reference level, so
the app can play every track at the same perceived volume.

Reference is **-18 LUFS** (ReplayGain 2.0), so measured gains are consistent with
the tags the scanner already read: `gain_db = -18 − integratedLoudness`.

This is heavy (ffmpeg decodes the whole file), so it runs as a background pass
over only the NOT-yet-analysed tracks, is resumable (re-run picks up where it
left off), and never writes to the NAS — it only reads the audio.
"""
from __future__ import annotations

import json
import logging
import shutil
import subprocess
import threading
import time

from sqlalchemy import select

from .db import SessionLocal
from .models import Track

log = logging.getLogger("loudness")

TARGET_LUFS = -18.0
_PER_FILE_TIMEOUT = 180  # seconds; a track that takes longer is skipped

# Progress/state for the admin UI. One pass at a time.
_state = {"running": False, "done": 0, "total": 0, "current": "", "cancelled": False}
_cancel = {"flag": False}
_lock = threading.Lock()


def state() -> dict:
    with _lock:
        return dict(_state)


def request_stop() -> dict:
    with _lock:
        if _state["running"]:
            _cancel["flag"] = True
    return state()


def measure_gain_db(path: str) -> float | None:
    """Integrated-loudness → gain (dB) to reach TARGET_LUFS. None on failure."""
    ff = shutil.which("ffmpeg")
    if not ff:
        return None
    try:
        p = subprocess.run(
            [ff, "-hide_banner", "-nostats", "-i", path,
             "-af", f"loudnorm=I={int(TARGET_LUFS)}:print_format=json",
             "-f", "null", "-"],
            capture_output=True, text=True, timeout=_PER_FILE_TIMEOUT,
        )
        # loudnorm prints its JSON block to stderr; take the LAST {...}.
        err = p.stderr or ""
        start = err.rfind("{")
        end = err.rfind("}")
        if start < 0 or end <= start:
            return None
        data = json.loads(err[start:end + 1])
        input_i = float(data.get("input_i"))
        if input_i <= -70:  # digital silence / unmeasurable
            return None
        gain = TARGET_LUFS - input_i
        return round(max(-40.0, min(40.0, gain)), 2)
    except Exception as e:  # noqa: BLE001
        log.debug("loudness measure failed %s: %s", path, e)
        return None


def analyze_missing(limit: int | None = None) -> dict:
    """Measure + store gain_db for tracks that still have none. Resumable."""
    with _lock:
        if _state["running"]:
            return dict(_state)
        _cancel["flag"] = False
        _state.update(running=True, done=0, current="", cancelled=False)

    db = SessionLocal()
    try:
        q = select(Track).where(Track.gain_db.is_(None)).order_by(Track.id)
        if limit:
            q = q.limit(limit)
        rows = db.execute(q).scalars().all()
        with _lock:
            _state["total"] = len(rows)
        for t in rows:
            if _cancel["flag"]:
                break
            with _lock:
                _state["current"] = f"{t.artist_name} — {t.title}"
            g = measure_gain_db(t.path)
            if g is not None:
                t.gain_db = g
                db.commit()
            with _lock:
                _state["done"] += 1
    finally:
        db.close()
        with _lock:
            _state["running"] = False
            _state["current"] = ""
            _state["cancelled"] = _cancel["flag"]
        log.info("loudness pass %s (%d done)",
                 "cancelled" if _cancel["flag"] else "complete", _state["done"])
    return state()


def analyze_async(limit: int | None = None) -> dict:
    """Kick off analyze_missing() on a background thread."""
    if _state["running"]:
        return state()
    threading.Thread(target=analyze_missing, kwargs={"limit": limit}, daemon=True).start()
    time.sleep(0.05)  # let it flip 'running' so the caller reports the right state
    return state()
