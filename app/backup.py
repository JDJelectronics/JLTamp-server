"""Consistent database snapshots for the backup.

The host's nightly rsync copies /data, but a live SQLite database in WAL mode can
be mid-write when rsync reads it — the copied file (and a stale/absent -wal) can
be torn. sqlite3's online backup API reads a transactionally-consistent image
while writers continue, so we periodically write one to `library.snapshot.db`.
The rsync then always has a clean, restorable copy alongside the live file.

Runs inside the app (a daemon thread) so the host backup script needs no change.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
import time

from .config import DATA_DIR

log = logging.getLogger("jltamp.backup")

DB_PATH = DATA_DIR / "library.db"
SNAPSHOT_PATH = DATA_DIR / "library.snapshot.db"
INTERVAL = 6 * 3600  # every 6 hours; the nightly backup picks up the latest


def make_snapshot() -> bool:
    """Write a consistent copy of the DB. Returns False on failure (never raises —
    a backup problem must not take the server down)."""
    try:
        src = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, timeout=30)
        try:
            # Write to a temp file then atomically replace, so a crash mid-copy
            # never leaves a half-written snapshot where a good one used to be.
            tmp = SNAPSHOT_PATH.with_suffix(".tmp")
            dst = sqlite3.connect(str(tmp))
            try:
                src.backup(dst)
            finally:
                dst.close()
            tmp.replace(SNAPSHOT_PATH)
            return True
        finally:
            src.close()
    except Exception as e:  # noqa: BLE001
        log.warning("snapshot failed: %s", e)
        return False


def _loop() -> None:
    # First snapshot shortly after boot, then on the interval.
    time.sleep(120)
    while True:
        if make_snapshot():
            log.info("db snapshot written: %s", SNAPSHOT_PATH.name)
        time.sleep(INTERVAL)


def start() -> None:
    threading.Thread(target=_loop, name="db-snapshot", daemon=True).start()
