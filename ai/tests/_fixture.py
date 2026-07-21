"""Shared, cached library snapshot for the benchmarks.

Every measurement paid four minutes of startup — two to fetch 69k tracks, two
more for the genre backfill — before it could score a single prompt. At that
price a hunch does not get checked, and today's wrong diagnoses all came from
not checking. Cached, a run costs seconds.

The cache is a plain JSON of the fields the scorer reads. It expires, and
`--fresh` forces a reload, so a stale snapshot cannot quietly mislead a
measurement the way a stale vector store once did.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config                                       # noqa: E402
from app.embed_client import EmbedClient                     # noqa: E402
from app.embed_store import EmbeddingStore                   # noqa: E402
from app.jltamp_client import JLTampClient, Library, Track    # noqa: E402

CACHE = Path(os.environ.get("AI_TEST_CACHE",
                            str(config.DATA_DIR / "test_library_cache.json")))
MAX_AGE_SEC = 6 * 3600

_FIELDS = ("rating_key", "title", "artist", "orig_artist", "album", "year",
           "genre", "duration_ms", "play_count", "last_played_at", "rating")


def _dump(tracks: list[Track]) -> None:
    rows = [{f: getattr(t, f) for f in _FIELDS} for t in tracks]
    tmp = CACHE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps({"at": time.time(), "tracks": rows}))
    os.replace(tmp, CACHE)


def _load() -> list[Track] | None:
    if not CACHE.exists():
        return None
    try:
        blob = json.loads(CACHE.read_text())
    except (json.JSONDecodeError, OSError):
        return None
    if time.time() - blob.get("at", 0) > MAX_AGE_SEC:
        return None
    return [Track(**row) for row in blob.get("tracks", [])]


def load_library(fresh: bool = False, quiet: bool = False) -> list[Track]:
    """Tracks with genres filled in, from cache when it is recent enough."""
    if not fresh:
        cached = _load()
        if cached:
            if not quiet:
                age = (time.time() - json.loads(CACHE.read_text())["at"]) / 60
                print(f"{len(cached)} tracks (cache, {age:.0f} min oud)")
            return cached

    client = JLTampClient()
    client.login()
    lib = Library(client)
    lib.refresh()
    tracks = lib.snapshot()
    _dump(tracks)
    if not quiet:
        print(f"{len(tracks)} tracks (vers opgehaald)")
    return tracks


def load_store(tracks: list[Track]) -> tuple[EmbedClient, EmbeddingStore]:
    emb = EmbedClient()
    if not emb.probe():
        raise SystemExit("❌ embedding server unreachable")
    store = EmbeddingStore(config.DATA_DIR, emb.dim, emb.model_id)
    return emb, store
