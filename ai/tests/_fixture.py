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
    # `features` (BPM/energy) is stored too, or the cached benchmark silently
    # tests the context prompts with no audio data — which it did, hiding
    # whether the whole audio analysis was even reaching the scorer.
    rows = [{**{f: getattr(t, f) for f in _FIELDS}, "features": t.features}
            for t in tracks]
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
    tracks = []
    for row in blob.get("tracks", []):
        feats = row.pop("features", {})
        t = Track(**row)
        t.features = feats or {}
        tracks.append(t)
    return tracks


def _apply_inferred(tracks: list[Track]) -> int:
    """Overlay the engine's inferred genres, so the benchmark measures what the
    live engine actually serves — not the raw tags the engine no longer uses
    alone. Read from the same file the engine writes."""
    from app import genre_infer
    overlay = genre_infer.load(config.INFERRED_GENRES_FILE)
    if not overlay:
        return 0
    n = 0
    for t in tracks:
        if genre_infer.is_placeholder(t.genre) and overlay.get(t.rating_key):
            t.genre = overlay[t.rating_key]
            n += 1
    return n


def load_library(fresh: bool = False, quiet: bool = False) -> list[Track]:
    """Tracks with genres filled in, from cache when it is recent enough.

    The cache stores raw genres; the inferred overlay is applied on load, so a
    changed overlay is picked up without rebuilding the whole cache."""
    if not fresh:
        cached = _load()
        if cached:
            n = _apply_inferred(cached)
            if not quiet:
                age = (time.time() - json.loads(CACHE.read_text())["at"]) / 60
                extra = f", +{n} afgeleid" if n else ""
                print(f"{len(cached)} tracks (cache, {age:.0f} min oud{extra})")
            return cached

    client = JLTampClient()
    client.login()
    feats = {}
    if config.FEATURES_FILE.exists():
        try:
            feats = json.loads(config.FEATURES_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            feats = {}
    lib = Library(client)
    lib.refresh(feats)
    tracks = lib.snapshot()
    _dump(tracks)                      # cache the RAW genres
    n = _apply_inferred(tracks)
    if not quiet:
        extra = f", +{n} afgeleid" if n else ""
        print(f"{len(tracks)} tracks (vers opgehaald{extra})")
    return tracks


def load_store(tracks: list[Track]) -> tuple[EmbedClient, EmbeddingStore]:
    emb = EmbedClient()
    if not emb.probe():
        raise SystemExit("❌ embedding server unreachable")
    store = EmbeddingStore(config.DATA_DIR, emb.dim, emb.model_id)
    return emb, store
