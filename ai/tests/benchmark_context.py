#!/usr/bin/env python3
"""Do context prompts actually land in the right tempo?

The genre and mood benchmarks check *what* a playlist contains. This checks the
thing the audio analysis exists for: a "gym" list must be fast and a "slapen"
list slow, regardless of genre. Nothing measured that until now — the cached
benchmark was even dropping the BPM data, so the whole audio pipeline could
have been dead and every other test would still have passed.

Each case sets a tempo the *median* of the playlist must clear. The median,
not every track, because one slow song on a workout list is forgivable; a
workout list that is slow on the whole is not.

    python3 tests/benchmark_context.py
"""
from __future__ import annotations

import argparse
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import scoring                                      # noqa: E402
from _fixture import load_library, load_store                # noqa: E402

# prompt -> (predicate on the median BPM, human description)
CASES = [
    ("hardloop", lambda m: m >= 140, "snel (≥140)"),
    ("gym", lambda m: m >= 118, "stevig (≥118)"),
    ("workout", lambda m: m >= 118, "stevig (≥118)"),
    ("feest", lambda m: m >= 115, "dansbaar (≥115)"),
    ("hardstyle", lambda m: m >= 140, "snel (≥140)"),
    ("slapen", lambda m: m <= 95, "traag (≤95)"),
    ("rustig", lambda m: m <= 100, "rustig (≤100)"),
    ("meditatie", lambda m: m <= 95, "traag (≤95)"),
    ("focus", lambda m: m <= 110, "kalm (≤110)"),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", action="store_true", help="skip the cache")
    args = ap.parse_args()

    tracks = load_library(fresh=args.fresh)
    have = sum(1 for t in tracks if t.features.get("bpm"))
    print(f"{have}/{len(tracks)} tracks have a measured BPM\n")
    if not have:
        print("❌ no audio features loaded — nothing to test")
        return 1

    emb, store = load_store(tracks)
    keys = [t.rating_key for t in tracks]

    passed = 0
    for prompt, ok, desc in CASES:
        query = emb.embed_one(scoring.expand_query(prompt))
        sims = store.search(query, keys)
        picked = scoring.select(scoring.score_tracks(prompt, tracks, sims))
        bpms = [t.features["bpm"] for t in picked if t.features.get("bpm")]
        if not bpms:
            print(f"  --   {prompt!r}: playlist has no BPM data")
            continue
        med = statistics.median(bpms)
        good = ok(med)
        passed += good
        mark = "✅" if good else "❌"
        print(f"  {mark} mediaan {med:5.0f}  {prompt:12} — wil {desc}")

    print(f"\n  {passed}/{len(CASES)} in het juiste tempo")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
