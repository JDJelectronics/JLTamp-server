#!/usr/bin/env python3
"""Ask for each genre by name and see how much of the playlist really is it.

The hand-written benchmark covers moods and eras with keyword lists I chose,
which means it also encodes my guesses. This one uses the library's own genre
tags as ground truth: prompt with the genre name, then measure what fraction
of the 50 returned tracks actually carry that tag.

Genres with few tracks are skipped — a genre with 12 tracks cannot fill a
50-track playlist, so a low score there says nothing about the ranking.

    python3 tests/benchmark_genres.py               # top 25 genres
    python3 tests/benchmark_genres.py --top 40      # more
    python3 tests/benchmark_genres.py --only hardstyle,hardcore
"""
from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import scoring                                     # noqa: E402
from _fixture import load_library, load_store              # noqa: E402

# A genre needs at least this many tracks before asking for 50 of them is fair.
MIN_TRACKS = 60


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--only", default="", help="comma-separated genre names")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--fresh", action="store_true", help="skip the cache")
    args = ap.parse_args()

    tracks = load_library(fresh=args.fresh)
    emb, store = load_store(tracks)
    keys = [t.rating_key for t in tracks]

    counts = Counter(t.genre.strip().lower() for t in tracks if t.genre.strip())
    if args.only:
        wanted = [g.strip().lower() for g in args.only.split(",")]
        cases = [(g, counts.get(g, 0)) for g in wanted]
    else:
        cases = [(g, n) for g, n in counts.most_common(args.top * 2)
                 if n >= MIN_TRACKS][:args.top]

    print(f"{len(tracks)} tracks · {len(counts)} genres · testing {len(cases)}\n")

    total, good = 0.0, 0
    for genre, n in cases:
        if n == 0:
            print(f"  --   {genre!r}: niet in de bibliotheek")
            continue
        query = emb.embed_one(scoring.expand_query(genre))
        sims = store.search(query, keys)
        picked = scoring.select(scoring.score_tracks(genre, tracks, sims))
        if not picked:
            print(f"    0%  {genre!r} — leeg")
            continue
        hit = sum(1 for t in picked if genre in t.genre.strip().lower())
        frac = hit / len(picked)
        total += frac
        good += frac >= 0.5
        mark = "✅" if frac >= 0.5 else ("⚠️ " if frac >= 0.25 else "❌")
        print(f"  {mark} {frac:4.0%}  {genre:24} ({n} tracks in de bibliotheek)")
        if args.verbose and frac < 0.5:
            for t in [t for t in picked if genre not in t.genre.lower()][:3]:
                print(f"         mis: {t.real_artist} - {t.clean_title[:34]} [{t.genre}]")

    if cases:
        print(f"\n  gemiddeld {total / len(cases):.0%} · {good}/{len(cases)} boven 50%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
