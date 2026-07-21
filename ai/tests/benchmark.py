#!/usr/bin/env python3
"""Score the engine's playlists against what each prompt actually asked for.

Eyeballing a handful of prompts is how four wrong diagnoses got made in a row:
a change that helps one prompt and quietly wrecks another looks like progress.
This gives one number per prompt and one overall, so a change can be shown to
help rather than argued to.

Each case says what a correct answer looks like — a word that should appear in
the track's metadata, a year range, a language. Judged over the whole 50-track
playlist, not just the head, because the tail is what the user scrolls into.

    python3 tests/benchmark.py            # score the current engine
    python3 tests/benchmark.py --verbose  # and show the misses
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import scoring                                      # noqa: E402
from app.jltamp_client import Track                          # noqa: E402
from _fixture import load_library, load_store                # noqa: E402

# Function words alone were too strict: "Deurdonderen", "Nederpop Medley" and
# "Kaplaarzen" are unmistakably Dutch and contain none of them, so real hits
# were scored as misses and the measurement pointed the wrong way.
_DUTCH_WORDS = re.compile(
    r"\b(de|het|een|ik|jij|jou|wij|zij|niet|maar|mijn|jouw|ons|nog|altijd|"
    r"nooit|zonder|meer|weer|liefde|hart|zon|nacht|dag|jaar|man|vrouw|kind|"
    r"thuis|toch|even|heel|dat|dit|wat|waar|hoe|omdat|want|dus|naar|voor|met|"
    r"ben|bent|zijn|heb|hebt|heeft|wil|kan|gaat|komt|blijf|zing|dans|kom|ga)\b")

# Spelling patterns that barely occur in English: ij, and the Dutch diminutive
# and plural endings. Together with the word list this catches titles like
# "Deurdonderen" and "Kaplaarzen".
_DUTCH_SHAPE = re.compile(
    r"(ij|aa|oo[rmnk]|uu|eu|oe[rmnkg]|sch|[a-z]{3,}tje|[a-z]{3,}je\b|"
    r"[a-z]{3,}en\b|[a-z]{3,}heid\b|[a-z]{3,}lijk)")

_DUTCH_MARKERS = ("neder", "hollan", "vlaam", "dutch")


def looks_dutch(t: Track) -> bool:
    text = f"{t.clean_title} {t.real_artist} {t.genre}".lower()
    if any(m in text for m in _DUTCH_MARKERS):
        return True
    if _DUTCH_WORDS.search(text):
        return True
    # Require two shape hits: one alone fires on English words like "book".
    return len(_DUTCH_SHAPE.findall(text)) >= 2


def has_any(t: Track, words: list[str]) -> bool:
    hay = t.haystack
    return any(w in hay for w in words)


def in_years(t: Track, lo: int, hi: int) -> bool:
    return bool(t.year and lo <= t.year <= hi)


# Each case: prompt -> predicate a correct track satisfies, plus how strict to
# be. `floor` is the fraction of the playlist that should satisfy it for the
# result to count as good.
CASES = [
    ("rustige piano muziek om bij in slaap te vallen",
     lambda t: has_any(t, ["sleep", "slaap", "calm", "rustig", "relax", "piano",
                           "ambient", "quiet", "night", "lullab", "meditat"]), 0.6),
    ("keiharde metal om me op te fokken",
     lambda t: has_any(t, ["metal", "hard", "heavy", "rock", "punk", "core",
                           "thrash", "rage", "brutal"]), 0.5),
    ("vrolijke zomerse muziek voor op het terras",
     lambda t: has_any(t, ["summer", "zomer", "sun", "zon", "beach", "strand",
                           "tropical", "happy", "vrolijk", "holiday", "vakantie"]), 0.5),
    ("nederlandse muziek om mee te zingen",
     looks_dutch, 0.7),
    ("nederlandse pop",
     looks_dutch, 0.7),
    ("instrumentale focus muziek om bij te werken",
     lambda t: has_any(t, ["instrumental", "focus", "concentrat", "study", "lofi",
                           "ambient", "piano", "calm", "background", "work"]), 0.5),
    ("jaren 80 hits",
     lambda t: in_years(t, 1980, 1989), 0.9),
    ("jaren 90 dance",
     lambda t: in_years(t, 1990, 1999), 0.9),
    ("romantische muziek voor een date",
     lambda t: has_any(t, ["love", "liefde", "romant", "heart", "hart", "kiss",
                           "together", "samen", "verliefd", "ballad"]), 0.5),
    ("hiphop",
     lambda t: has_any(t, ["hip", "rap", "trap", "urban", "r&b", "drill"]), 0.4),
    ("reggae",
     lambda t: has_any(t, ["reggae", "ska", "dub", "marley", "jamaic", "rasta"]), 0.4),
    ("klassieke muziek",
     lambda t: has_any(t, ["classic", "klassiek", "piano", "orchestr", "sonata",
                           "symphon", "concerto", "bach", "mozart", "chopin"]), 0.5),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verbose", action="store_true", help="show the misses")
    ap.add_argument("--fresh", action="store_true", help="skip the cache")
    args = ap.parse_args()

    tracks = load_library(fresh=args.fresh)
    emb, store = load_store(tracks)
    keys = [t.rating_key for t in tracks]
    print(f"{len(store)} vectors\n")

    total, passed = 0.0, 0
    for prompt, ok, floor in CASES:
        query = emb.embed_one(scoring.expand_query(prompt))
        sims = store.search(query, keys)
        scored = scoring.score_tracks(prompt, tracks, sims)
        low = prompt.lower()
        relaxed = bool(scoring.active_contexts(low) or scoring.extract_year(low))
        picked = scoring.select(scored, relaxed=relaxed)
        if not picked:
            print(f"  0%   {prompt!r} — empty")
            continue

        hits = [t for t in picked if ok(t)]
        frac = len(hits) / len(picked)
        total += frac
        good = frac >= floor
        passed += good
        mark = "✅" if good else "❌"
        print(f"  {mark} {frac:4.0%} (nodig {floor:.0%})  {prompt!r}")
        if args.verbose and not good:
            for t in [t for t in picked if not ok(t)][:4]:
                print(f"         mis: {t.real_artist} - {t.clean_title[:40]}")

    n = len(CASES)
    print(f"\n  gemiddeld {total / n:.0%} raak · {passed}/{n} prompts gehaald")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
