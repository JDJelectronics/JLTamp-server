#!/usr/bin/env python3
"""Combine the per-shard audio-feature files into one.

Each analyser shard writes its own JSON so parallel runs cannot overwrite one
another. This folds them back into the single file the engine reads, and
reports what is still missing.

    python3 scripts/merge_features.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config  # noqa: E402

OUT_FILE = Path(os.environ.get("AI_FEATURES_FILE") or config.FEATURES_FILE)


def main() -> int:
    shards = sorted(OUT_FILE.parent.glob(f"{OUT_FILE.stem}.shard*.json"))
    if not shards:
        print(f"Geen shard-bestanden naast {OUT_FILE}.")
        return 1

    merged: dict = {}
    if OUT_FILE.exists():
        try:
            merged = json.loads(OUT_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            merged = {}

    for path in shards:
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ⚠️  {path.name} overgeslagen: {e}")
            continue
        usable = {k: v for k, v in data.items()
                  if isinstance(v, dict) and v.get("bpm")}
        merged.update(usable)
        print(f"  {path.name}: {len(usable)} metingen")

    tmp = OUT_FILE.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(merged))
    os.replace(tmp, OUT_FILE)
    print(f"\n✅ {len(merged)} tracks samengevoegd → {OUT_FILE}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
