"""Infer a genre for tracks that have none, from their embedding neighbours.

More than half this library carries no usable genre — the tag is blank, or a
placeholder like "music", "other" or "hi-res". Those tracks cannot match a
genre prompt at all, which quietly excludes them from half of what the AI is
asked for.

A track's nearest neighbours in embedding space are, in practice, the same
kind of music. So an untagged track is labelled by a similarity-weighted vote
of its K tagged neighbours, and only when that vote is confident enough. The
result is written to a SEPARATE field: it never overwrites a real tag, so a
guess can be revised and the user's own metadata is left untouched.
"""
from __future__ import annotations

import json
import os
from collections import defaultdict
from pathlib import Path

import numpy as np

# Tags that name no genre. A track carrying only one of these is "untagged"
# for our purposes and is a candidate to be labelled.
PLACEHOLDER_GENRES = {
    "", "music", "muziek", "other", "diverse", "hi-res", "hires", "audio",
    "verzamel", "verzamelaar", "unknown", "onbekend",
}

K_NEIGHBOURS = 15


def is_placeholder(genre: str) -> bool:
    return genre.strip().lower() in PLACEHOLDER_GENRES


def infer_with_confidence(tracks, store,
                          k: int = K_NEIGHBOURS) -> dict[str, tuple[str, float]]:
    """Return {rating_key: (best_genre, confidence)} for every untagged track.

    Confidence is the winning genre's share of the similarity-weighted vote.
    Callers apply their own threshold; computing this once and thresholding
    afterwards avoids repeating the heavy neighbour search per threshold.
    """
    tagged, untagged = [], []
    for t in tracks:
        (untagged if is_placeholder(t.genre) else tagged).append(t)
    if not tagged or not untagged:
        return {}

    tag_mat, tag_keys = store.matrix([t.rating_key for t in tagged])
    by_key = {t.rating_key: t for t in tagged}
    genres = np.array([by_key[k_].genre.strip().lower() for k_ in tag_keys])

    un_mat, un_keys = store.matrix([t.rating_key for t in untagged])
    out: dict[str, tuple[str, float]] = {}

    # Batched so the untagged×tagged similarity matrix never materialises whole.
    for i in range(0, len(un_keys), 512):
        chunk = un_mat[i:i + 512]
        sims = chunk @ tag_mat.T
        idx = np.argpartition(-sims, k, axis=1)[:, :k]
        for r in range(chunk.shape[0]):
            weight: dict[str, float] = defaultdict(float)
            for j in idx[r]:
                weight[genres[j]] += float(sims[r, j])
            total = sum(weight.values())
            if total <= 0:
                continue
            top = max(weight, key=weight.get)
            out[un_keys[i + r]] = (top, weight[top] / total)
    return out


def infer(tracks, store, threshold: float = 0.6,
          k: int = K_NEIGHBOURS) -> dict[str, str]:
    """Inferred genre per untagged track whose vote clears `threshold`."""
    return {key: g for key, (g, conf) in infer_with_confidence(tracks, store, k).items()
            if conf >= threshold}


def save(inferred: dict[str, str], path: Path) -> None:
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(inferred))
    os.replace(tmp, path)


def load(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
