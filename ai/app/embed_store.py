"""Embedding storage.

The old engine kept every vector as a Python list inside one big dict and
rewrote the whole thing as JSON after each batch. At 4096 dimensions that grew
into a 4.1 GB file that took minutes to write, held gigabytes of float objects
in RAM, and left a half-written .tmp behind whenever it was interrupted.

Here vectors live in a memory-mapped float32 array. Adding rows touches only
those rows, the OS pages in what the search actually reads, and a crash can
lose at most the index update — never the vectors already on disk.

Vectors are L2-normalised on the way in, so cosine similarity is a plain dot
product and the whole search becomes one matrix multiply.
"""
from __future__ import annotations

import hashlib
import json
import os
import threading
from pathlib import Path

import numpy as np


def text_hash(text: str) -> str:
    """Short digest of the text a vector was built from. Not security-relevant;
    it only needs to change when the text does."""
    return hashlib.blake2b(text.encode("utf-8"), digest_size=8).hexdigest()

INDEX_NAME = "index.json"
VECTORS_NAME = "vectors.npy"
INITIAL_CAPACITY = 4096


def normalise(vec: np.ndarray) -> np.ndarray:
    """Unit-length, so dot(a, b) == cosine(a, b). A zero vector stays zero
    rather than becoming NaN."""
    v = np.asarray(vec, dtype=np.float32).ravel()
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


class DimensionMismatch(RuntimeError):
    """The store holds vectors from a different model than the one now running.

    This is the failure the old engine could not see: switching the model in
    the launcher menu silently left every stored vector in a different space,
    so search kept working and kept returning nonsense.
    """


class EmbeddingStore:
    def __init__(self, data_dir: Path, dim: int, model_id: str = ""):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.dim = int(dim)
        self.model_id = model_id
        self.index_path = self.dir / INDEX_NAME
        self.vectors_path = self.dir / VECTORS_NAME
        self._lock = threading.RLock()
        self._keys: dict[str, int] = {}
        # What text each vector was built from, so a track whose metadata
        # changed gets re-embedded instead of keeping a stale vector. Without
        # this, improving how we phrase the input means rebuilding all of it.
        self._hashes: dict[str, str] = {}
        self._order: list[str] = []
        self._vectors: np.ndarray | None = None
        self._dirty = False
        self._load()

    # ── persistence ──────────────────────────────────────────────────────────
    def _load(self) -> None:
        meta = {}
        if self.index_path.exists():
            try:
                meta = json.loads(self.index_path.read_text())
            except (json.JSONDecodeError, OSError):
                # A corrupt index is recoverable: the vectors are intact but we
                # no longer know which row is which, so we start over rather
                # than hand back mislabelled vectors.
                meta = {}

        stored_dim = int(meta.get("dim") or 0)
        stored_model = meta.get("model") or ""
        keys = meta.get("keys") or {}

        if keys and stored_dim and stored_dim != self.dim:
            raise DimensionMismatch(
                f"{self.index_path} holds {stored_dim}-dimensional vectors from "
                f"model '{stored_model or 'unknown'}', but the running model "
                f"produces {self.dim}. Delete {self.dir} to rebuild, or point "
                f"AI_DATA_DIR somewhere else."
            )
        if keys and stored_model and self.model_id and stored_model != self.model_id:
            raise DimensionMismatch(
                f"{self.index_path} was built with model '{stored_model}' but "
                f"'{self.model_id}' is running. Vectors from different models "
                f"are not comparable. Delete {self.dir} to rebuild."
            )

        self._keys = {str(k): int(v) for k, v in keys.items()}
        self._hashes = {str(k): str(v) for k, v in (meta.get("hashes") or {}).items()}
        self._order = [""] * len(self._keys)
        for k, row in self._keys.items():
            if 0 <= row < len(self._order):
                self._order[row] = k

        if self.vectors_path.exists() and self._keys:
            arr = np.load(self.vectors_path, mmap_mode="r+")
            if arr.shape[1] != self.dim:
                raise DimensionMismatch(
                    f"{self.vectors_path} has width {arr.shape[1]}, expected {self.dim}."
                )
            self._vectors = arr
        else:
            self._vectors = None

    def _ensure_capacity(self, needed: int) -> None:
        """Grow the backing file by doubling. Reallocation copies, so doubling
        keeps that cost amortised to O(1) per row."""
        current = 0 if self._vectors is None else self._vectors.shape[0]
        if needed <= current:
            return
        capacity = max(INITIAL_CAPACITY, current * 2 or INITIAL_CAPACITY)
        while capacity < needed:
            capacity *= 2

        tmp = self.vectors_path.with_suffix(".npy.resize")
        new = np.lib.format.open_memmap(
            tmp, mode="w+", dtype=np.float32, shape=(capacity, self.dim)
        )
        if self._vectors is not None and current:
            new[:current] = self._vectors[:current]
        new.flush()
        # Drop our handle before replacing the file underneath it.
        self._vectors = None
        del new
        os.replace(tmp, self.vectors_path)
        self._vectors = np.load(self.vectors_path, mmap_mode="r+")

    def save(self) -> None:
        """Persist the index. Vectors are already on disk via the memmap."""
        with self._lock:
            if self._vectors is not None:
                self._vectors.flush()
            if not self._dirty:
                return
            payload = {
                "dim": self.dim,
                "model": self.model_id,
                "count": len(self._keys),
                "keys": self._keys,
                "hashes": self._hashes,
            }
            tmp = self.index_path.with_suffix(".json.tmp")
            with open(tmp, "w") as f:
                json.dump(payload, f)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.index_path)
            self._dirty = False

    # ── access ───────────────────────────────────────────────────────────────
    def __len__(self) -> int:
        return len(self._keys)

    def has(self, key: str) -> bool:
        return key in self._keys

    def missing(self, keys: list[str]) -> list[str]:
        with self._lock:
            return [k for k in keys if k not in self._keys]

    def stale(self, items: dict[str, str]) -> list[str]:
        """Keys from {key: text} that have no vector, or one built from
        different text."""
        with self._lock:
            return [k for k, text in items.items()
                    if k not in self._keys or self._hashes.get(k) != text_hash(text)]

    def add_many(self, items: dict[str, np.ndarray],
                 texts: dict[str, str] | None = None) -> int:
        """Append or overwrite vectors. Returns how many rows were written."""
        if not items:
            return 0
        texts = texts or {}
        with self._lock:
            new_keys = [k for k in items if k not in self._keys]
            self._ensure_capacity(len(self._keys) + len(new_keys))
            for key, vec in items.items():
                v = normalise(vec)
                if v.shape[0] != self.dim:
                    continue
                row = self._keys.get(key)
                if row is None:
                    row = len(self._order)
                    self._keys[key] = row
                    self._order.append(key)
                self._vectors[row] = v
                if key in texts:
                    self._hashes[key] = text_hash(texts[key])
            self._dirty = True
            return len(items)

    def matrix(self, keys: list[str]) -> tuple[np.ndarray, list[str]]:
        """Stacked vectors for `keys` that we actually have, plus those keys in
        matching order. One contiguous array so scoring is a single matmul."""
        with self._lock:
            rows, present = [], []
            for k in keys:
                row = self._keys.get(k)
                if row is not None:
                    rows.append(row)
                    present.append(k)
            if not rows:
                return np.zeros((0, self.dim), dtype=np.float32), []
            return np.asarray(self._vectors[rows], dtype=np.float32), present

    def search(self, query: np.ndarray, keys: list[str]) -> dict[str, float]:
        """Cosine similarity of `query` against `keys`, as {key: score}."""
        mat, present = self.matrix(keys)
        if not present:
            return {}
        scores = mat @ normalise(query)
        return dict(zip(present, scores.tolist()))
