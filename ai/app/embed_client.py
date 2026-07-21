"""Client for the llama.cpp embedding server running on the Jetson's GPU.

llama.cpp has shipped several response shapes for /embedding across versions —
a bare object, a list of objects, and a list whose `embedding` is itself nested
one level deeper for multi-sequence input. `_vectors_from` accepts all of them
so a llama.cpp upgrade does not silently break embedding generation.
"""
from __future__ import annotations

import time

import numpy as np
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from . import config


class EmbedError(RuntimeError):
    pass


def _vectors_from(payload) -> list[list[float]]:
    """Flatten any of llama.cpp's response shapes into a list of vectors."""
    def one(obj):
        if isinstance(obj, dict):
            emb = obj.get("embedding", obj.get("data"))
        else:
            emb = obj
        if not isinstance(emb, list) or not emb:
            return None
        # Nested: [[...floats...]] — take the first (and only) sequence.
        if isinstance(emb[0], list):
            return [float(x) for x in emb[0]] if emb[0] else None
        if isinstance(emb[0], (int, float)):
            return [float(x) for x in emb]
        return None

    if isinstance(payload, dict):
        if isinstance(payload.get("data"), list):
            payload = payload["data"]
        else:
            v = one(payload)
            return [v] if v else []
    if not isinstance(payload, list):
        return []
    out = []
    for item in payload:
        v = one(item)
        if v:
            out.append(v)
    return out


class EmbedClient:
    def __init__(self, url: str | None = None):
        self.endpoint = (url or config.EMBED_URL).rstrip("/") + "/embedding"
        self.props_url = (url or config.EMBED_URL).rstrip("/") + "/props"
        self.session = requests.Session()
        # llama.cpp drops idle keep-alive connections, which surfaces as a bare
        # ProtocolError on the next request and killed a whole benchmark run
        # mid-way. Retrying a POST is safe here: embedding is a pure function,
        # so a repeat costs time and nothing else.
        retries = Retry(total=3, backoff_factor=0.5,
                        status_forcelist=[500, 502, 503, 504],
                        allowed_methods=["POST", "GET"])
        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        self.dim = 0
        self.model_id = ""

    # ── handshake ────────────────────────────────────────────────────────────
    def probe(self, attempts: int = 1, delay: float = 3.0) -> bool:
        """Confirm the server answers and learn its dimension + model name.

        The dimension is measured, not assumed: it is what decides whether the
        stored vectors are still valid.
        """
        for _ in range(max(1, attempts)):
            try:
                vecs = self.embed(["test"], timeout=20)
                if vecs and len(vecs[0]) > 8:
                    self.dim = len(vecs[0])
                    self.model_id = self._model_name()
                    return True
            except (requests.RequestException, EmbedError):
                pass
            time.sleep(delay)
        return False

    def _model_name(self) -> str:
        try:
            r = self.session.get(self.props_url, timeout=10)
            r.raise_for_status()
            props = r.json() or {}
        except (requests.RequestException, ValueError):
            return ""
        for key in ("model_path", "model", "default_generation_settings"):
            val = props.get(key)
            if isinstance(val, str) and val:
                return val.rsplit("/", 1)[-1]
            if isinstance(val, dict):
                inner = val.get("model")
                if isinstance(inner, str) and inner:
                    return inner.rsplit("/", 1)[-1]
        return ""

    # ── embedding ────────────────────────────────────────────────────────────
    def embed(self, texts: list[str], timeout: int | None = None) -> list[list[float]]:
        """Embed a batch in one request.

        Returns fewer vectors than texts only if the server does; callers pair
        results positionally, so a short response must not be silently padded.
        """
        if not texts:
            return []
        r = self.session.post(
            self.endpoint,
            json={"content": texts if len(texts) > 1 else texts[0]},
            timeout=timeout or config.EMBED_TIMEOUT,
        )
        if r.status_code != 200:
            raise EmbedError(f"embedding server returned {r.status_code}")
        vecs = _vectors_from(r.json())
        if self.dim and any(len(v) != self.dim for v in vecs):
            raise EmbedError("embedding server changed dimension mid-run")
        return vecs

    def embed_one(self, text: str) -> np.ndarray:
        vecs = self.embed([text])
        if not vecs:
            raise EmbedError("no vector returned")
        return np.asarray(vecs[0], dtype=np.float32)

    def embed_batched(self, texts: list[str], batch: int | None = None):
        """Yield (offset, vectors) per chunk so callers can persist as they go.

        A failed chunk yields nothing for that range rather than aborting the
        run — one bad title should not stop a library-wide build.
        """
        size = batch or config.EMBED_BATCH
        for i in range(0, len(texts), size):
            chunk = texts[i:i + size]
            try:
                vecs = self.embed(chunk)
            except (requests.RequestException, EmbedError) as e:
                print(f"⚠️  embed batch at {i} failed: {e}")
                continue
            if len(vecs) != len(chunk):
                # Positional pairing is no longer safe for this chunk.
                print(f"⚠️  embed batch at {i}: got {len(vecs)} of {len(chunk)}")
                continue
            yield i, vecs
