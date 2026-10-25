#!/usr/bin/env python3
"""
CLAP audio-embeddings voor JLTamp — échte "klinkt zoals dit".

Embedt elk nummer met een CLAP-model (LAION, gedeelde audio<->tekst-ruimte) naar
een 512-dim vector. Daarmee kan de engine op TIMBRE/klank matchen (i.p.v. alleen
op tags/tekst): een tekstvraag wordt via de CLAP-tekst-encoder in dezelfde ruimte
gezet, en "radio vanaf dit nummer" wordt echte audio-gelijkenis.

Zelfde patroon als analyze_audio.py: lokale paden (mount), resumable, sharded.
Output: <out>/clap_vectors.npy (float32, L2-genormaliseerd) + <out>/clap_index.json
(rating_key -> rij). Beide worden incrementeel weggeschreven.

Draai in de container (deploy/clap.Dockerfile) met de muziek read-only gemount.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app.jltamp_client import JLTampClient  # noqa: E402
from app import config  # noqa: E402

OUT_DIR = Path(os.environ.get("CLAP_OUT", "/out"))
VEC_FILE = OUT_DIR / "clap_vectors.npy"
IDX_FILE = OUT_DIR / "clap_index.json"
MODEL_ID = os.environ.get("CLAP_MODEL", "laion/clap-htsat-unfused")
SAMPLE_SECONDS = int(os.environ.get("CLAP_SECONDS", "10"))
SR = 48000  # CLAP verwacht 48 kHz
FLUSH_EVERY = int(os.environ.get("CLAP_FLUSH", "50"))


def path_map() -> list[tuple[str, str]]:
    pairs = []
    for entry in os.environ.get("MUSIC_PATH_MAP", "").split(","):
        if ":" in entry:
            src, _, dst = entry.strip().partition(":")
            pairs.append((src.strip(), dst.strip()))
    return pairs


def local_path(remote: str, mapping) -> str | None:
    for src, dst in mapping:
        if remote and remote.startswith(src):
            cand = dst + remote[len(src):]
            if os.path.exists(cand):
                return cand
    return remote if remote and os.path.exists(remote) else None


def _load_index() -> dict:
    if IDX_FILE.exists():
        try:
            return json.loads(IDX_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", default="", metavar="I/N")
    args = ap.parse_args()
    shard_i, shard_n = 0, 1
    if args.shard:
        shard_i, shard_n = (int(x) for x in args.shard.split("/", 1))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    index = _load_index()                       # rating_key -> rij
    vectors = (list(np.load(VEC_FILE)) if VEC_FILE.exists() and index else [])
    print(f"📂 {len(index)} tracks al ge-embed → {VEC_FILE}", flush=True)

    # Model pas laden als er werk is (scheelt bij een lege run).
    client = JLTampClient()
    client.login()
    tracks = client.fetch_tracks()
    print(f"🎵 {len(tracks)} tracks in JLTamp.", flush=True)

    mapping = path_map()
    todo = [t for t in tracks if t.rating_key not in index]
    if shard_n > 1:
        todo = todo[shard_i::shard_n]
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("✅ Niets te doen.", flush=True)
        return 0

    print(f"⏳ Model laden ({MODEL_ID}) …", flush=True)
    import torch
    from transformers import ClapAudioModelWithProjection, ClapProcessor
    import librosa
    torch.set_num_threads(int(os.environ.get("TORCH_THREADS", "4")))
    model = ClapAudioModelWithProjection.from_pretrained(MODEL_ID)
    processor = ClapProcessor.from_pretrained(MODEL_ID)
    model.eval()
    print(f"🚀 {len(todo)} tracks embedden.", flush=True)

    def flush():
        arr = np.asarray(vectors, dtype=np.float32)
        np.save(VEC_FILE, arr)
        IDX_FILE.write_text(json.dumps(index))

    done = 0
    for t in todo:
        path = local_path(t.path, mapping)
        if not path:
            continue
        try:
            y, _ = librosa.load(path, sr=SR, mono=True, duration=SAMPLE_SECONDS)
            if y.size == 0:
                continue
            inputs = processor(audio=y, sampling_rate=SR, return_tensors="pt")
            with torch.no_grad():
                emb = model(**inputs).audio_embeds[0].numpy()
            emb = emb / (np.linalg.norm(emb) or 1.0)   # L2-normaliseren
            index[t.rating_key] = len(vectors)
            vectors.append(emb.astype(np.float32))
            done += 1
            if done % FLUSH_EVERY == 0:
                flush()
                print(f"   {done}/{len(todo)} — {len(index)} totaal", flush=True)
        except Exception as e:                          # noqa: BLE001
            print(f"⚠️  {t.rating_key}: {str(e)[:100]}", flush=True)

    flush()
    print(f"✅ {done} ge-embed, {len(index)} totaal.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
