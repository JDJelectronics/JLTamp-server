#!/usr/bin/env python3
"""Measure BPM, energy and brightness per track, for context-aware prompts.

Text embeddings know that "Eye of the Tiger" is a rock song; they do not know
it sits at 109 BPM. Prompts like "gym" or "slapen" are really about tempo and
energy, so the scorer reads these measurements (see scoring.AUDIO_TARGETS).

⛔ READ-ONLY. This opens audio files and never writes to them. Its only output
is the JSON cache, which lives outside the music directory. Treat the music
library as read-only — it is the user's irreplaceable collection.

Two ways to reach the audio, in order of preference:

  1. Local files. Fastest by far (~0.6 s/track). JLTamp reports container paths
     (/music/mp3/...), so map them to what this machine can see:
         MUSIC_PATH_MAP=/music/mp3:/path/on/this/host/mp3,/music/flac:/path/on/this/host/flac
     Run this on the server that has the mounts.

  2. Streaming over the API. Works anywhere, but downloading ~9 MB per track
     dominates the cost (~2.9 s/track). Enable with AUDIO_ALLOW_STREAM=1.

Concurrency is capped hard: measured on your server, 2 worker processes finish
every track and 3 kill the pool outright (numba inside a process pool). Scale
out with several containers instead, each taking a shard — separate processes
cannot take each other down:

    python3 scripts/analyze_audio.py --shard 0/3 &
    python3 scripts/analyze_audio.py --shard 1/3 &
    python3 scripts/analyze_audio.py --shard 2/3 &
    python3 scripts/merge_features.py      # afterwards

Progress is saved continuously, so stopping and resuming is free.

Usage:
    python3 scripts/analyze_audio.py [--workers N] [--limit N] [--shard I/N]
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import config                      # noqa: E402
from app.jltamp_client import JLTampClient  # noqa: E402

OUT_FILE = Path(os.environ.get("AI_FEATURES_FILE") or config.FEATURES_FILE)
SAMPLE_SECONDS = 60       # a minute establishes tempo and energy well enough
# Flush often: the file is small (a few hundred KB), while a crash or a stop
# loses everything measured since the last write — and with several shards
# running, a coarse interval also hides progress for minutes at a time.
SAVE_EVERY = 25
# A single track that kills the pool gets this many fresh attempts before we
# accept it is genuinely unreadable.
SINGLE_RETRIES = 3
ALLOW_STREAM = os.environ.get("AUDIO_ALLOW_STREAM", "").lower() in ("1", "true", "yes")


def path_map() -> list[tuple[str, str]]:
    pairs = []
    for entry in os.environ.get("MUSIC_PATH_MAP", "").split(","):
        entry = entry.strip()
        if ":" in entry:
            src, _, dst = entry.partition(":")
            pairs.append((src.strip(), dst.strip()))
    return pairs


def local_path(remote: str, mapping: list[tuple[str, str]]) -> str | None:
    for src, dst in mapping:
        if remote.startswith(src):
            candidate = dst + remote[len(src):]
            if os.path.exists(candidate):
                return candidate
    return remote if remote and os.path.exists(remote) else None


def _measure(path: str) -> dict | None:
    import librosa
    import numpy as np

    y, sr = librosa.load(path, duration=SAMPLE_SECONDS, mono=True)
    if y.size == 0:
        return None
    tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
    # librosa <0.11 returns a scalar here, 0.11+ a 1-element array. float() on
    # the array raises, so normalise before converting.
    bpm = float(np.atleast_1d(tempo)[0])
    energy = float(np.mean(librosa.feature.rms(y=y)))
    brightness = float(np.mean(librosa.feature.spectral_centroid(y=y, sr=sr)))
    # No "danceability" here. Deriving it needs librosa.effects.percussive, a
    # full harmonic/percussive separation that profiling put at 86% of the
    # per-track cost — 3.8 s of the 4.4 s — and nothing reads the result:
    # scoring.AUDIO_TARGETS matches on tempo and energy only. Dropping it takes
    # a track from ~4.4 s to ~0.6 s. Add it back with the scorer that uses it.
    return {
        "bpm": int(round(bpm)),
        "energy": round(energy, 4),
        "brightness": round(brightness, 2),
    }


def analyse_one(job: tuple[str, str, str, str]) -> tuple[str, dict | None, str]:
    """Runs in a worker process. Returns (rating_key, features, error)."""
    rating_key, path, stream_url, token = job
    tmp = None
    try:
        if path is None:
            if not stream_url:
                return rating_key, None, "not readable here"
            import requests
            import tempfile
            r = requests.get(stream_url, headers={"X-Plex-Token": token}, timeout=120)
            r.raise_for_status()
            with tempfile.NamedTemporaryFile(suffix=".audio", delete=False) as f:
                f.write(r.content)
                tmp = path = f.name
        return rating_key, _measure(path), ""
    except Exception as e:                   # noqa: BLE001 — one bad file must not stop the run
        return rating_key, None, str(e)[:120]
    finally:
        if tmp:
            try:
                os.unlink(tmp)
            except OSError:
                pass


def main() -> int:
    ap = argparse.ArgumentParser()
    # Two, not more: measured on your server, 2 workers completes every track and
    # 3 kills the pool outright. Scale out with several containers instead —
    # separate processes cannot take each other down.
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0, help="stop after N tracks")
    ap.add_argument("--shard", default="", metavar="I/N",
                    help="handle only shard I of N (e.g. 0/3), for running "
                         "several instances side by side")
    args = ap.parse_args()

    shard_i, shard_n = 0, 1
    if args.shard:
        try:
            shard_i, shard_n = (int(x) for x in args.shard.split("/", 1))
        except ValueError:
            print(f"❌ --shard must look like 0/3, got {args.shard!r}")
            return 2
        if not (0 <= shard_i < shard_n):
            print(f"❌ --shard {args.shard}: I must be between 0 and N-1")
            return 2

    # Each shard owns its own file. They all write the whole JSON on every
    # flush, so sharing one path would have them overwrite each other's work.
    # scripts/merge_features.py combines them afterwards.
    out_file = OUT_FILE
    if shard_n > 1:
        out_file = OUT_FILE.with_suffix(f".shard{shard_i}of{shard_n}.json")

    cache: dict = {}
    # Read every shard's results so a restart never re-measures what a sibling
    # already did, but only ever write our own.
    for existing in [OUT_FILE, *sorted(OUT_FILE.parent.glob(f"{OUT_FILE.stem}.shard*.json"))]:
        if not existing.exists():
            continue
        try:
            cache.update(json.loads(existing.read_text()))
        except (json.JSONDecodeError, OSError):
            continue
    # Entries from a previous run that produced nothing are worth retrying:
    # the old Plex-era cache is 31k tracks of empty dicts.
    cache = {k: v for k, v in cache.items() if isinstance(v, dict) and v.get("bpm")}
    print(f"📂 {len(cache)} tracks already measured → {out_file}")

    client = JLTampClient()
    client.login()
    tracks = client.fetch_tracks()
    print(f"🎵 {len(tracks)} tracks in JLTamp.")

    mapping = path_map()
    todo = [t for t in tracks if t.rating_key not in cache]
    if shard_n > 1:
        # Interleave rather than slice into blocks: an album's tracks sit next
        # to each other, so contiguous blocks would give one shard all the long
        # FLACs and another all the short pop songs.
        todo = todo[shard_i::shard_n]
        print(f"🔀 Shard {shard_i}/{shard_n}: {len(todo)} tracks of this run.")
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("✅ Nothing to do.")
        return 0

    jobs, unreachable = [], 0
    for t in todo:
        local = local_path(t.path, mapping)
        if local is None:
            unreachable += 1
            if not ALLOW_STREAM:
                continue
            stream = f"{client.url}/library/parts/{t.rating_key}/file"
            jobs.append((t.rating_key, None, stream, client._token))
        else:
            jobs.append((t.rating_key, local, "", ""))

    if unreachable:
        where = "streaming over the API" if ALLOW_STREAM else "SKIPPED"
        print(f"⚠️  {unreachable} tracks are not readable on this machine → {where}")
        if not ALLOW_STREAM and not jobs:
            print("   Set MUSIC_PATH_MAP to point at the music, or "
                  "AUDIO_ALLOW_STREAM=1 to fetch over the network.")
            return 1

    print(f"🚀 Measuring {len(jobs)} tracks across {args.workers} workers.")
    print("   Ctrl-C is safe — progress is saved as it goes.\n")

    # Seed from this shard's own file, not empty: flush() writes `mine` over
    # that file, so starting blank would erase everything the shard measured
    # in an earlier run. (It did — a restart dropped the total from 432 to 196.)
    mine: dict = {}
    if out_file.exists():
        try:
            mine = {k: v for k, v in json.loads(out_file.read_text()).items()
                    if isinstance(v, dict) and v.get("bpm")}
        except (json.JSONDecodeError, OSError):
            mine = {}

    def flush() -> None:
        tmp = out_file.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(mine))
        os.replace(tmp, out_file)

    out_file.parent.mkdir(parents=True, exist_ok=True)
    done = failed = 0
    started = time.time()
    errors: dict[str, int] = {}

    # "spawn", not the Linux default "fork": numba's JIT state does not survive
    # being forked, and the worker dies with a BrokenProcessPool the moment it
    # touches beat_track. Spawned workers start clean and each pay a one-off
    # ~80 s JIT warm-up, which a long-lived pool amortises away.
    ctx = multiprocessing.get_context("spawn")

    # Submitting all 69k jobs at once fills the executor's queues and makes a
    # single dying worker take the whole run with it (BrokenProcessPool). Keep
    # only a small window in flight, and rebuild the pool if it does break so
    # one bad file costs a batch rather than the entire job.
    window = max(8, args.workers * 4)
    pending = list(jobs)
    processed = 0

    def report() -> None:
        rate = processed / max(1e-6, time.time() - started)
        left = (len(jobs) - processed) / rate if rate else 0
        print(f"   {processed}/{len(jobs)} — {done} ok, {failed} failed, "
              f"{rate:.2f}/s, ~{left / 3600:.1f} h left", flush=True)

    def run_batch(batch: list) -> None:
        """Measure a batch, isolating any file that kills the worker.

        Some MP3s crash libmpg123 inside the decoder — a native abort, not a
        Python exception, so the worker process dies and takes the pool with
        it. Skipping the whole batch cost 32 tracks per bad file and lost two
        shards' entire share overnight. Halving down to the single offender
        costs a handful of pool restarts instead.
        """
        nonlocal processed, done, failed
        try:
            with ProcessPoolExecutor(max_workers=args.workers,
                                     mp_context=ctx) as pool:
                futures = {pool.submit(analyse_one, j): j[0] for j in batch}
                for fut in as_completed(futures):
                    rating_key, features, err = fut.result()
                    processed += 1
                    if features:
                        cache[rating_key] = features
                        mine[rating_key] = features
                        done += 1
                    else:
                        failed += 1
                        key = (err.split(":")[0][:40] or "unknown")
                        errors[key] = errors.get(key, 0) + 1
                    if done and done % SAVE_EVERY == 0:
                        flush()
                        report()
            return
        except BrokenProcessPool:
            pass

        if len(batch) == 1:
            # Do NOT conclude the file is bad. The pool dies intermittently for
            # reasons unrelated to its contents, and bisecting simply blames
            # whichever track happened to be running: an earlier version marked
            # 8,448 files "unreadable", including ABBA's SOS, which measures
            # fine on its own. Retry in a fresh pool before giving up.
            for _ in range(SINGLE_RETRIES):
                try:
                    with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as pool:
                        rating_key, features, err = pool.submit(
                            analyse_one, batch[0]).result()
                    processed += 1
                    if features:
                        cache[rating_key] = features
                        mine[rating_key] = features
                        done += 1
                    else:
                        failed += 1
                        errors[err.split(":")[0][:40] or "unknown"] = errors.get(
                            err.split(":")[0][:40] or "unknown", 0) + 1
                    return
                except BrokenProcessPool:
                    continue
            processed += 1
            failed += 1
            errors["crashed the worker repeatedly"] = (
                errors.get("crashed the worker repeatedly", 0) + 1)
            print(f"   ⚠️  gave up after {SINGLE_RETRIES} retries: {batch[0][1]}",
                  flush=True)
            flush()
            return

        mid = len(batch) // 2
        run_batch(batch[:mid])
        run_batch(batch[mid:])

    try:
        while pending:
            batch, pending = pending[:window * 4], pending[window * 4:]
            run_batch(batch)
    except KeyboardInterrupt:
        print("\n⏸️  Interrupted — saving what we have.")
    finally:
        flush()

    print(f"\n✅ {done} measured, {failed} failed. This shard holds {len(mine)}; known overall {len(cache)}.")
    if errors:
        print("   Most common failures:")
        for msg, n in sorted(errors.items(), key=lambda kv: -kv[1])[:5]:
            print(f"     {n:>6}×  {msg}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
