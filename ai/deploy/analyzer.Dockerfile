# Audio analysis, containerised.
#
# librosa leans on numba, which JIT-compiles through llvmlite. On your server's
# Python 3.13 that combination segfaults inside beat_track — reproducibly, and
# pinning numpy did not help. Python 3.11 is what numba is actually tested
# against, so the analysis runs in here rather than on the host interpreter.
#
# ⛔ The music mount MUST be :ro. This container only ever reads audio; its
#    output is the JSON cache, written to a separate writable path.
#
# Build:
#   docker build -f deploy/analyzer.Dockerfile -t jltamp-analyzer .
#
# Run (music read-only, cache writable):
#   docker run --rm \
#     -v /path/to/your/music/mp3:/music/mp3:ro \
#     -v /path/to/your/music/flac:/music/flac:ro \
#     -v $HOME/jltamp-ai:/out \
#     -e MUSIC_PATH_MAP=/music/mp3:/music/mp3,/music/flac:/music/flac \
#     -e AI_FEATURES_FILE=/out/track_features.json \
#     -e JLTAMP_URL=http://192.168.1.10:8090 \
#     -e JLTAMP_EMAIL=... -e JLTAMP_PASSWORD=... \
#     jltamp-analyzer --workers 8
FROM python:3.11-slim

# libsndfile for FLAC/WAV, ffmpeg for everything else audioread hands off.
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# numpy is pinned below 2.3: numba 0.6x builds its typing against the 2.x ABI
# and the newest numpy moves faster than numba follows.
RUN pip install --no-cache-dir "numpy<2.3" librosa requests

# One compute thread per worker process. Left at the default, each of N worker
# processes also starts a full OpenMP/BLAS thread pool — six workers became
# roughly forty-eight threads fighting over eight cores, and the pool died with
# a BrokenProcessPool. Measured: 1 worker completed 60/60, 6 workers completed 0.
ENV OMP_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 \
    # Workers compiling the same functions race over one cache directory.
    NUMBA_CACHE_DIR=/tmp/numba-cache

WORKDIR /app
COPY app/ ./app/
COPY scripts/analyze_audio.py ./scripts/

# Paths inside the container match what JLTamp reports, so MUSIC_PATH_MAP is
# an identity mapping and the script needs no container-specific knowledge.
ENTRYPOINT ["python", "scripts/analyze_audio.py"]
