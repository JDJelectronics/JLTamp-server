# CLAP audio-embeddings, containerised (CPU). Model wordt bij de eerste run naar
# /models gedownload (mount dat als volume zodat het blijft staan).
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*
RUN pip install --no-cache-dir "numpy<2.3" \
        torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir transformers librosa soundfile requests
ENV OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 MKL_NUM_THREADS=4 \
    TORCH_THREADS=4 HF_HOME=/models NUMBA_CACHE_DIR=/tmp/numba-cache \
    MUSIC_PATH_MAP=/music/mp3:/music/mp3,/music/flac:/music/flac \
    CLAP_OUT=/out
WORKDIR /app
COPY app/ ./app/
COPY scripts/clap_embed.py ./scripts/
ENTRYPOINT ["python", "scripts/clap_embed.py"]
