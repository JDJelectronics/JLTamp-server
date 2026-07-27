# JLTamp Music Server — self-hosted, music-only, Plex-compatible API.
FROM python:3.12-slim

# ffmpeg = on-the-fly transcoding (Hi-Res FLAC/ALAC/WAV → mp3 for cast / low
# bandwidth). libmagic etc. aren't needed — mutagen + Pillow are pure-python
# wheels. Keep the image slim.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app

# Guard: a name Python can't resolve (missing import, typo) is NOT an import
# error. The module loads, the container reports healthy and /healthz is green —
# the NameError only fires on the first request that reaches that line, so a
# single endpoint 500s in production while everything looks fine. pyflakes finds
# undefined names statically, so the build fails instead of the server. Only
# "undefined name" blocks; style and unused-import warnings do not.
RUN pip install --no-cache-dir pyflakes==3.4.0 \
    && { python -m pyflakes app || true; } | grep "undefined name" > /tmp/undefined.txt; \
    if [ -s /tmp/undefined.txt ]; then \
        echo "BUILD STOPPED — undefined name(s); this would be a 500 at runtime:"; \
        sed "s/^/  /" /tmp/undefined.txt; \
        exit 1; \
    fi; \
    rm -f /tmp/undefined.txt && pip uninstall -y -q pyflakes

# The web app (Expo web export) — served at "/" so opening the server in a
# browser gives the full JLTamp UI out of the box, talking to this same origin.
COPY web ./web

# Music is mounted read-only at /music; the server writes only to /data.
ENV MUSIC_DIR=/music \
    DATA_DIR=/data \
    WEB_DIR=/app/web \
    PORT=32400
VOLUME ["/data"]
EXPOSE 32400

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT}"]
