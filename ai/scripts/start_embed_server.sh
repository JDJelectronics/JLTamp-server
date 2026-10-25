#!/bin/bash
# Start llama.cpp as an embedding server on :3100, then the AI service.
#
# Differences from the old ai_music launcher, which had two bugs that made
# startup unreliable:
#   * It listed every .gguf in the models dir, including 15-byte git-lfs
#     pointers, so picking one crashed the server.
#   * It started the Python engine itself AND was called by run_all.sh which
#     started it again — two processes fighting over port 5000.
# This starts each thing exactly once, and only offers models that are real.
set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
MODEL_DIR="${MODEL_DIR:-/home/USER/llama.cpp/models}"
BIN_SERVER="${BIN_SERVER:-/home/USER/llama.cpp/build/bin/llama-server}"
PORT="${EMBED_PORT:-3100}"
PREFERRED="${EMBED_MODEL:-bge-m3-Q8_0.gguf}"
LOG_FILE="$BASE_DIR/logs/embed_server.log"

mkdir -p "$BASE_DIR/logs"

cleanup() {
    echo ""
    echo "🛑 Shutting down..."
    [ -n "${SERVER_PID:-}" ] && kill "$SERVER_PID" 2>/dev/null || true
    exit 0
}
# Registered before anything long-running, so Ctrl-C works from the start.
trap cleanup SIGINT SIGTERM

echo "===================================="
echo "🎧 JLTamp AI"
echo "===================================="

if [ ! -x "$BIN_SERVER" ]; then
    echo "❌ llama-server not found at $BIN_SERVER"
    exit 1
fi

# Only real models: a GGUF is tens of MB at minimum, pointers are bytes.
mapfile -t MODELS < <(find "$MODEL_DIR" -name "*.gguf" -size +50M | sort)
if [ ${#MODELS[@]} -eq 0 ]; then
    echo "❌ No usable models in $MODEL_DIR"
    echo "   Run: ./scripts/fetch_model.sh"
    exit 1
fi

MODEL=""
for m in "${MODELS[@]}"; do
    [ "$(basename "$m")" = "$PREFERRED" ] && MODEL="$m" && break
done
if [ -z "$MODEL" ]; then
    # Any embedding model beats a chat model for this job.
    for m in "${MODELS[@]}"; do
        case "$(basename "$m" | tr '[:upper:]' '[:lower:]')" in
            *bge*|*embed*|*nomic*|*e5-*) MODEL="$m"; break ;;
        esac
    done
fi
if [ -z "$MODEL" ]; then
    MODEL="${MODELS[0]}"
    echo "⚠️  No embedding model found — falling back to $(basename "$MODEL")."
    echo "   Results will be noticeably worse. Run ./scripts/fetch_model.sh."
fi

# Jetson to maximum clocks; harmless (and skipped) elsewhere.
sudo nvpmodel -m 0 >/dev/null 2>&1 || true
sudo jetson_clocks >/dev/null 2>&1 || true

echo "🧹 Freeing port $PORT ..."
fuser -k "${PORT}/tcp" 2>/dev/null || true
sleep 1

echo "🔥 Starting: $(basename "$MODEL")"
"$BIN_SERVER" \
    -m "$MODEL" \
    --embedding \
    --pooling mean \
    --port "$PORT" \
    --host 127.0.0.1 \
    --n-gpu-layers 999 \
    --threads "$(nproc)" \
    --ctx-size 8192 \
    --batch-size 2048 \
    --ubatch-size 2048 \
    --flash-attn on \
    --log-verbosity 0 \
    > "$LOG_FILE" 2>&1 &
SERVER_PID=$!

# Poll for readiness instead of sleeping a fixed guess — a cold model load can
# take a minute, and a warm one is ready in seconds.
echo -n "⏳ Waiting for the model to load"
for _ in $(seq 1 60); do
    if ! kill -0 "$SERVER_PID" 2>/dev/null; then
        echo ""
        echo "❌ llama-server died during startup. Last lines:"
        tail -n 20 "$LOG_FILE"
        exit 1
    fi
    if curl -sf -m 2 "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
        echo " ready."
        break
    fi
    echo -n "."
    sleep 2
done

echo "✅ Embedding server up (PID $SERVER_PID)"
echo "------------------------------------"

cd "$BASE_DIR"
export PYTHONUNBUFFERED=1
python3 -m app.main

cleanup
