#!/bin/bash
# Download the embedding model.
#
# bge-m3 is a purpose-built embedding model: multilingual (this library is Dutch
# and English), 1024 dimensions instead of a chat model's 4096, and trained for
# exactly this — putting semantically similar text near each other. A generative
# model like Llama-3 produces vectors as a side effect and ranks noticeably
# worse, while costing four times the storage.
set -euo pipefail

MODEL_DIR="${MODEL_DIR:-/home/USER/llama.cpp/models}"
MODEL_FILE="bge-m3-Q8_0.gguf"
URL="https://huggingface.co/gpustack/bge-m3-GGUF/resolve/main/bge-m3-Q8_0.gguf"
TARGET="$MODEL_DIR/$MODEL_FILE"

mkdir -p "$MODEL_DIR"

if [ -f "$TARGET" ]; then
    size=$(stat -c%s "$TARGET")
    # Anything tiny is a git-lfs pointer, not a model — several of those are
    # already sitting in the models dir and they crash llama-server on load.
    if [ "$size" -gt 100000000 ]; then
        echo "✅ Already present: $TARGET ($(numfmt --to=iec "$size"))"
        exit 0
    fi
    echo "⚠️  $TARGET is only $size bytes — a broken/pointer file. Re-downloading."
    rm -f "$TARGET"
fi

echo "⬇️  Downloading $MODEL_FILE (~610 MB) ..."
# Write to .part so an interrupted download is never mistaken for a real model.
curl -L --fail --progress-bar -o "$TARGET.part" "$URL"
mv "$TARGET.part" "$TARGET"

echo "✅ Done: $TARGET ($(numfmt --to=iec "$(stat -c%s "$TARGET")"))"
