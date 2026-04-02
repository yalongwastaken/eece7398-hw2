#!/usr/bin/env bash
# @file    download_models.sh
# @author  Anthony Yalong
# @nuid    002156860
# @brief   Download all required model weights into models/.
#          Whisper and Kokoro weights are downloaded automatically on first run.
#          Only the Qwen2.5-7B GGUF requires an explicit download via HuggingFace CLI.
# @usage   bash scripts/download_models.sh

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HF_CLI="$ROOT/.venv/bin/hf"

if [ ! -f "$HF_CLI" ]; then
    echo "[models] error: .venv not found. run 'pip install -r requirements.txt' first."
    exit 1
fi

echo "[models] downloading Qwen2.5-7B-Instruct Q4_K_M..."
"$HF_CLI" download bartowski/Qwen2.5-7B-Instruct-GGUF \
    Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --local-dir "$ROOT/models/"

echo "[models] done."