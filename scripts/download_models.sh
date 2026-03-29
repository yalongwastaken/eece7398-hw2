#!/usr/bin/env bash
# @file    download_models.sh
# @brief   download all required model weights into models/
# @usage   bash scripts/download_models.sh

set -e

PYTHON="$(cd "$(dirname "$0")/.." && pwd)/.venv/bin/python"
HF_CLI="$(cd "$(dirname "$0")/.." && pwd)/.venv/bin/hf"

echo "[models] downloading Qwen2.5-7B-Instruct Q4_K_M..."
"$HF_CLI" download bartowski/Qwen2.5-7B-Instruct-GGUF \
    Qwen2.5-7B-Instruct-Q4_K_M.gguf \
    --local-dir models/

echo "[models] all models downloaded."