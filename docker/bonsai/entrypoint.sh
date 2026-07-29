#!/usr/bin/env bash
set -euo pipefail

MODELS_DIR="${MODELS_DIR:-/models}"
GGUF="${BONSAI_GGUF:-Bonsai-27B-Q1_0.gguf}"
MMPROJ="${BONSAI_MMPROJ:-Bonsai-27B-mmproj-Q8_0.gguf}"
REPO="${BONSAI_HF_REPO:-prism-ml/Bonsai-27B-gguf}"
PORT="${BONSAI_PORT:-8080}"

mkdir -p "$MODELS_DIR"
cd "$MODELS_DIR"

download_if_missing() {
  local file="$1"
  if [ -f "$file" ]; then
    echo "[bonsai] found $file"
    return 0
  fi
  echo "[bonsai] downloading $REPO /$file from Hugging Face ..."
  python3 - <<PY
from huggingface_hub import hf_hub_download
import os
hf_hub_download(
    repo_id=os.environ.get("BONSAI_HF_REPO", "$REPO"),
    filename="$file",
    local_dir=".",
    local_dir_use_symlinks=False,
)
print("downloaded", "$file")
PY
}

download_if_missing "$GGUF"
if [ "${BONSAI_ENABLE_VISION:-1}" = "1" ]; then
  download_if_missing "$MMPROJ" || echo "[bonsai] mmproj download failed — text-only mode"
fi

echo "[bonsai] starting Python OpenAI-compatible server on :$PORT"
exec python3 /opt/bonsai/server.py
