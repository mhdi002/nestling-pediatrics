#!/usr/bin/env bash
set -euo pipefail

SERVED_MODEL="${NESTLING_LLM_MODEL:-openbmb/MiniCPM5-1B}"
PORT="${VLLM_PORT:-8000}"
MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-1536}"
GPU_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.86}"
TENSOR_PARALLEL="${VLLM_TENSOR_PARALLEL_SIZE:-1}"
ENFORCE_EAGER="${VLLM_ENFORCE_EAGER:-1}"
MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-1}"
KV_DTYPE="${VLLM_KV_CACHE_DTYPE:-fp8}"
QUANTIZATION="${VLLM_QUANTIZATION:-fp8}"
HF_HOME_DIR="${HF_HOME:-/root/.cache/huggingface}"
# Derive the HF cache dir from the model id (org/name -> models--org--name)
# instead of hardcoding it, so overriding NESTLING_LLM_MODEL actually works.
HUB_MODEL_DIR="${HF_HOME_DIR}/hub/models--$(printf '%s' "$SERVED_MODEL" | sed 's#/#--#g')"
MODEL_ROOT="${NESTLING_LLM_MODEL_PATH:-}"

if command -v python3 >/dev/null 2>&1; then PY=python3
elif command -v python >/dev/null 2>&1; then PY=python
else echo "[llm] ERROR: no python"; exit 1; fi

resolve_snapshot() {
  local root="$1" ref_file snapshot_rev snapshot_path first_snapshot
  ref_file="$root/refs/main"
  if [ -f "$ref_file" ]; then
    snapshot_rev="$(tr -d '\r\n' < "$ref_file")"
    snapshot_path="$root/snapshots/$snapshot_rev"
    if [ -d "$snapshot_path" ]; then echo "$snapshot_path"; return 0; fi
  fi
  first_snapshot="$(ls -1 "$root/snapshots" 2>/dev/null | head -n 1 || true)"
  if [ -n "$first_snapshot" ] && [ -d "$root/snapshots/$first_snapshot" ]; then
    echo "$root/snapshots/$first_snapshot"; return 0
  fi
  echo "$root"
}

weights_present() {
  local path="$1"
  ls "$path"/model*.safetensors >/dev/null 2>&1 || [ -f "$path/pytorch_model.bin" ]
}

SNAPSHOT=""
if [ -d "$HUB_MODEL_DIR" ]; then SNAPSHOT="$(resolve_snapshot "$HUB_MODEL_DIR")"
elif [ -n "$MODEL_ROOT" ] && [ -d "$MODEL_ROOT" ]; then SNAPSHOT="$(resolve_snapshot "$MODEL_ROOT")"; fi

if [ -z "$SNAPSHOT" ] || [ ! -d "$SNAPSHOT" ]; then
  echo "[llm] ERROR: model cache not found at ${HUB_MODEL_DIR}"; exit 1
fi
if ! weights_present "$SNAPSHOT"; then
  echo "[llm] ERROR: no weights under $SNAPSHOT"; exit 1
fi

MODEL_ARG="$SNAPSHOT"
if [ "${NESTLING_LLM_USE_MODEL_ID:-0}" = "1" ]; then MODEL_ARG="$SERVED_MODEL"; fi

EXTRA_ARGS=(--enforce-eager)
[ "$ENFORCE_EAGER" = "1" ] || EXTRA_ARGS=()
[ -n "$QUANTIZATION" ] && [ "$QUANTIZATION" != "none" ] && EXTRA_ARGS+=(--quantization "$QUANTIZATION")
[ -n "$KV_DTYPE" ] && [ "$KV_DTYPE" != "auto" ] && EXTRA_ARGS+=(--kv-cache-dtype "$KV_DTYPE")
# How many images a prompt may carry. Zero skips vision-tower profiling and
# leaves that VRAM for KV cache, which is the right trade on a small card --
# but it was pinned to zero on every GPU, so a 24 GB board also refused every
# image with "At most 0 image(s) may be provided in one prompt", and a parent
# who uploaded a photo of a rash was answered from their caption alone.
# deploy.sh derives this from the GPU and the checkpoint (scripts/size_llm.py)
# and passes it through; it stays 0 when the card has no room to spare.
LIMIT_MM_IMAGE="${VLLM_LIMIT_MM_IMAGE:-0}"
EXTRA_ARGS+=(--limit-mm-per-prompt "{\"image\":${LIMIT_MM_IMAGE}}")

echo "[llm] starting vLLM :$PORT python=$PY"
echo "[llm] model-arg=$MODEL_ARG quant=$QUANTIZATION kv=$KV_DTYPE"
echo "[llm] max_model_len=$MAX_MODEL_LEN gpu_util=$GPU_UTIL max_num_seqs=$MAX_NUM_SEQS"

exec "$PY" -m vllm.entrypoints.openai.api_server \
  --host 0.0.0.0 \
  --port "$PORT" \
  --model "$MODEL_ARG" \
  --served-model-name "$SERVED_MODEL" \
  --trust-remote-code \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-batched-tokens "$MAX_MODEL_LEN" \
  --gpu-memory-utilization "$GPU_UTIL" \
  --tensor-parallel-size "$TENSOR_PARALLEL" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --dtype auto \
  "${EXTRA_ARGS[@]}"
