#!/usr/bin/env bash
# Nestling deploy automation (Linux/macOS). See deploy.ps1 for the Windows
# equivalent, and `make help` for the short-form wrapper targets.
#
# Fully automates: prerequisite checks (Docker, GPU/NVIDIA Container
# Toolkit), .env bootstrap, LLM model download, `docker compose build/up`,
# health polling, and a post-deploy summary. vLLM and CUDA are never
# installed on the host directly -- they live inside the `llm` service's
# container image (docker/llm/Dockerfile, FROM vllm/vllm-openai) -- this
# script only needs the host GPU driver + NVIDIA Container Toolkit so that
# image can see the GPU.
set -Eeuo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

MODEL_ID="Qwen/Qwen3.5-4B"
HUB_SUBDIR="models--Qwen--Qwen3.5-4B"
LB_PORT="${NESTLING_LB_HOST_PORT:-8080}"
LLM_PORT="${NESTLING_LLM_HOST_PORT:-8001}"
HF_CMD=""
# Fallback location for the huggingface_hub CLI on PEP 668 systems.
HF_VENV="$ROOT/.venv-hf"

MODE=full
SKIP_MODEL_DOWNLOAD=0
ACTION=deploy
LOGS_SERVICE=""
ASSUME_YES=0

# ---------- console helpers ----------
if [ -t 1 ]; then
  c_cyan=$'\033[1;36m'; c_green=$'\033[1;32m'; c_yellow=$'\033[1;33m'; c_red=$'\033[1;31m'; c_reset=$'\033[0m'
else
  c_cyan=""; c_green=""; c_yellow=""; c_red=""; c_reset=""
fi
step()  { printf '%s==>%s %s\n' "$c_cyan" "$c_reset" "$1"; }
ok()    { printf '%s[ok]%s %s\n' "$c_green" "$c_reset" "$1"; }
warn()  { printf '%s[warn]%s %s\n' "$c_yellow" "$c_reset" "$1" >&2; }
err()   { printf '%s[error]%s %s\n' "$c_red" "$c_reset" "$1" >&2; }

trap 'err "deploy.sh failed (line $LINENO)"' ERR

usage() {
  cat <<'EOF'
Usage: ./deploy.sh [action] [options]

Actions (default: deploy):
  (none)                  run the full deploy pipeline
  --down                  stop the stack (keeps data volumes)
  --clean                 stop the stack AND delete data volumes (destructive)
  --logs [service]        tail logs (all services, or one: nginx|nestling|llm)
  --status                health check + `docker compose ps`, no changes
  --model-only            download the LLM model only, then exit
  -h, --help              show this help

Options (deploy action only):
  --mode app|full         app-only, or full stack with GPU LLM sidecar (default: full)
  --skip-model-download   don't try to download the model even in full mode
  -y, --yes               auto-confirm destructive prompts (used by --clean)
EOF
}

have_cmd() { command -v "$1" >/dev/null 2>&1; }

# Runs "$@" with a hard wall-clock cap when the `timeout` coreutil is
# available (Linux, and this Windows/Cygwin bash); falls back to running
# uncapped on hosts without it (e.g. stock macOS) rather than failing.
run_with_timeout() {
  local secs="$1"; shift
  if have_cmd timeout; then
    timeout "$secs" "$@"
  else
    "$@"
  fi
}

# ---------- prerequisite checks ----------
assert_docker() {
  if ! have_cmd docker; then
    err "Docker not found. Install: https://docs.docker.com/engine/install/"
    exit 1
  fi
  if ! run_with_timeout 10 docker version >/dev/null 2>&1; then
    err "Docker daemon not reachable (not running, or permission denied -- is your user in the 'docker' group?)"
    exit 1
  fi
  if ! docker compose version >/dev/null 2>&1; then
    err "Docker Compose v2 plugin not found (bundled with modern Docker)."
    exit 1
  fi
  ok "Docker + Compose v2 available"
}

gpu_available() {
  if ! have_cmd nvidia-smi; then
    warn "no nvidia-smi found -- no NVIDIA driver on this host"
    return 1
  fi
  if ! nvidia-smi --query-gpu=name --format=csv,noheader >/dev/null 2>&1; then
    warn "nvidia-smi present but failed to query a GPU"
    return 1
  fi
  if ! docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
    warn "GPU driver found but 'docker run --gpus all' failed -- install the NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
    return 1
  fi
  ok "GPU detected and usable by Docker"
  return 0
}

# ---------- .env bootstrap ----------
init_env_file() {
  if [ -f .env ]; then
    ok ".env already present, leaving as-is"
    return
  fi
  local default_hf_cache="${NESTLING_HF_CACHE_HOST:-${HF_HOME:-$HOME/.cache/huggingface}}"
  mkdir -p "$default_hf_cache"
  cat > .env <<EOF
# Created by deploy.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ). Safe to edit;
# deploy.sh/deploy.ps1 will not overwrite an existing .env.
NESTLING_HF_CACHE_HOST=$default_hf_cache
# NESTLING_LB_HOST_PORT=8080
# NESTLING_LLM_HOST_PORT=8001
# NESTLING_INSTALL_ML=0
EOF
  ok "wrote .env (NESTLING_HF_CACHE_HOST=$default_hf_cache)"
}

hf_cache_from_env() {
  if [ -f .env ]; then
    local v
    v="$(grep -m1 '^NESTLING_HF_CACHE_HOST=' .env | cut -d= -f2-)"
    if [ -n "$v" ]; then echo "$v"; return; fi
  fi
  echo "${NESTLING_HF_CACHE_HOST:-${HF_HOME:-$HOME/.cache/huggingface}}"
}

# ---------- model download (mirrors docker/llm/entrypoint.sh) ----------
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
  return 1
}

weights_present() {
  local path="$1"
  ls "$path"/model*.safetensors >/dev/null 2>&1 || [ -f "$path/pytorch_model.bin" ]
}

model_downloaded() {
  local hf_home="$1" hub_dir snapshot
  hub_dir="$hf_home/hub/$HUB_SUBDIR"
  [ -d "$hub_dir" ] || return 1
  snapshot="$(resolve_snapshot "$hub_dir")" || return 1
  weights_present "$snapshot"
}

install_hf_cli() {
  if have_cmd hf; then HF_CMD=hf; return; fi
  if have_cmd huggingface-cli; then HF_CMD=huggingface-cli; return; fi
  # left behind by a previous run
  if [ -x "$HF_VENV/bin/hf" ]; then HF_CMD="$HF_VENV/bin/hf"; return; fi
  if [ -x "$HF_VENV/bin/huggingface-cli" ]; then HF_CMD="$HF_VENV/bin/huggingface-cli"; return; fi

  step "installing huggingface_hub CLI"
  local py=python3
  have_cmd python3 || py=python
  if ! have_cmd "$py"; then
    err "no python3/python found -- install Python 3 to download the model, or download it manually."
    exit 1
  fi

  # Try a plain --user install first. Note that ~/.local/bin is frequently not
  # on PATH (root shells especially), so probe the install location directly
  # rather than trusting have_cmd alone.
  if "$py" -m pip install --user -q -U "huggingface_hub[cli]" >/dev/null 2>&1; then
    have_cmd hf && { HF_CMD=hf; return; }
    have_cmd huggingface-cli && { HF_CMD=huggingface-cli; return; }
    [ -x "$HOME/.local/bin/hf" ] && { HF_CMD="$HOME/.local/bin/hf"; return; }
    [ -x "$HOME/.local/bin/huggingface-cli" ] && { HF_CMD="$HOME/.local/bin/huggingface-cli"; return; }
  fi

  # Debian/Ubuntu (24.04+) mark the system Python "externally managed" (PEP 668)
  # and refuse --user installs. Use an isolated venv rather than
  # --break-system-packages, which can damage the distro's own Python.
  warn "system pip install unavailable (PEP 668) -- using an isolated venv"
  if ! "$py" -m venv "$HF_VENV" >/dev/null 2>&1; then
    err "could not create a venv at $HF_VENV. Install the venv module first: apt-get install -y python3-venv"
    exit 1
  fi
  if ! "$HF_VENV/bin/pip" install -q -U "huggingface_hub[cli]" >/dev/null 2>&1; then
    err "pip install huggingface_hub failed inside $HF_VENV"
    exit 1
  fi
  [ -x "$HF_VENV/bin/hf" ] && { HF_CMD="$HF_VENV/bin/hf"; return; }
  [ -x "$HF_VENV/bin/huggingface-cli" ] && { HF_CMD="$HF_VENV/bin/huggingface-cli"; return; }
  err "huggingface_hub CLI not found after installing into $HF_VENV"
  exit 1
}

download_model() {
  local hf_home
  hf_home="$(hf_cache_from_env)"
  mkdir -p "$hf_home"
  if model_downloaded "$hf_home"; then
    ok "model already cached at $hf_home -- skipping download"
    return
  fi
  install_hf_cli
  step "downloading $MODEL_ID (several GB, this can take a while)"
  if ! HF_HOME="$hf_home" "$HF_CMD" download "$MODEL_ID"; then
    err "model download failed. Check network access to huggingface.co and retry, or run: ./deploy.sh --model-only"
    exit 1
  fi
  if model_downloaded "$hf_home"; then
    ok "model downloaded to $hf_home"
  else
    warn "download finished but weights not found at the expected path ($hf_home/hub/$HUB_SUBDIR) -- the llm service may fail to start"
  fi
}

# ---------- compose orchestration ----------
compose_build() {
  if [ "$MODE" = full ]; then
    docker compose --profile llm build
  else
    docker compose build nestling nginx
  fi
}

compose_up() {
  if [ "$MODE" = full ]; then
    docker compose --profile llm up --build -d
  else
    docker compose up --build -d nestling nginx
  fi
}

wait_health() {
  local url="$1" budget="${2:-180}" waited=0
  step "waiting for $url (up to ${budget}s)"
  while [ "$waited" -lt "$budget" ]; do
    if curl -fsS --max-time 5 "$url" >/dev/null 2>&1; then
      ok "healthy: $url"
      return 0
    fi
    sleep 3
    waited=$((waited + 3))
  done
  err "health check timed out after ${budget}s: $url"
  run_with_timeout 10 docker compose logs --tail 80 || true
  exit 1
}

wait_llm_health() {
  local url="http://localhost:${LLM_PORT}/v1/models" waited=0 budget=30
  step "checking LLM sidecar (advisory only -- first load can take up to 15 min)"
  while [ "$waited" -lt "$budget" ]; do
    if curl -fsS --max-time 5 "$url" >/dev/null 2>&1; then
      ok "LLM sidecar ready: $url"
      return 0
    fi
    sleep 3
    waited=$((waited + 3))
  done
  warn "LLM sidecar not ready yet -- this is normal on first run (large model load)."
  warn "watch progress with: docker compose --profile llm logs -f llm"
}

db_migrate_safety() {
  step "running DB schema safety net inside the container"
  if ! docker compose exec -T nestling python scripts/migrate_db.py; then
    warn "migrate_db.py step failed (non-fatal -- the app creates tables lazily on first use)"
  else
    ok "database schema verified"
  fi
}

show_summary() {
  local gpu_used="$1"
  echo
  echo "=================================================="
  echo "Nestling is up."
  echo
  echo "  Web UI / API (via nginx):  http://localhost:${LB_PORT}"
  echo "  Health check:              http://localhost:${LB_PORT}/api/health"
  if [ "$MODE" = full ] && [ "$gpu_used" = 1 ]; then
    echo "  LLM (OpenAI-compatible):   http://localhost:${LLM_PORT}/v1"
  elif [ "$MODE" = app ]; then
    echo
    echo "  Note: deployed in APP-ONLY mode -- chat uses extractive RAG, not generative Qwen."
    echo "  Re-run with GPU + toolkit available: ./deploy.sh --mode full"
  fi
  echo
  echo "  Logs:      ./deploy.sh --logs"
  echo "  Stop:      ./deploy.sh --down"
  echo "  Wipe data: ./deploy.sh --clean   (DESTROYS volumes)"
  echo "=================================================="
}

do_down() {
  docker compose --profile llm down
}

do_clean() {
  if [ "$ASSUME_YES" -ne 1 ]; then
    read -r -p "This will DELETE all Nestling data volumes (children DB, uploads, HF cache volume). Type 'yes' to continue: " reply
    if [ "$reply" != "yes" ]; then
      err "aborted"
      exit 1
    fi
  fi
  docker compose --profile llm down -v
}

show_logs() {
  docker compose --profile llm logs -f ${LOGS_SERVICE:+"$LOGS_SERVICE"}
}

show_status() {
  local url="http://localhost:${LB_PORT}/api/health"
  if curl -fsS --max-time 5 "$url" 2>/dev/null; then echo; else warn "not reachable: $url"; fi
  if ! run_with_timeout 10 docker compose ps; then
    warn "docker compose ps did not respond (Docker daemon likely unreachable)"
  fi
}

# ---------- arg parsing ----------
while [ $# -gt 0 ]; do
  case "$1" in
    --mode)
      MODE="${2:-}"
      if [ "$MODE" != app ] && [ "$MODE" != full ]; then
        err "--mode must be 'app' or 'full'"
        exit 2
      fi
      shift 2
      ;;
    --skip-model-download) SKIP_MODEL_DOWNLOAD=1; shift ;;
    --down) ACTION=down; shift ;;
    --clean) ACTION=clean; shift ;;
    --logs)
      ACTION=logs
      shift
      if [ $# -gt 0 ] && [[ "$1" != --* ]]; then LOGS_SERVICE="$1"; shift; fi
      ;;
    --status) ACTION=status; shift ;;
    --model-only) ACTION=model-only; shift ;;
    -y|--yes) ASSUME_YES=1; shift ;;
    -h|--help) ACTION=help; shift ;;
    *)
      err "unknown argument: $1"
      usage
      exit 2
      ;;
  esac
done

# ---------- main ----------
case "$ACTION" in
  help)
    usage
    exit 0
    ;;
  down)
    do_down
    exit 0
    ;;
  clean)
    do_clean
    exit 0
    ;;
  logs)
    show_logs
    exit 0
    ;;
  status)
    show_status
    exit 0
    ;;
  model-only)
    init_env_file
    download_model
    exit 0
    ;;
  deploy)
    step "checking prerequisites"
    assert_docker

    step "preparing .env"
    init_env_file

    gpu=0
    if [ "$MODE" = full ]; then
      step "checking for GPU"
      if gpu_available; then
        gpu=1
      else
        warn "no usable GPU -- falling back to app-only deploy"
        MODE=app
      fi
    fi

    if [ "$MODE" = full ] && [ "$SKIP_MODEL_DOWNLOAD" -ne 1 ]; then
      step "ensuring LLM model is downloaded"
      download_model
    elif [ "$MODE" = full ]; then
      warn "skipping model download (--skip-model-download) -- llm service will fail to start if weights are missing"
    fi

    step "building images"
    compose_build

    step "starting stack (mode=$MODE)"
    compose_up

    wait_health "http://localhost:${LB_PORT}/api/health" 180
    if [ "$MODE" = full ]; then
      wait_llm_health
    fi

    db_migrate_safety

    show_summary "$gpu"
    ;;
esac
