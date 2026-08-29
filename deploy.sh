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
# auto = try the OCI registry first, fall back to the Hugging Face Hub.
MODEL_SOURCE=auto
INSTALL_PREREQS=1

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
  --model-source SRC      where to get weights: auto|hub|hf (default: auto)
                            hub = OCI registry (Docker Hub) -- fast, works where
                                  the Hugging Face LFS CDN is blocked/throttled
                            hf  = Hugging Face Hub via the `hf` CLI
                            auto= try hub, fall back to hf
  --no-install-prereqs    don't auto-install Docker / NVIDIA Container Toolkit
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

# ---------- prerequisite install (Linux/apt only) ----------
is_root()    { [ "$(id -u)" = "0" ]; }
have_apt()   { have_cmd apt-get; }

# Runs apt-get with retries. Two failure modes matter here: another apt/dpkg
# process still holding the lock (the Docker install may not have fully
# settled), and transient mirror errors. Swallowing these silently causes the
# far more confusing "Unable to locate package" later on.
# A freshly-booted Ubuntu box usually runs unattended-upgrades for several
# minutes, holding the dpkg lock. Wait it out rather than racing it.
# Ubuntu cloud images run unattended-upgrades on boot, and it can sit on the
# dpkg lock for many minutes (or wedge entirely on broken packages). Stopping
# the service is not enough: apt-daily.timer / apt-daily-upgrade.timer simply
# start it again, so the timers have to be masked for the duration too.
# Opt out with NESTLING_KEEP_AUTO_UPGRADES=1 if a host must keep them running.
pause_auto_upgrades() {
  [ "${NESTLING_KEEP_AUTO_UPGRADES:-0}" = "1" ] && return 0
  is_root || return 0
  have_cmd systemctl || return 0
  systemctl stop apt-daily.timer apt-daily-upgrade.timer unattended-upgrades >/dev/null 2>&1 || true
  systemctl mask apt-daily.timer apt-daily-upgrade.timer >/dev/null 2>&1 || true
}

wait_for_apt_lock() {
  local i announced=0
  pause_auto_upgrades
  for i in $(seq 1 60); do
    # `apt-get check` acquires the same locks a real install needs, so it is
    # an accurate probe. Matching process names is NOT: the harmless
    # unattended-upgrade-shutdown watcher runs permanently and looks identical
    # to the real upgrader, so a name match never clears.
    if apt-get -qq check >/dev/null 2>&1; then
      [ "$announced" = 1 ] && ok "package manager is free"
      return 0
    fi
    if [ "$announced" = 0 ]; then
      step "waiting for another package manager (e.g. unattended-upgrades) to release the dpkg lock"
      announced=1
    fi
    sleep 10
  done
  # A freshly-booted Ubuntu image can leave unattended-upgrades wedged retrying
  # broken packages, holding the lock indefinitely. Say so plainly rather than
  # failing later with a confusing "Unable to locate package".
  warn "dpkg lock still held after 10 minutes."
  warn "check the holder with: ps -eo pid,etime,args | grep -E 'apt|dpkg|unattended'"
  warn "a wedged upgrader can be cleared with: pkill -9 -f /usr/bin/unattended-upgrade"
  return 1
}

apt_retry() {
  local i out rc
  for i in $(seq 1 12); do
    if out="$(DEBIAN_FRONTEND=noninteractive apt-get "$@" 2>&1)"; then
      return 0
    fi
    rc=$?
    case "$out" in
      *"Could not get lock"*|*"Unable to lock"*|*"held by process"*)
        [ "$i" = 1 ] && step "waiting for another apt process to finish"
        sleep 10
        ;;
      *)
        sleep 5
        ;;
    esac
  done
  printf '%s\n' "$out" >&2
  return "${rc:-1}"
}

# Docker's own repo is occasionally unreachable; the distro package is a
# perfectly good fallback and ships compose v2 alongside it.
install_docker_linux() {
  step "installing Docker (not present)"
  export DEBIAN_FRONTEND=noninteractive
  wait_for_apt_lock || true
  apt_retry update -qq >/dev/null 2>&1 || warn "apt-get update had errors -- continuing"
  if curl -fsSL https://get.docker.com -o /tmp/get-docker.sh 2>/dev/null \
     && sh /tmp/get-docker.sh >/tmp/nestling-docker-install.log 2>&1; then
    ok "installed Docker from get.docker.com"
  elif apt_retry install -y -qq docker.io docker-compose-v2 >>/tmp/nestling-docker-install.log 2>&1; then
    ok "installed Docker from the distro repo"
  else
    err "could not install Docker automatically -- see /tmp/nestling-docker-install.log"
    err "install it manually: https://docs.docker.com/engine/install/"
    exit 1
  fi
  systemctl enable --now docker >/dev/null 2>&1 || true
}

install_nvidia_toolkit_linux() {
  step "installing NVIDIA Container Toolkit (not present)"
  export DEBIAN_FRONTEND=noninteractive
  wait_for_apt_lock || true
  curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey 2>/dev/null \
    | gpg --dearmor --yes -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg 2>/dev/null || {
      warn "could not fetch the NVIDIA toolkit signing key"; return 1; }
  curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list 2>/dev/null \
    | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
    > /etc/apt/sources.list.d/nvidia-container-toolkit.list
  if ! apt_retry update -qq >/tmp/nestling-nvidia-aptupdate.log 2>&1; then
    warn "apt-get update failed while adding the NVIDIA repo -- see /tmp/nestling-nvidia-aptupdate.log"
    return 1
  fi
  apt_retry install -y -qq nvidia-container-toolkit >/tmp/nestling-nvidia-install.log 2>&1 || true
  # Verify the outcome instead of trusting the exit code: a lock contention or
  # a partially-applied repo can leave apt reporting success with nothing
  # installed, which then fails much later and far less obviously.
  if ! have_cmd nvidia-ctk; then
    warn "NVIDIA Container Toolkit did not install -- see /tmp/nestling-nvidia-install.log"
    return 1
  fi
  nvidia-ctk runtime configure --runtime=docker >/dev/null 2>&1 || true
  # Docker 28+ resolves `--gpus` through the Container Device Interface, so a
  # CDI spec must exist or the daemon reports "no known GPU vendor found"
  # even with the toolkit and driver installed.
  mkdir -p /etc/cdi
  if ! nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml >/tmp/nestling-cdi.log 2>&1; then
    warn "could not generate the CDI spec -- see /tmp/nestling-cdi.log"
  fi
  systemctl restart docker >/dev/null 2>&1 || true
  sleep 8
  ok "NVIDIA Container Toolkit installed"
}

# ---------- prerequisite checks ----------
assert_docker() {
  if ! have_cmd docker; then
    if [ "$INSTALL_PREREQS" = "1" ] && is_root && have_apt; then
      install_docker_linux
    else
      err "Docker not found. Install: https://docs.docker.com/engine/install/"
      [ "$INSTALL_PREREQS" = "1" ] || err "(auto-install disabled via --no-install-prereqs)"
      is_root || err "(auto-install needs root)"
      exit 1
    fi
  fi
  if ! run_with_timeout 15 docker version >/dev/null 2>&1; then
    # A just-installed daemon may still be coming up.
    systemctl start docker >/dev/null 2>&1 || true
    sleep 5
    if ! run_with_timeout 15 docker version >/dev/null 2>&1; then
      err "Docker daemon not reachable (not running, or permission denied -- is your user in the 'docker' group?)"
      exit 1
    fi
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
    # The driver is there, so the missing piece is almost always the container
    # runtime hook. Install it and retry once before giving up on the GPU.
    if [ "$INSTALL_PREREQS" = "1" ] && is_root && have_apt && ! have_cmd nvidia-ctk; then
      install_nvidia_toolkit_linux || true
    fi
    if ! docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
      warn "GPU driver found but 'docker run --gpus all' failed -- install the NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html"
      return 1
    fi
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
  # Secure by default: without a key the API is fully open, and it serves
  # children's names, dates of birth, growth measurements and chat history.
  local api_key
  if have_cmd openssl; then
    api_key="$(openssl rand -hex 32)"
  else
    api_key="$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  fi
  # Interactive login for the web UI: a username/password is far easier to use
  # on a phone than a 64-char key, and the key stays available for scripts.
  local auth_user="${NESTLING_AUTH_USERNAME:-admin}"
  local auth_pass auth_hash auth_secret
  auth_pass="${NESTLING_AUTH_PASSWORD:-}"
  if [ -z "$auth_pass" ]; then
    # Readable but high-entropy: 4 hex groups (~64 bits), typable on a phone.
    auth_pass="$(printf 'nestling-%s-%s-%s' \
      "$(head -c 3 /dev/urandom | od -An -tx1 | tr -d ' \n')" \
      "$(head -c 3 /dev/urandom | od -An -tx1 | tr -d ' \n')" \
      "$(head -c 2 /dev/urandom | od -An -tx1 | tr -d ' \n')")"
  fi
  if have_cmd openssl; then
    auth_secret="$(openssl rand -hex 32)"
  else
    auth_secret="$(head -c 32 /dev/urandom | od -An -tx1 | tr -d ' \n')"
  fi
  # Hash with stdlib only. Importing app.api.auth here would drag in FastAPI,
  # which is installed in the container, not on the host -- keep the format in
  # sync with hash_password() in app/api/auth.py. The ':' separator matters:
  # Docker Compose interpolates '$' inside .env values and would truncate it.
  auth_hash="$(NESTLING_PW="$auth_pass" python3 -c "
import hashlib, os, secrets
pw = os.environ['NESTLING_PW'].encode('utf-8')
iterations = 240000
salt = secrets.token_bytes(16)
digest = hashlib.pbkdf2_hmac('sha256', pw, salt, iterations)
print(':'.join(['pbkdf2_sha256', str(iterations), salt.hex(), digest.hex()]))
" 2>/dev/null || true)"
  if [ -z "$auth_hash" ]; then
    warn "could not hash the login password (python3 unavailable?) -- login stays disabled"
  fi
  NESTLING_GENERATED_USER="$auth_user"
  NESTLING_GENERATED_PASS="$auth_pass"

  cat > .env <<EOF
# Created by deploy.sh on $(date -u +%Y-%m-%dT%H:%M:%SZ). Safe to edit;
# deploy.sh/deploy.ps1 will not overwrite an existing .env.
NESTLING_HF_CACHE_HOST=$default_hf_cache

# --- Web UI sign-in (username + password) ---
# The UI shows a sign-in form and exchanges these for a short-lived bearer
# token. Only the PBKDF2 hash is stored here, never the password itself.
NESTLING_AUTH_USERNAME=$auth_user
NESTLING_AUTH_PASSWORD_HASH=$auth_hash
NESTLING_AUTH_SECRET=$auth_secret
NESTLING_SESSION_TTL_HOURS=12

# Generated API key. Every /api/* route except /api/health requires it via
# the X-API-Key header (or Authorization: Bearer). The web UI asks for it
# once and keeps it in the browser's localStorage. It is deliberately NOT
# embedded in the served HTML -- that would hand the key to any visitor and
# defeat the point. For a trusted private network you may instead add
#   <meta name="nestling-api-key" content="...">
# to web/index.html, or set this to empty to disable auth entirely.
NESTLING_API_KEY=$api_key

# Restrict this to your real origins once the domain is known; "*" is only
# safe because credentials are never sent cross-origin.
NESTLING_CORS_ORIGINS=*
# NESTLING_LB_HOST_PORT=8080
# NESTLING_LLM_HOST_PORT=8001
# NESTLING_INSTALL_ML=0
EOF
  chmod 600 .env
  ok "wrote .env (HF cache=$default_hf_cache, generated login + API key)"
  echo
  echo "  ------------------------------------------------------------"
  echo "  WEB UI SIGN-IN (save these -- the password is not stored)"
  echo "      username: $auth_user"
  echo "      password: $auth_pass"
  echo "  ------------------------------------------------------------"
  echo
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

download_model_from_registry() {
  local hf_home="$1"
  [ -x "$ROOT/scripts/fetch_model_registry.sh" ] || chmod +x "$ROOT/scripts/fetch_model_registry.sh" 2>/dev/null || true
  if [ ! -f "$ROOT/scripts/fetch_model_registry.sh" ]; then
    warn "scripts/fetch_model_registry.sh missing -- cannot use the registry source"
    return 1
  fi
  bash "$ROOT/scripts/fetch_model_registry.sh" "$hf_home"
}

download_model_from_hf() {
  local hf_home="$1"
  install_hf_cli
  step "downloading $MODEL_ID from the Hugging Face Hub (several GB)"
  HF_HOME="$hf_home" "$HF_CMD" download "$MODEL_ID"
}

download_model() {
  local hf_home
  hf_home="$(hf_cache_from_env)"
  mkdir -p "$hf_home"
  if model_downloaded "$hf_home"; then
    ok "model already cached at $hf_home -- skipping download"
    return
  fi

  case "$MODEL_SOURCE" in
    hub)
      download_model_from_registry "$hf_home" || { err "registry model fetch failed"; exit 1; }
      ;;
    hf)
      download_model_from_hf "$hf_home" || { err "Hugging Face model download failed"; exit 1; }
      ;;
    auto)
      # Prefer the registry: it is CDN-backed and resumable, and the HF LFS
      # endpoint is throttled or unreachable on some networks.
      if ! download_model_from_registry "$hf_home"; then
        warn "registry fetch failed -- falling back to the Hugging Face Hub"
        download_model_from_hf "$hf_home" || {
          err "both the registry and Hugging Face downloads failed."
          err "retry with: ./deploy.sh --model-only"
          exit 1
        }
      fi
      ;;
    *)
      err "unknown --model-source: $MODEL_SOURCE"
      exit 2
      ;;
  esac

  if model_downloaded "$hf_home"; then
    ok "model ready at $hf_home"
  else
    err "download finished but no weights found under $hf_home/hub/$HUB_SUBDIR"
    exit 1
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
  # nginx resolves its upstream once at startup, so a recreated app container
  # leaves it proxying an address that no longer exists and every request
  # hangs until the proxy timeout -- a 504 that looks like a slow backend
  # rather than a stale route. Restarting it last forces re-resolution.
  step "restarting the load balancer so it re-resolves the app"
  docker compose restart nginx >/dev/null 2>&1 || warn "could not restart nginx"
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
    --model-source)
      MODEL_SOURCE="${2:-}"
      case "$MODEL_SOURCE" in
        auto|hub|hf) ;;
        *) err "--model-source must be 'auto', 'hub' or 'hf'"; exit 2 ;;
      esac
      shift 2
      ;;
    --no-install-prereqs) INSTALL_PREREQS=0; shift ;;
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
