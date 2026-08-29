#!/usr/bin/env bash
# Fetch model weights from an OCI registry (Docker Hub) instead of the
# Hugging Face Hub, and lay them out in the HF cache structure that
# docker/llm/entrypoint.sh already resolves.
#
# Why this exists: large HF LFS blobs are throttled or outright blocked on
# some networks (unauthenticated Hub downloads are rate-limited, and the LFS
# CDN is unreachable from some regions), while Docker Hub's CDN is fast and
# resumable. Docker's `ai/*-safetensors` repos publish each model file as its
# own OCI layer annotated with `org.cncf.model.filepath`, so the files can be
# reassembled directly from the registry API -- no `docker model` CLI plugin
# required, which matters because Ubuntu's docker.io package doesn't ship it.
#
# Usage: scripts/fetch_model_registry.sh [HF_CACHE_DIR]
set -Eeuo pipefail

MODEL_IMAGE="${NESTLING_MODEL_IMAGE:-ai/qwen3.5-safetensors:4B}"
MODEL_ID="${NESTLING_LLM_MODEL:-Qwen/Qwen3.5-4B}"
HF_CACHE="${1:-${NESTLING_HF_CACHE_HOST:-$HOME/.cache/huggingface}}"
JOBS="${NESTLING_MODEL_FETCH_JOBS:-4}"
REGISTRY="https://registry-1.docker.io"

REPO="${MODEL_IMAGE%:*}"
TAG="${MODEL_IMAGE##*:}"
# Qwen/Qwen3.5-4B -> models--Qwen--Qwen3.5-4B
HUB_SUBDIR="models--$(printf '%s' "$MODEL_ID" | sed 's#/#--#g')"
MODEL_DIR="$HF_CACHE/hub/$HUB_SUBDIR"
REV="dockerhub"
SNAPSHOT="$MODEL_DIR/snapshots/$REV"

step() { printf '\033[1;36m==>\033[0m %s\n' "$1"; }
ok()   { printf '\033[1;32m[ok]\033[0m %s\n' "$1"; }
err()  { printf '\033[1;31m[error]\033[0m %s\n' "$1" >&2; }

for c in curl python3; do
  command -v "$c" >/dev/null 2>&1 || { err "$c is required"; exit 1; }
done

step "authenticating to registry for $REPO"
TOKEN="$(curl -fsS --retry 3 --retry-delay 2 \
  "https://auth.docker.io/token?service=registry.docker.io&scope=repository:${REPO}:pull" \
  | python3 -c 'import json,sys; print(json.load(sys.stdin)["token"])')"
[ -n "$TOKEN" ] || { err "could not obtain registry token"; exit 1; }

step "fetching manifest for $MODEL_IMAGE"
MANIFEST="$(mktemp)"
trap 'rm -f "$MANIFEST"' EXIT
curl -fsS --retry 3 --retry-delay 2 \
  -H "Authorization: Bearer $TOKEN" \
  -H "Accept: application/vnd.oci.image.manifest.v1+json,application/vnd.docker.distribution.manifest.v2+json" \
  "$REGISTRY/v2/$REPO/manifests/$TAG" > "$MANIFEST"

# Emit "<digest> <filepath> <size>" per layer, skipping layers with no
# filepath annotation (e.g. duplicate LICENSE entries).
PLAN="$(mktemp)"
python3 - "$MANIFEST" > "$PLAN" <<'PY'
import json, sys
m = json.load(open(sys.argv[1]))
seen = set()
for layer in m.get("layers", []):
    ann = layer.get("annotations") or {}
    fp = ann.get("org.cncf.model.filepath")
    if not fp or fp in seen:
        continue
    seen.add(fp)
    print(f'{layer["digest"]}\t{fp}\t{layer["size"]}')
PY

total=$(awk -F'\t' '{s+=$3} END {printf "%.2f", s/1073741824}' "$PLAN")
count=$(wc -l < "$PLAN")
step "downloading $count files (${total} GB) -> $SNAPSHOT"

mkdir -p "$SNAPSHOT" "$MODEL_DIR/refs"

FAILDIR="$(mktemp -d)"
trap 'rm -f "$MANIFEST" "$PLAN"; rm -rf "$FAILDIR"' EXIT

fetch_one() {
  local digest="$1" filepath="$2" size="$3" dest="$SNAPSHOT/$filepath"
  mkdir -p "$(dirname "$dest")"
  # Skip if already complete, so re-runs resume instead of refetching 8+ GB.
  if [ -f "$dest" ] && [ "$(stat -c %s "$dest" 2>/dev/null || echo 0)" = "$size" ]; then
    printf '  [skip] %s (already complete)\n' "$filepath"
    return 0
  fi
  local attempt got
  # The registry redirects blobs to a CDN whose DNS/edge can fail
  # intermittently; retry the whole request, not just curl's internal retry.
  for attempt in 1 2 3 4 5; do
    if curl -fsS -L --retry 3 --retry-delay 2 --retry-all-errors \
         --connect-timeout 20 \
         -H "Authorization: Bearer $TOKEN" \
         -o "$dest.part" "$REGISTRY/v2/$REPO/blobs/$digest"; then
      got="$(stat -c %s "$dest.part" 2>/dev/null || echo 0)"
      if [ "$got" = "$size" ]; then
        mv "$dest.part" "$dest"
        printf '  [done] %s (%s MB)\n' "$filepath" "$((size / 1048576))"
        return 0
      fi
      printf '  [retry %s] %s: expected %s bytes, got %s\n' "$attempt" "$filepath" "$size" "$got" >&2
    else
      printf '  [retry %s] %s: transfer failed\n' "$attempt" "$filepath" >&2
    fi
    rm -f "$dest.part"
    sleep $((attempt * 3))
  done
  printf '  [fail] %s after 5 attempts\n' "$filepath" >&2
  : > "$FAILDIR/$(printf '%s' "$filepath" | tr '/' '_')"
  return 1
}

# Bounded parallelism via background jobs. Deliberately not xargs -I{}: the
# plan is tab-separated and round-tripping it through xargs quoting mangles
# the fields.
while IFS=$'\t' read -r digest filepath size; do
  [ -n "$filepath" ] || continue
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n 2>/dev/null || break; done
  fetch_one "$digest" "$filepath" "$size" &
done < "$PLAN"
wait

if [ -n "$(ls -A "$FAILDIR" 2>/dev/null)" ]; then
  err "one or more layers failed to download:"
  ls -1 "$FAILDIR" >&2
  err "re-run this script to resume (completed files are skipped)"
  exit 1
fi

printf '%s' "$REV" > "$MODEL_DIR/refs/main"

if ! ls "$SNAPSHOT"/model*.safetensors >/dev/null 2>&1; then
  err "no model*.safetensors present in $SNAPSHOT after fetch"
  exit 1
fi

ok "model ready at $SNAPSHOT"
du -sh "$SNAPSHOT" | awk '{print "    size: " $1}'
