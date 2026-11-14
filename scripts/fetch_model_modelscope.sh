#!/usr/bin/env bash
# Fetch model weights from ModelScope, laid out in the Hugging Face cache
# structure that docker/llm/entrypoint.sh already resolves.
#
# Why this exists: the Hugging Face CDN is unreachable from some networks
# this deploys to. Not throttled -- unreachable. On the box this was written
# for, huggingface.co's API answered in 0.4s while every CDN host it hands
# out (us.aws.cdn.hf.co, cas-server.xethub.hf.co, cdn-lfs-us-1.hf.co) timed
# out, so the metadata fetch succeeded and the download died ten files of
# eleven in. The Docker Hub route exists for the same reason and does not
# cover every model: its ai/ namespace publishes seven safetensors repos and
# MiniCPM is not among them.
#
# ModelScope hosts the same files -- byte-identical sizes -- on a CDN that is
# reachable there, and it is where the Chinese labs publish first, which is
# exactly the set of models the other two routes are worst at.
#
# Usage: scripts/fetch_model_modelscope.sh [HF_CACHE_DIR]
set -Eeuo pipefail

MODEL_ID="${NESTLING_LLM_MODEL:-Qwen/Qwen3.5-2B}"
HF_CACHE="${1:-${NESTLING_HF_CACHE_HOST:-$HOME/.cache/huggingface}}"
JOBS="${NESTLING_MODEL_FETCH_JOBS:-4}"
BASE="${NESTLING_MODELSCOPE_BASE:-https://www.modelscope.cn}"
REVISION="${NESTLING_MODELSCOPE_REVISION:-master}"

_HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
_PAIRS="$_HERE/config/model_mirrors.txt"

step() { printf '\033[1;36m==>\033[0m %s\n' "$1"; }
ok()   { printf '\033[1;32m[ok]\033[0m %s\n' "$1"; }
err()  { printf '\033[1;31m[error]\033[0m %s\n' "$1" >&2; }

for c in curl python3; do
  command -v "$c" >/dev/null 2>&1 || { err "$c is required"; exit 1; }
done

# Which ModelScope repo carries this model. Paired rather than derived: the
# id is usually the same name under a differently-cased org (openbmb ->
# OpenBMB), and "usually" is not good enough to build a download on.
MS_ID="${NESTLING_MODELSCOPE_ID:-}"
if [ -z "$MS_ID" ] && [ -f "$_PAIRS" ]; then
  MS_ID="$(awk -v m="$MODEL_ID" '$1==m {print $2; exit}' "$_PAIRS")"
fi
if [ -z "$MS_ID" ]; then
  err "no ModelScope repo is paired with $MODEL_ID in config/model_mirrors.txt"
  exit 3
fi

# openbmb/MiniCPM5-1B -> models--openbmb--MiniCPM5-1B, keyed on the Hugging
# Face id because that is the name the sidecar is served under and the name
# the cache layout is built from. Where the weights came from is not part of
# the model's identity.
HUB_SUBDIR="models--$(printf '%s' "$MODEL_ID" | sed 's#/#--#g')"
MODEL_DIR="$HF_CACHE/hub/$HUB_SUBDIR"
REV="modelscope"
SNAPSHOT="$MODEL_DIR/snapshots/$REV"

step "listing $MS_ID on ModelScope"
PLAN="$(mktemp)"
FAILDIR="$(mktemp -d)"
trap 'rm -f "$PLAN"; rm -rf "$FAILDIR"' EXIT

# The listing is fetched to a file first, so a refusal is reported with the
# server's own message rather than as a JSON parse error on an empty body.
# ModelScope rate-limits bursts with 403/429 ("操作过于频繁"), which clears
# after a short wait, so those are retried with a growing backoff.
LISTING="$(mktemp)"
trap 'rm -f "$PLAN" "$LISTING"; rm -rf "$FAILDIR"' EXIT
code=000
for wait_s in 0 15 30 60 120; do
  [ "$wait_s" -gt 0 ] && { echo "ModelScope answered $code; retrying in ${wait_s}s"; sleep "$wait_s"; }
  code="$(curl -sS --retry 3 --retry-delay 2 -o "$LISTING" -w '%{http_code}' \
    "$BASE/api/v1/models/$MS_ID/repo/files?Revision=$REVISION" 2>/dev/null || echo 000)"
  case "$code" in
    2??) break ;;
    403|429|5??|000) continue ;;
    *) break ;;
  esac
done
if [ "${code#2}" = "$code" ]; then
  echo "ModelScope listing failed (HTTP $code): $(head -c 300 "$LISTING")" >&2
  exit 1
fi

python3 -c '
import json, sys

data = json.load(sys.stdin)
files = (data.get("Data") or {}).get("Files") or []
if not files:
    sys.exit("ModelScope listed no files")
for f in files:
    path, size = f.get("Path"), f.get("Size")
    # Directories come back with no size; documentation and git metadata are
    # weight-adjacent noise this does not need to move.
    if not path or size is None:
        continue
    if path.startswith(".") or path.lower().endswith((".md", ".png", ".jpg")):
        continue
    print(f"{path}\t{size}")
' < "$LISTING" > "$PLAN"

count=$(wc -l < "$PLAN")
total=$(awk -F'\t' '{s+=$2} END {printf "%.2f", s/1073741824}' "$PLAN")
[ "$count" -gt 0 ] || { err "nothing to download"; exit 1; }
step "downloading $count files (${total} GB) -> $SNAPSHOT"

mkdir -p "$SNAPSHOT" "$MODEL_DIR/refs"

fetch_one() {
  local path="$1" size="$2" dest="$SNAPSHOT/$path"
  mkdir -p "$(dirname "$dest")"
  # Skip what is already complete, so a re-run resumes instead of refetching
  # two gigabytes.
  if [ -f "$dest" ] && [ "$(stat -c %s "$dest" 2>/dev/null || echo 0)" = "$size" ]; then
    printf '  [skip] %s (already complete)\n' "$path"
    return 0
  fi
  local attempt got
  for attempt in 1 2 3 4 5; do
    # --continue-at resumes a part file across attempts: on a slow link the
    # largest shard is the one most likely to be cut, and restarting it from
    # zero five times is how a retry loop becomes an infinite one.
    if curl -fsSL --retry 3 --retry-delay 2 --retry-all-errors \
         --connect-timeout 20 --continue-at - \
         -o "$dest.part" "$BASE/models/$MS_ID/resolve/$REVISION/$path"; then
      got="$(stat -c %s "$dest.part" 2>/dev/null || echo 0)"
      if [ "$got" = "$size" ]; then
        mv "$dest.part" "$dest"
        printf '  [done] %s (%s MB)\n' "$path" "$((size / 1048576))"
        return 0
      fi
      printf '  [retry %s] %s: expected %s bytes, got %s\n' "$attempt" "$path" "$size" "$got" >&2
      # A short file that will not grow is a bad transfer, not a partial one;
      # resuming from it would retry the same truncation forever.
      [ "$got" -gt "$size" ] && rm -f "$dest.part"
    else
      printf '  [retry %s] %s: transfer failed\n' "$attempt" "$path" >&2
    fi
    sleep $((attempt * 3))
  done
  printf '  [fail] %s after 5 attempts\n' "$path" >&2
  : > "$FAILDIR/$(printf '%s' "$path" | tr '/' '_')"
  return 1
}

while IFS=$'\t' read -r path size; do
  [ -n "$path" ] || continue
  while [ "$(jobs -rp | wc -l)" -ge "$JOBS" ]; do wait -n 2>/dev/null || break; done
  fetch_one "$path" "$size" &
done < "$PLAN"
wait

if [ -n "$(ls -A "$FAILDIR" 2>/dev/null)" ]; then
  err "one or more files failed to download:"
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
