#!/usr/bin/env bash
# Nestling container entrypoint: ensure data, then exec CMD (uvicorn).
set -euo pipefail
cd /app

echo "[entrypoint] Nestling bootstrapping..."

if [ ! -d extracted/asq ] || [ -z "$(ls -A extracted/asq 2>/dev/null || true)" ]; then
  echo "[entrypoint] extracted/asq missing — running extract_texts.py"
  python extract_texts.py || echo "[entrypoint] extract_texts.py failed (continuing if data present)"
else
  echo "[entrypoint] extracted/asq present"
fi

if [ ! -d data/en ] || [ -z "$(ls -A data/en 2>/dev/null || true)" ]; then
  echo "[entrypoint] data/en missing — running assistant.translate (glossary fallback ok offline)"
  python -m assistant.translate || echo "[entrypoint] translate failed (continuing)"
else
  echo "[entrypoint] data/en present"
fi

# Build medical RAG index if missing
if [ ! -f data/knowledge/rag_index/docs.json ]; then
  echo "[entrypoint] building medical RAG index"
  python - <<'PY'
from assistant.agent.orchestrator import ParentAssistant
try:
    n = ParentAssistant().refresh_medical_index()
    print(f"[entrypoint] indexed {n} medical chunks")
except Exception as e:
    print(f"[entrypoint] RAG index skipped: {e}")
PY
else
  echo "[entrypoint] medical RAG index present"
fi

mkdir -p data/children data/overlays

echo "[entrypoint] starting: $*"
exec "$@"
