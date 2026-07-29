#!/usr/bin/env python3
"""Container bootstrap: seed knowledge volume, index RAG, then exec uvicorn."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SEED = Path("/opt/nestling-seed")


def _nonempty_dir(p: Path) -> bool:
    return p.is_dir() and any(p.iterdir())


def _seed_knowledge() -> None:
    """Named volume can mount empty over /app/data/knowledge — restore from image seed."""
    knowledge = ROOT / "data" / "knowledge"
    chunks = knowledge / "chunks.json"
    seed_chunks = SEED / "knowledge" / "chunks.json"
    if chunks.exists():
        return
    if seed_chunks.exists():
        print("[entrypoint] seeding knowledge volume from image", flush=True)
        knowledge.mkdir(parents=True, exist_ok=True)
        for item in (SEED / "knowledge").iterdir():
            dest = knowledge / item.name
            if dest.exists():
                continue
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)


def main() -> int:
    os.chdir(ROOT)
    print("[entrypoint] Nestling bootstrapping...", flush=True)

    (ROOT / "data" / "children").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "overlays").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "uploads").mkdir(parents=True, exist_ok=True)
    (ROOT / "data" / "knowledge").mkdir(parents=True, exist_ok=True)

    _seed_knowledge()

    extracted_asq = ROOT / "extracted" / "asq"
    if not _nonempty_dir(extracted_asq):
        print("[entrypoint] extracted/asq missing — running extract_texts.py", flush=True)
        try:
            subprocess.run([sys.executable, "extract_texts.py"], cwd=ROOT, check=True)
        except Exception as exc:
            print(f"[entrypoint] extract_texts.py failed: {exc}", flush=True)
    else:
        print("[entrypoint] extracted/asq present", flush=True)

    en_dir = ROOT / "data" / "en"
    if not _nonempty_dir(en_dir):
        print("[entrypoint] data/en missing — running assistant.translate", flush=True)
        try:
            subprocess.run([sys.executable, "-m", "assistant.translate"], cwd=ROOT, check=True)
        except Exception as exc:
            print(f"[entrypoint] translate failed: {exc}", flush=True)
    else:
        print("[entrypoint] data/en present", flush=True)

    rag_docs = ROOT / "data" / "knowledge" / "rag_index" / "docs.json"
    chunks = ROOT / "data" / "knowledge" / "chunks.json"
    if not rag_docs.exists() and chunks.exists():
        print("[entrypoint] building medical RAG index", flush=True)
        try:
            from assistant.agent.orchestrator import ParentAssistant

            n = ParentAssistant().refresh_medical_index()
            print(f"[entrypoint] indexed {n} medical chunks", flush=True)
        except Exception as exc:
            print(f"[entrypoint] RAG index skipped: {exc}", flush=True)
    elif rag_docs.exists():
        print("[entrypoint] medical RAG index present", flush=True)
    else:
        print("[entrypoint] WARNING: no chunks.json — medical RAG empty", flush=True)

    argv = sys.argv[1:]
    if not argv:
        argv = ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
    print(f"[entrypoint] starting: {' '.join(argv)}", flush=True)
    os.execvp(argv[0], argv)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
