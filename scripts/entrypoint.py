#!/usr/bin/env python3
"""Container bootstrap: seed knowledge volume, index RAG, then exec uvicorn."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("NESTLING_ROOT") or Path(__file__).resolve().parent.parent)
# Baked into the image by the Dockerfile; overridable so the bootstrap can be
# exercised outside a container.
SEED = Path(os.environ.get("NESTLING_SEED_DIR", "/opt/nestling-seed"))
DEFAULT_HOST = os.environ.get("NESTLING_HOST", "0.0.0.0")
DEFAULT_PORT = os.environ.get("NESTLING_PORT", "8000")


def _nonempty_dir(p: Path) -> bool:
    return p.is_dir() and any(p.iterdir())


def _feeding_count(path: Path) -> int:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"[entrypoint] could not read {path}: {exc}", flush=True)
        return 0
    if not isinstance(data, list):
        return 0
    return sum(1 for c in data if isinstance(c, dict) and "feeding" in str(c.get("id", "")).lower())


def _seed_knowledge() -> None:
    """Named volume can mount empty/stale over /app/data/knowledge — restore curated chunks."""
    knowledge = ROOT / "data" / "knowledge"
    chunks = knowledge / "chunks.json"
    seed_chunks = SEED / "knowledge" / "chunks.json"
    knowledge.mkdir(parents=True, exist_ok=True)

    if seed_chunks.exists():
        seed_feed = _feeding_count(seed_chunks)
        cur_feed = _feeding_count(chunks) if chunks.exists() else -1
        # Prefer image seed when volume chunks are missing feeding guidance.
        if not chunks.exists() or (seed_feed > 0 and cur_feed < seed_feed):
            print(
                f"[entrypoint] refreshing chunks.json from image seed "
                f"(volume feeding={cur_feed}, seed feeding={seed_feed})",
                flush=True,
            )
            shutil.copy2(seed_chunks, chunks)
        for item in (SEED / "knowledge").iterdir():
            if item.name == "chunks.json":
                continue
            dest = knowledge / item.name
            if dest.exists():
                continue
            if item.is_dir():
                shutil.copytree(item, dest)
            else:
                shutil.copy2(item, dest)
    elif not chunks.exists():
        print("[entrypoint] WARNING: no chunks.json and no image seed", flush=True)


def _ensure_medical_index() -> None:
    chunks = ROOT / "data" / "knowledge" / "chunks.json"
    rag_docs = ROOT / "data" / "knowledge" / "rag_index" / "docs.json"
    if not chunks.exists():
        print("[entrypoint] WARNING: no chunks.json — medical RAG empty", flush=True)
        return

    needs_rebuild = not rag_docs.exists()
    if not needs_rebuild:
        try:
            docs = json.loads(rag_docs.read_text(encoding="utf-8"))
            feed_docs = sum(1 for d in docs if "feeding" in str(d.get("id", "")).lower())
            feed_chunks = _feeding_count(chunks)
            if feed_chunks > 0 and feed_docs < feed_chunks:
                needs_rebuild = True
                print(
                    f"[entrypoint] RAG index stale "
                    f"(index feeding={feed_docs}, chunks feeding={feed_chunks})",
                    flush=True,
                )
            elif chunks.stat().st_mtime > rag_docs.stat().st_mtime + 1:
                needs_rebuild = True
                print("[entrypoint] chunks.json newer than RAG index — rebuilding", flush=True)
        except Exception as exc:
            needs_rebuild = True
            print(f"[entrypoint] RAG index check failed ({exc}) — rebuilding", flush=True)

    if needs_rebuild:
        print("[entrypoint] building medical RAG index", flush=True)
        try:
            from assistant.agent.orchestrator import ParentAssistant

            n = ParentAssistant().refresh_medical_index()
            print(f"[entrypoint] indexed {n} medical chunks", flush=True)
        except Exception as exc:
            print(f"[entrypoint] RAG index skipped: {exc}", flush=True)
    else:
        print("[entrypoint] medical RAG index present", flush=True)


def main() -> int:
    os.chdir(ROOT)
    # Ensure repo root is importable before indexing (docker entrypoint).
    root_s = str(ROOT)
    if root_s not in sys.path:
        sys.path.insert(0, root_s)
    os.environ.setdefault("PYTHONPATH", root_s)
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

    _ensure_medical_index()

    argv = sys.argv[1:]
    if not argv:
        argv = ["uvicorn", "app.main:app", "--host", DEFAULT_HOST, "--port", DEFAULT_PORT]
    print(f"[entrypoint] starting: {' '.join(argv)}", flush=True)
    os.execvp(argv[0], argv)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
