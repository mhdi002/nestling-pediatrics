#!/usr/bin/env python3
"""Container bootstrap: make sure the screening data and the care-library index exist, then exec uvicorn.

The care library (knowledge/) ships in the image and is read as files by the RLM
orchestrator; nothing is indexed or seeded into a volume."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(os.environ.get("NESTLING_ROOT") or Path(__file__).resolve().parent.parent)
# Baked into the image by the Dockerfile; overridable so the bootstrap can be
# exercised outside a container.
DEFAULT_HOST = os.environ.get("NESTLING_HOST", "0.0.0.0")
DEFAULT_PORT = os.environ.get("NESTLING_PORT", "8000")


def _nonempty_dir(p: Path) -> bool:
    return p.is_dir() and any(p.iterdir())


def _ensure_knowledge_workspace() -> None:
    """RLM mode: refresh knowledge/INDEX.md from a directory walk (no RAG chunks)."""
    use_rlm = (os.environ.get("NESTLING_USE_RLM") or "1").strip() not in {"0", "false", "False"}
    if not use_rlm:
        return
    try:
        from assistant.rlm.knowledge import knowledge_root, write_index

        root = knowledge_root()
        if not root.is_dir():
            print(f"[entrypoint] WARNING: knowledge dir missing at {root}", flush=True)
            return
        index = write_index(root)
        print(f"[entrypoint] knowledge index ready: {index}", flush=True)
    except Exception as exc:
        print(f"[entrypoint] knowledge index skipped: {exc}", flush=True)


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
    (ROOT / "data" / "rlm_state").mkdir(parents=True, exist_ok=True)

    # The screening questionnaires the tools read ship as data/en. Only when they
    # are missing are they rebuilt: extracted from the PDFs, then translated.
    en_dir = ROOT / "data" / "en"
    if not _nonempty_dir(en_dir):
        print("[entrypoint] data/en missing — extracting and translating the questionnaires", flush=True)
        for cmd in ([sys.executable, "extract_texts.py"], [sys.executable, "-m", "assistant.translate"]):
            try:
                subprocess.run(cmd, cwd=ROOT, check=True)
            except Exception as exc:
                print(f"[entrypoint] {' '.join(cmd[1:])} failed: {exc}", flush=True)
                break
    else:
        print("[entrypoint] data/en present", flush=True)

    _ensure_knowledge_workspace()

    argv = sys.argv[1:]
    if not argv:
        argv = ["uvicorn", "app.main:app", "--host", DEFAULT_HOST, "--port", DEFAULT_PORT]
    print(f"[entrypoint] starting: {' '.join(argv)}", flush=True)
    os.execvp(argv[0], argv)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
