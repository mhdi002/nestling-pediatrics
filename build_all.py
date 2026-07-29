#!/usr/bin/env python3
"""End-to-end build: re-extract → translate → index RAG → cleanup temps."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(cmd: list[str]):
    print(">", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(ROOT))
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def cleanup():
    patterns = [
        "_analyze_pdfs.py",
        "_pdf_analysis_summary.txt",
        "_pdf_list.txt",
        "_pdf_meta.json",
        "_asq_*",
        "_mchat_*",
        "_extract_*",
        "_dump_*",
        "_compare_*",
        "_finalize_*",
        "preview_*.png",
    ]
    removed = []
    for pat in patterns:
        for p in ROOT.glob(pat):
            if p.is_file():
                p.unlink()
                removed.append(p.name)
    print(f"Removed {len(removed)} temp files")


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    run([sys.executable, "extract_texts.py"])
    run([sys.executable, "verify_equations.py"])
    run([sys.executable, "-m", "assistant.translate"])
    # Build medical RAG index
    from assistant.agent.orchestrator import ParentAssistant

    asst = ParentAssistant()
    n = asst.refresh_medical_index()
    print(f"Medical RAG chunks indexed: {n}")
    cleanup()
    print("Build complete.")


if __name__ == "__main__":
    main()
