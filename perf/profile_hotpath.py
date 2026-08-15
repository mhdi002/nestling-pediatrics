"""cProfile the hot paths so bottleneck claims come with function-level numbers.

    python -m perf.profile_hotpath --target chat --iterations 20
    python -m perf.profile_hotpath --target chart --iterations 20
    python -m perf.profile_hotpath --target bm25 --iterations 50

Writes a ``.prof`` file (open with snakeviz/pstats) and prints the top
cumulative-time functions.
"""

from __future__ import annotations

import argparse
import cProfile
import io
import os
import pstats
import shutil
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("NESTLING_LOAD_MODELS", "0")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perf import config as cfg  # noqa: E402

QUESTIONS = [
    "My baby has a fever of 38.5, what should I do?",
    "When should my 9 month old start finger foods?",
    "How much milk does a 6 month old need per day?",
    "My toddler is not walking at 15 months, should I worry?",
]


def _chat_workload(iterations: int):
    from app.services import create_services

    tmp = Path(tempfile.mkdtemp(prefix="nestling_prof_"))
    svc = create_services(child_db_path=tmp / "child.db", chat_db_path=tmp / "chat.db")
    child_id = svc.db.create_child("prof", "male", gestational_age_weeks=39.0)
    session_id = svc.chat.create_session(child_id=child_id)
    svc.assistant.chat(session_id, QUESTIONS[0], child_id=child_id, ui_lang="en")

    def run():
        for i in range(iterations):
            svc.assistant.chat(
                session_id, QUESTIONS[i % len(QUESTIONS)], child_id=child_id, ui_lang="en"
            )

    def cleanup():
        svc.close()
        shutil.rmtree(tmp, ignore_errors=True)

    return run, cleanup


def _chart_workload(iterations: int):
    from assistant.tools.clinical import overlay_growth_on_chart

    overlay_growth_on_chart(
        sex="male", measure="weight", value=7.0, age_months=8.0,
        gestational_age_weeks=39.0, child_id="profwarm",
    )

    def run():
        for i in range(iterations):
            overlay_growth_on_chart(
                sex="male",
                measure="weight",
                value=7.0 + i * 0.01,
                age_months=8.0,
                gestational_age_weeks=39.0,
                child_id="profchart",
            )

    return run, lambda: None


def _bm25_workload(iterations: int):
    from assistant.rag.stores import MedicalRAG

    rag = MedicalRAG()
    rag.load()
    rag.retrieve(QUESTIONS[0], top_k=5)

    def run():
        for i in range(iterations):
            rag.retrieve(QUESTIONS[i % len(QUESTIONS)], top_k=5)

    return run, lambda: None


WORKLOADS = {"chat": _chat_workload, "chart": _chart_workload, "bm25": _bm25_workload}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Profile a Nestling hot path")
    parser.add_argument("--target", choices=sorted(WORKLOADS), default="chat")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--top", type=int, default=30)
    parser.add_argument("--sort", default="cumulative", choices=["cumulative", "tottime"])
    args = parser.parse_args(argv)

    run, cleanup = WORKLOADS[args.target](args.iterations)
    profiler = cProfile.Profile()
    profiler.enable()
    try:
        run()
    finally:
        profiler.disable()
        cleanup()

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    prof_path = cfg.RESULTS_DIR / f"profile_{args.target}_{stamp}.prof"
    profiler.dump_stats(str(prof_path))

    buf = io.StringIO()
    stats = pstats.Stats(profiler, stream=buf).sort_stats(args.sort)
    stats.print_stats(args.top)
    report = buf.getvalue()
    print(report)

    txt_path = prof_path.with_suffix(".txt")
    txt_path.write_text(
        f"target={args.target} iterations={args.iterations} sort={args.sort}\n\n{report}",
        encoding="utf-8",
    )
    print(f"Wrote {prof_path}\nWrote {txt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
