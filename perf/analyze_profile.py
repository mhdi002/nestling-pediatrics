"""Summarise a py-spy speedscope profile into self/total time per function.

    py-spy record --pid <server> --duration 30 --format speedscope -o perf/results/p.json
    python -m perf.analyze_profile perf/results/p.json --top 25

Self time answers "where is CPU actually burned", which a flamegraph makes you
eyeball. Per-thread totals show whether one thread is monopolising the GIL.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path


def summarise(path: Path, top: int) -> dict:
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data["shared"]["frames"]

    def label(idx: int) -> str:
        f = frames[idx]
        name = f.get("name") or "?"
        file = (f.get("file") or "").replace("\\", "/")
        # Keep the last two path segments: enough to tell app code from stdlib.
        short = "/".join(file.rsplit("/", 2)[-2:]) if file else ""
        return f"{name}  ({short}:{f.get('line', '?')})"

    self_time: dict[str, float] = defaultdict(float)
    total_time: dict[str, float] = defaultdict(float)
    per_thread: dict[str, float] = defaultdict(float)
    grand_total = 0.0

    for profile in data.get("profiles", []):
        thread = profile.get("name") or "?"
        samples = profile.get("samples") or []
        weights = profile.get("weights") or [1.0] * len(samples)
        for stack, weight in zip(samples, weights):
            if not stack:
                continue
            grand_total += weight
            per_thread[thread] += weight
            # py-spy emits leaf-last stacks.
            self_time[label(stack[-1])] += weight
            for idx in set(stack):
                total_time[label(idx)] += weight

    def table(counts: dict[str, float]) -> list[dict]:
        return [
            {
                "function": name,
                "seconds": round(val, 3),
                "percent": round(100.0 * val / grand_total, 2) if grand_total else 0.0,
            }
            for name, val in sorted(counts.items(), key=lambda kv: -kv[1])[:top]
        ]

    return {
        "file": str(path),
        "total_sampled_seconds": round(grand_total, 2),
        "threads_by_cpu": table(per_thread),
        "top_self_time": table(self_time),
        "top_total_time": table(total_time),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Summarise a py-spy speedscope profile")
    parser.add_argument("profile", type=Path)
    parser.add_argument("--top", type=int, default=25)
    args = parser.parse_args(argv)

    out = summarise(args.profile, args.top)
    print(f"Sampled CPU time: {out['total_sampled_seconds']}s  ({out['file']})\n")
    for section in ("threads_by_cpu", "top_self_time", "top_total_time"):
        print(f"--- {section} ---")
        for row in out[section]:
            print(f"  {row['percent']:6.2f}%  {row['seconds']:8.3f}s  {row['function']}")
        print()

    dest = args.profile.with_suffix(".summary.json")
    dest.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"Wrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
