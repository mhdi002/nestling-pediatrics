"""Drive a multi-stage Locust ramp and collect per-stage results.

    python -m perf.run_ramp --host http://127.0.0.1:8080 --stages 50,200,500,1000 --duration 90s

For each stage this runs Locust headless, samples CPU/RSS for both the app
server and Locust itself, then writes ``summary.json`` plus a Markdown table to
``perf/results/<run-id>/``. Comparing the two CPU series is how we tell an app
limit apart from a test-machine limit.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from perf import config as cfg
from perf.monitor import ResourceMonitor


def _parse_stats_csv(path: Path) -> dict:
    """Read Locust's ``*_stats.csv`` aggregated row + per-endpoint rows."""
    if not path.is_file():
        return {}
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {}

    def num(row: dict, *keys: str) -> float:
        for key in keys:
            val = row.get(key)
            if val not in (None, "", "N/A"):
                try:
                    return float(val)
                except ValueError:
                    continue
        return 0.0

    def shape(row: dict) -> dict:
        reqs = num(row, "Request Count")
        fails = num(row, "Failure Count")
        return {
            "name": f"{row.get('Type', '').strip()} {row.get('Name', '').strip()}".strip(),
            "requests": int(reqs),
            "failures": int(fails),
            "error_rate_pct": round(100.0 * fails / reqs, 2) if reqs else 0.0,
            "rps": round(num(row, "Requests/s"), 2),
            "p50_ms": round(num(row, "50%", "Median Response Time"), 1),
            "p95_ms": round(num(row, "95%"), 1),
            "p99_ms": round(num(row, "99%"), 1),
            "max_ms": round(num(row, "Max Response Time"), 1),
            "avg_ms": round(num(row, "Average Response Time"), 1),
        }

    agg = next((r for r in rows if (r.get("Name") or "").strip() == "Aggregated"), None)
    endpoints = [shape(r) for r in rows if (r.get("Name") or "").strip() != "Aggregated"]
    return {
        "aggregated": shape(agg) if agg else {},
        "endpoints": sorted(endpoints, key=lambda e: -e["p95_ms"]),
    }


# Failure buckets. Distinguishing these matters: a NameError from a half-saved
# edit says nothing about capacity, whereas an sqlite3 InterfaceError under load
# is exactly the architectural problem we are hunting.
FAILURE_CLASSES: list[tuple[str, tuple[str, ...]]] = [
    (
        "sqlite_concurrency",
        (
            "bad parameter or other api misuse",
            "cannot commit - no transaction is active",
            "error return without exception set",
            "another row available",
            "not an error",
            "database is locked",
            "sqlite objects created in a thread",
        ),
    ),
    ("lost_write", ("unknown session_id", "unknown child_id")),
    (
        "client_socket_exhaustion",
        ("10048", "10055", "10054", "cannot assign requested address", "too many open files"),
    ),
    ("client_timeout", ("readtimeout", "connecttimeout", "timed out")),
    ("app_code_error", ("is not defined", "has no attribute", "keyerror", "typeerror")),
    ("http_4xx", ("http 400", "http 401", "http 404", "http 422")),
]


def classify_failure(text: str) -> str:
    low = (text or "").lower()
    for label, markers in FAILURE_CLASSES:
        if any(m in low for m in markers):
            return label
    return "other"


def _parse_failures_csv(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    with path.open(encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    out = []
    for row in rows:
        try:
            count = int(float(row.get("Occurrences") or 0))
        except ValueError:
            count = 0
        error = (row.get("Error") or "").strip()
        out.append(
            {
                "name": (row.get("Name") or "").strip(),
                "error": error[:200],
                "class": classify_failure(error),
                "occurrences": count,
            }
        )
    return sorted(out, key=lambda r: -r["occurrences"])


def _failure_classes(failures: list[dict]) -> dict[str, int]:
    totals: dict[str, int] = {}
    for f in failures:
        totals[f["class"]] = totals.get(f["class"], 0) + f["occurrences"]
    return dict(sorted(totals.items(), key=lambda kv: -kv[1]))


def _classify_limit(stage: dict) -> str:
    """App limit vs test-machine limit, from the two CPU series and failure kinds."""
    if not (stage.get("aggregated") or {}).get("requests"):
        return "inconclusive (no requests recorded — check the Locust log)"
    server_cpu = (stage.get("server") or {}).get("cpu_percent_p95") or 0.0
    locust_cpu = (stage.get("locust") or {}).get("cpu_percent_p95") or 0.0
    err = (stage.get("aggregated") or {}).get("error_rate_pct") or 0.0
    classes = stage.get("failure_classes") or {}
    total_failures = sum(classes.values()) or 1
    client_side = (
        classes.get("client_socket_exhaustion", 0) + classes.get("client_timeout", 0)
    ) / total_failures > 0.5
    if client_side and locust_cpu >= server_cpu:
        return "test-machine (client socket/CPU exhaustion)"
    if locust_cpu > 70.0 and locust_cpu > server_cpu * 1.2:
        return "test-machine (Locust CPU-bound)"
    if server_cpu > 70.0:
        return "app (server CPU saturated)"
    if err > 1.0 and client_side:
        return "test-machine (client errors dominate)"
    if server_cpu < 40.0 and locust_cpu < 40.0:
        return "app (serialisation/blocking — neither side CPU-bound)"
    return "app"


def run_stage(users: int, args, run_dir: Path) -> dict:
    prefix = run_dir / f"stage-{users:04d}"
    env = dict(os.environ)
    env["PERF_HOST"] = args.host
    if args.api_key:
        env["PERF_API_KEY"] = args.api_key
    spawn = args.spawn_rate or max(1.0, users / 10.0)

    base = [
        sys.executable,
        "-m",
        "locust",
        "-f",
        str(cfg.PERF_DIR / "locustfile.py"),
        "--headless",
        "--host",
        args.host,
    ]
    master = base + [
        "-u", str(users),
        "-r", str(spawn),
        "-t", args.duration,
        "--csv", str(prefix),
        "--csv-full-history",
        "--only-summary",
        "--reset-stats",
    ]
    # `--processes` is unsupported on Windows, so distributed mode is wired by
    # hand: one master plus N worker processes over the local master port.
    if args.workers > 1:
        master += ["--master", "--expect-workers", str(args.workers)]

    print(f"\n=== stage: {users} users (spawn {spawn}/s, {args.duration}) ===", flush=True)
    monitor = ResourceMonitor(
        {"server": args.server_match, "locust": "locust"}, interval=args.monitor_interval
    ).start()
    started = time.time()
    log_path = Path(f"{prefix}-locust.log")
    workers: list[subprocess.Popen] = []
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        if args.workers > 1:
            proc = subprocess.Popen(master, env=env, stdout=log, stderr=subprocess.STDOUT)
            worker_log = Path(f"{prefix}-locust-workers.log")
            with worker_log.open("w", encoding="utf-8", errors="replace") as wlog:
                for _ in range(args.workers):
                    workers.append(
                        subprocess.Popen(
                            base + ["--worker"], env=env, stdout=wlog, stderr=subprocess.STDOUT
                        )
                    )
                proc.wait()
            for w in workers:
                try:
                    w.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    w.kill()
        else:
            proc = subprocess.run(master, env=env, stdout=log, stderr=subprocess.STDOUT, check=False)
    elapsed = time.time() - started
    resources = monitor.stop()
    monitor.write_csv(Path(f"{prefix}-resources.csv"))

    stats = _parse_stats_csv(Path(f"{prefix}_stats.csv"))
    failures = _parse_failures_csv(Path(f"{prefix}_failures.csv"))
    stage = {
        "users": users,
        "spawn_rate": spawn,
        "duration_requested": args.duration,
        "wall_seconds": round(elapsed, 1),
        "locust_exit_code": proc.returncode,
        "aggregated": stats.get("aggregated", {}),
        "endpoints": stats.get("endpoints", []),
        "top_failures": failures[:15],
        "failure_classes": _failure_classes(failures),
        "server": resources.get("server", {}),
        "locust": resources.get("locust", {}),
        "log": log_path.name,
    }
    stage["limited_by"] = _classify_limit(stage)

    agg = stage["aggregated"]
    print(
        f"  rps={agg.get('rps')} p50={agg.get('p50_ms')}ms p95={agg.get('p95_ms')}ms "
        f"p99={agg.get('p99_ms')}ms err={agg.get('error_rate_pct')}% "
        f"server_cpu_p95={stage['server'].get('cpu_percent_p95')}% "
        f"locust_cpu_p95={stage['locust'].get('cpu_percent_p95')}% "
        f"limited_by={stage['limited_by']}",
        flush=True,
    )
    return stage


def _markdown(run: dict) -> str:
    lines = [
        f"# Nestling load-test run `{run['run_id']}`",
        "",
        f"- Host: `{run['host']}`",
        f"- Commit: `{run.get('git_commit', 'unknown')}`",
        f"- Started: {run['started_utc']}",
        f"- Test machine: {run['cpu_count']} logical CPUs",
        "",
        "## Per-stage summary",
        "",
        "| Users | RPS | p50 ms | p95 ms | p99 ms | Err % | Server CPU p95 % | Server RSS max MB | Locust CPU p95 % | Limited by |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for stage in run["stages"]:
        agg = stage.get("aggregated", {})
        srv = stage.get("server", {})
        loc = stage.get("locust", {})
        lines.append(
            f"| {stage['users']} | {agg.get('rps', 0)} | {agg.get('p50_ms', 0)} | "
            f"{agg.get('p95_ms', 0)} | {agg.get('p99_ms', 0)} | {agg.get('error_rate_pct', 0)} | "
            f"{srv.get('cpu_percent_p95', 0)} | {srv.get('rss_mb_max', 0)} | "
            f"{loc.get('cpu_percent_p95', 0)} | {stage.get('limited_by', '?')} |"
        )
    for stage in run["stages"]:
        lines += [
            "",
            f"### {stage['users']} users — slowest endpoints (by p95)",
            "",
            "| Endpoint | Reqs | RPS | p50 ms | p95 ms | p99 ms | Err % |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for ep in stage.get("endpoints", [])[:12]:
            lines.append(
                f"| `{ep['name']}` | {ep['requests']} | {ep['rps']} | {ep['p50_ms']} | "
                f"{ep['p95_ms']} | {ep['p99_ms']} | {ep['error_rate_pct']} |"
            )
        if stage.get("failure_classes"):
            lines += ["", "Failures by class:", ""]
            for label, count in stage["failure_classes"].items():
                lines.append(f"- **{label}**: {count}")
        if stage.get("top_failures"):
            lines += ["", "Top failures:", ""]
            for fail in stage["top_failures"][:8]:
                lines.append(
                    f"- `{fail['name']}` ×{fail['occurrences']} "
                    f"[{fail['class']}] — {fail['error']}"
                )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run a Locust ramp against Nestling")
    parser.add_argument("--host", default=cfg.HOST)
    parser.add_argument("--stages", default=cfg.STAGES, help="comma-separated user counts")
    parser.add_argument("--duration", default=cfg.DURATION, help="per-stage duration, e.g. 90s")
    parser.add_argument("--spawn-rate", type=float, default=None, help="default: users/10")
    parser.add_argument("--api-key", default=cfg.API_KEY)
    parser.add_argument(
        "--workers", type=int, default=cfg.env_int("PERF_LOCUST_WORKERS", 1),
        help="Locust worker processes (>1 uses distributed mode on this machine)",
    )
    parser.add_argument("--server-match", default=cfg.SERVER_PROC_MATCH)
    parser.add_argument("--monitor-interval", type=float, default=cfg.MONITOR_INTERVAL)
    parser.add_argument("--run-id", default=None)
    parser.add_argument("--cooldown", type=float, default=10.0, help="seconds between stages")
    args = parser.parse_args(argv)

    stages = [int(s) for s in str(args.stages).split(",") if s.strip()]
    run_id = args.run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = cfg.RESULTS_DIR / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    try:
        commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cfg.REPO_ROOT, capture_output=True, text=True, check=False,
        ).stdout.strip()
    except OSError:
        commit = "unknown"

    import psutil

    run = {
        "run_id": run_id,
        "host": args.host,
        "git_commit": commit or "unknown",
        "started_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "cpu_count": psutil.cpu_count(logical=True),
        "total_ram_mb": round(psutil.virtual_memory().total / 1024 / 1024),
        "stages": [],
        "config": {
            "think_time_s": [cfg.THINK_MIN, cfg.THINK_MAX],
            "duration": args.duration,
            "locust_workers": args.workers,
        },
    }

    for i, users in enumerate(stages):
        run["stages"].append(run_stage(users, args, run_dir))
        (run_dir / "summary.json").write_text(json.dumps(run, indent=2), encoding="utf-8")
        (run_dir / "summary.md").write_text(_markdown(run), encoding="utf-8")
        if i + 1 < len(stages) and args.cooldown > 0:
            time.sleep(args.cooldown)

    print(f"\nWrote {run_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
