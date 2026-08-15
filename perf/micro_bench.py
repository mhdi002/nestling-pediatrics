"""In-process micro-benchmarks that isolate the suspected bottlenecks.

    python -m perf.micro_bench --threads 16 --iterations 40

These bypass HTTP entirely so a slow number points at one component instead of
"the app is slow". Results go to stdout and to ``perf/results/micro_<ts>.json``.

Read-only with respect to the repo: it writes temp SQLite files and chart PNGs
under the system temp dir, never into ``data/``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
import statistics
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("NESTLING_LOAD_MODELS", "0")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perf import config as cfg  # noqa: E402


def _pct(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def _stat_block(samples_ms: list[float]) -> dict:
    return {
        "n": len(samples_ms),
        "mean_ms": round(statistics.fmean(samples_ms), 2) if samples_ms else 0.0,
        "p50_ms": round(_pct(samples_ms, 0.50), 2),
        "p95_ms": round(_pct(samples_ms, 0.95), 2),
        "p99_ms": round(_pct(samples_ms, 0.99), 2),
        "max_ms": round(max(samples_ms), 2) if samples_ms else 0.0,
    }


# --- 1. SQLite journal mode / pragma inspection -----------------------------


def bench_sqlite_pragmas() -> dict:
    """What journal/sync mode do the real ChatMemory / ChildMemoryDB use?"""
    from assistant.memory.chat_memory import ChatMemory
    from assistant.memory.child_db import ChildMemoryDB

    tmp = Path(tempfile.mkdtemp(prefix="nestling_perf_"))
    out = {}
    try:
        for label, ctor, name in (
            ("chat_memory", ChatMemory, "chat.db"),
            ("child_db", ChildMemoryDB, "child.db"),
        ):
            store = ctor(path=tmp / name)
            conn = store.conn
            out[label] = {
                "journal_mode": conn.execute("PRAGMA journal_mode").fetchone()[0],
                "synchronous": conn.execute("PRAGMA synchronous").fetchone()[0],
                "busy_timeout_ms": conn.execute("PRAGMA busy_timeout").fetchone()[0],
                "shared_connection_object": True,
                "check_same_thread": False,
                "sqlite_threadsafety": sqlite3.threadsafety,
            }
            store.close()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return out


# --- 2. SQLite write contention: shared connection vs per-thread + WAL ------


def _write_loop(
    conn: sqlite3.Connection | None,
    db_path: Path,
    n: int,
    samples: list[float],
    errors: list[str],
    lock: threading.Lock,
):
    """conn=None means 'open your own connection here', i.e. inside this thread."""
    own = conn is None
    if own:
        conn = sqlite3.connect(str(db_path), timeout=30.0)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    local: list[float] = []
    local_errors: list[str] = []
    try:
        for i in range(n):
            start = time.perf_counter()
            try:
                conn.execute(
                    "INSERT INTO messages(id, session_id, role, content, created_at) "
                    "VALUES(?,?,?,?,?)",
                    (f"{threading.get_ident()}-{i}", "s1", "user", "x" * 400, "now"),
                )
                conn.commit()
            except Exception as exc:
                local_errors.append(type(exc).__name__)
                continue
            local.append((time.perf_counter() - start) * 1000.0)
    finally:
        if own:
            conn.close()
    with lock:
        samples.extend(local)
        errors.extend(local_errors)


def _make_db(path: Path, wal: bool) -> None:
    conn = sqlite3.connect(str(path))
    if wal:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
    conn.executescript(
        "CREATE TABLE IF NOT EXISTS messages (id TEXT PRIMARY KEY, session_id TEXT, "
        "role TEXT, content TEXT, created_at TEXT);"
        "CREATE INDEX IF NOT EXISTS idx_m ON messages(session_id, created_at);"
    )
    conn.commit()
    conn.close()


def bench_sqlite_write_contention(threads: int, writes_per_thread: int) -> dict:
    """Compare today's shape (one shared conn, delete journal) against alternatives."""
    tmp = Path(tempfile.mkdtemp(prefix="nestling_perf_sql_"))
    results = {}
    try:
        scenarios = [
            ("shared_conn_delete_journal", False, True),
            ("shared_conn_wal", True, True),
            ("per_thread_conn_wal", True, False),
        ]
        for label, wal, shared in scenarios:
            db = tmp / f"{label}.db"
            _make_db(db, wal)
            samples: list[float] = []
            errors: list[str] = []
            lock = threading.Lock()
            shared_conn: sqlite3.Connection | None = None
            if shared:
                # Mirrors the app: one connection object created once, reused by
                # every threadpool worker via check_same_thread=False.
                shared_conn = sqlite3.connect(str(db), check_same_thread=False)
            started = time.perf_counter()
            workers = [
                threading.Thread(
                    target=_write_loop,
                    args=(shared_conn, db, writes_per_thread, samples, errors, lock),
                )
                for _ in range(threads)
            ]
            for w in workers:
                w.start()
            for w in workers:
                w.join()
            wall = time.perf_counter() - started
            if shared_conn is not None:
                shared_conn.close()
            total = threads * writes_per_thread
            ok = len(samples)
            error_kinds: dict[str, int] = {}
            for kind in errors:
                error_kinds[kind] = error_kinds.get(kind, 0) + 1
            results[label] = {
                **_stat_block(samples),
                "threads": threads,
                "writes_attempted": total,
                "writes_succeeded": ok,
                "writes_failed": len(errors),
                "error_kinds": error_kinds,
                "wall_seconds": round(wall, 3),
                "successful_writes_per_second": round(ok / wall, 1) if wall else 0.0,
            }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return results


# --- 3. matplotlib chart rendering cost ------------------------------------


def bench_chart_render(iterations: int, threads: int) -> dict:
    """Per-chart wall + CPU cost, serial and concurrent, via the real code path."""
    from assistant.tools.clinical import overlay_growth_on_chart

    def one(i: int) -> float:
        start = time.perf_counter()
        out = overlay_growth_on_chart(
            sex="male",
            measure="weight",
            value=7.0 + (i % 40) * 0.05,
            age_months=8.0,
            gestational_age_weeks=39.0,
            child_id=f"perfbench{i % max(1, threads)}",
        )
        elapsed = (time.perf_counter() - start) * 1000.0
        if out.get("plot_error"):
            raise RuntimeError(out["plot_error"])
        return elapsed

    one(0)  # warm the matplotlib font cache so it is not counted

    cpu_start = time.process_time()
    serial = [one(i) for i in range(iterations)]
    serial_cpu = time.process_time() - cpu_start

    cpu_start = time.process_time()
    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=threads) as pool:
        concurrent = list(pool.map(one, range(iterations)))
    concurrent_wall = time.perf_counter() - started
    concurrent_cpu = time.process_time() - cpu_start

    return {
        "serial": {
            **_stat_block(serial),
            "cpu_ms_per_chart": round(serial_cpu * 1000.0 / max(1, iterations), 2),
            "charts_per_second_1_core": round(iterations / (sum(serial) / 1000.0), 2)
            if serial
            else 0.0,
        },
        "concurrent": {
            **_stat_block(concurrent),
            "threads": threads,
            "wall_seconds": round(concurrent_wall, 3),
            "charts_per_second": round(iterations / concurrent_wall, 2)
            if concurrent_wall
            else 0.0,
            "cpu_ms_per_chart": round(concurrent_cpu * 1000.0 / max(1, iterations), 2),
        },
        "speedup_from_threads": round(
            (sum(serial) / 1000.0) / concurrent_wall, 2
        )
        if concurrent_wall
        else 0.0,
    }


# --- 4. BM25 retrieval cost ------------------------------------------------


def bench_bm25(iterations: int, threads: int) -> dict:
    from assistant.rag.stores import MedicalRAG

    rag = MedicalRAG()
    loaded = rag.load()
    doc_count = len(rag.store.docs)
    queries = [
        "fever in a 9 month old baby",
        "how much milk does a 6 month old need",
        "toddler not walking at 15 months",
        "signs of dehydration in infants",
        "when to introduce solid foods",
    ]

    def one(i: int) -> float:
        start = time.perf_counter()
        rag.retrieve(queries[i % len(queries)], top_k=5)
        return (time.perf_counter() - start) * 1000.0

    if not loaded or doc_count == 0:
        return {"loaded": loaded, "doc_count": doc_count, "note": "index empty — skipped"}

    one(0)
    serial = [one(i) for i in range(iterations)]

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=threads) as pool:
        concurrent = list(pool.map(one, range(iterations)))
    concurrent_wall = time.perf_counter() - started

    # Index build cost tells us how bad a per-request rebuild would be.
    build_start = time.perf_counter()
    rag.store._rebuild()
    rebuild_ms = (time.perf_counter() - build_start) * 1000.0

    return {
        "loaded": loaded,
        "doc_count": doc_count,
        "index_rebuild_ms": round(rebuild_ms, 2),
        "index_built_once_per_process": True,
        "serial": {**_stat_block(serial), "queries_per_second_1_core": round(
            iterations / (sum(serial) / 1000.0), 1) if serial else 0.0},
        "concurrent": {
            **_stat_block(concurrent),
            "threads": threads,
            "queries_per_second": round(iterations / concurrent_wall, 1)
            if concurrent_wall
            else 0.0,
        },
        "speedup_from_threads": round((sum(serial) / 1000.0) / concurrent_wall, 2)
        if concurrent_wall
        else 0.0,
    }


# --- 5. Child RAG reindex: does per-write cost grow with total children? ----


def bench_child_reindex(children_counts: list[int], docs_per_child: int) -> dict:
    """`refresh_child_index()` runs on every growth submit and every screening.

    It rebuilds the BM25 index over the whole shared child store and rewrites
    docs.json, so we measure the cost of ONE child's reindex as the number of
    other children in the process grows.
    """
    from assistant.rag.stores import ChildRAG

    tmp = Path(tempfile.mkdtemp(prefix="nestling_perf_rag_"))
    out: dict = {"docs_per_child": docs_per_child, "scaling": []}
    try:
        rag = ChildRAG()
        rag.store.index_dir = tmp  # keep docs.json out of the repo's data dir
        rag.store.docs = []
        target = "child-target"

        def docs_for(cid: str) -> list[dict]:
            return [
                {
                    "id": f"{cid}_doc{i}",
                    "collection": "child",
                    "child_id": cid,
                    "title": f"Growth weight @ {i}w",
                    "text": (
                        f"Measured weight={3 + i * 0.1} at {30 + i} postmenstrual weeks; "
                        f"z=0.4, centile=61, status=on_track for child {cid}"
                    ),
                }
                for i in range(docs_per_child)
            ]

        rag.reindex_child(docs_for(target))
        added = 0
        for count in sorted(children_counts):
            while added < count:
                rag.store.docs.extend(docs_for(f"child-other-{added}"))
                added += 1
            rag.store._rebuild()

            samples = []
            for _ in range(5):
                start = time.perf_counter()
                rag.reindex_child(docs_for(target))
                samples.append((time.perf_counter() - start) * 1000.0)

            search_start = time.perf_counter()
            rag.retrieve("weight centile on track", target, top_k=5)
            search_ms = (time.perf_counter() - search_start) * 1000.0

            docs_path = tmp / "docs.json"
            out["scaling"].append(
                {
                    "other_children_in_store": count,
                    "total_docs_in_store": len(rag.store.docs),
                    "reindex_ms_median": round(statistics.median(samples), 2),
                    "reindex_ms_max": round(max(samples), 2),
                    "retrieve_ms": round(search_ms, 2),
                    "docs_json_kb": round(docs_path.stat().st_size / 1024, 1)
                    if docs_path.exists()
                    else 0.0,
                }
            )
        first, last = out["scaling"][0], out["scaling"][-1]
        if first["reindex_ms_median"]:
            out["reindex_slowdown"] = round(
                last["reindex_ms_median"] / first["reindex_ms_median"], 1
            )
        out["thread_safe"] = False
        out["note"] = (
            "reindex_child() does read-modify-write on a shared list, rebuilds BM25 "
            "over every child's docs, and rewrites docs.json — with no lock."
        )
        return out
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --- 6. Full chat turn cost + memory growth --------------------------------


def bench_chat_turn(iterations: int, threads: int) -> dict:
    """End-to-end assistant.chat() without HTTP, plus RSS growth over the run."""
    import psutil

    from app.services import create_services

    tmp = Path(tempfile.mkdtemp(prefix="nestling_perf_chat_"))
    proc = psutil.Process()
    try:
        svc = create_services(child_db_path=tmp / "child.db", chat_db_path=tmp / "chat.db")
        child_id = svc.db.create_child("bench", "male", gestational_age_weeks=39.0)
        questions = [
            "My baby has a fever of 38.5, what should I do?",
            "When should my 9 month old start finger foods?",
            "How much milk does a 6 month old need?",
            "My toddler is not walking at 15 months, should I worry?",
        ]
        sessions = [svc.chat.create_session(child_id=child_id) for _ in range(max(1, threads))]

        def one(i: int) -> tuple[float, str | None]:
            """Returns (elapsed_ms, error_kind). Errors are data, not a crash:
            a failing concurrent turn is exactly what we are trying to measure."""
            start = time.perf_counter()
            try:
                svc.assistant.chat(
                    sessions[i % len(sessions)],
                    questions[i % len(questions)],
                    child_id=child_id,
                    ui_lang="en",
                )
            except Exception as exc:
                return (time.perf_counter() - start) * 1000.0, f"{type(exc).__name__}: {exc}"
            return (time.perf_counter() - start) * 1000.0, None

        one(0)
        rss_start = proc.memory_info().rss / 1024 / 1024

        def split(results: list[tuple[float, str | None]]) -> tuple[list[float], dict[str, int]]:
            oks = [ms for ms, err in results if err is None]
            kinds: dict[str, int] = {}
            for _, err in results:
                if err:
                    kinds[err[:120]] = kinds.get(err[:120], 0) + 1
            return oks, kinds

        cpu_start = time.process_time()
        serial_raw = [one(i) for i in range(iterations)]
        serial_cpu = time.process_time() - cpu_start
        serial, serial_errors = split(serial_raw)

        started = time.perf_counter()
        with ThreadPoolExecutor(max_workers=threads) as pool:
            concurrent_raw = list(pool.map(one, range(iterations)))
        concurrent_wall = time.perf_counter() - started
        concurrent, concurrent_errors = split(concurrent_raw)
        rss_end = proc.memory_info().rss / 1024 / 1024

        svc.close()
        serial_total_s = sum(serial) / 1000.0
        return {
            "serial": {
                **_stat_block(serial),
                "attempted": iterations,
                "failed": iterations - len(serial),
                "error_kinds": serial_errors,
                "cpu_ms_per_turn": round(serial_cpu * 1000.0 / max(1, iterations), 2),
                "turns_per_second_1_core": round(len(serial) / serial_total_s, 2)
                if serial_total_s
                else 0.0,
            },
            "concurrent": {
                **_stat_block(concurrent),
                "threads": threads,
                "attempted": iterations,
                "failed": iterations - len(concurrent),
                "error_rate_pct": round(100.0 * (iterations - len(concurrent)) / iterations, 1)
                if iterations
                else 0.0,
                "error_kinds": concurrent_errors,
                "successful_turns_per_second": round(len(concurrent) / concurrent_wall, 2)
                if concurrent_wall
                else 0.0,
            },
            "speedup_from_threads": round(serial_total_s / concurrent_wall, 2)
            if concurrent_wall
            else 0.0,
            "rss_mb_start": round(rss_start, 1),
            "rss_mb_end": round(rss_end, 1),
            "rss_growth_mb_over_turns": round(rss_end - rss_start, 1),
        }
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# --- 7. Unbounded cache inspection ----------------------------------------


def bench_cache_growth() -> dict:
    """Look for module-level caches and report whether they can actually evict."""
    from assistant.rag import dense
    from assistant.settings import get_settings

    has_lock = hasattr(dense, "_cache_lock")
    max_size = getattr(get_settings(), "nestling_dense_cache_size", None)
    bounded = max_size is not None and type(dense._cache).__name__ == "OrderedDict" and has_lock
    return {
        "dense_embedding_cache": {
            "module": "assistant.rag.dense._cache",
            "type": type(dense._cache).__name__,
            "entries_now": len(dense._cache),
            "max_entries": max_size,
            "thread_locked": has_lock,
            "bounded": bounded,
            "worst_case_mb": round((max_size or 0) * 1024 * 4 / 1024 / 1024, 1),
            "note": (
                "LRU-bounded via NESTLING_DENSE_CACHE_SIZE; ~4 KB per 1024-dim "
                "float32 bge-m3 vector, per worker process"
                if bounded
                else "unbounded — grows with distinct query/doc text for process lifetime"
            ),
        }
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Nestling bottleneck micro-benchmarks")
    parser.add_argument("--threads", type=int, default=cfg.env_int("PERF_BENCH_THREADS", 16))
    parser.add_argument("--iterations", type=int, default=cfg.env_int("PERF_BENCH_ITER", 40))
    parser.add_argument(
        "--only",
        default="",
        help="comma-separated subset: pragmas,sqlite,chart,bm25,reindex,chat,cache",
    )
    parser.add_argument(
        "--reindex-children",
        default=cfg.env_str("PERF_BENCH_REINDEX_CHILDREN", "1,10,50,200,500"),
        help="child-count points for the reindex scaling curve",
    )
    args = parser.parse_args(argv)

    wanted = {s.strip() for s in args.only.split(",") if s.strip()} or {
        "pragmas", "sqlite", "chart", "bm25", "reindex", "chat", "cache",
    }
    out: dict = {
        "generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "threads": args.threads,
        "iterations": args.iterations,
        "python": sys.version.split()[0],
    }
    steps = [
        ("pragmas", "SQLite pragmas / connection shape", lambda: bench_sqlite_pragmas()),
        ("sqlite", "SQLite write contention", lambda: bench_sqlite_write_contention(
            args.threads, max(5, args.iterations // 4))),
        ("chart", "matplotlib chart render", lambda: bench_chart_render(
            args.iterations, args.threads)),
        ("bm25", "BM25 retrieval", lambda: bench_bm25(args.iterations, args.threads)),
        ("reindex", "child RAG reindex scaling", lambda: bench_child_reindex(
            [int(x) for x in args.reindex_children.split(",") if x.strip()],
            cfg.env_int("PERF_BENCH_DOCS_PER_CHILD", 12),
        )),
        ("chat", "full chat turn", lambda: bench_chat_turn(
            max(8, args.iterations // 2), args.threads)),
        ("cache", "unbounded caches", lambda: bench_cache_growth()),
    ]
    for key, label, fn in steps:
        if key not in wanted:
            continue
        print(f"\n--- {label} ---", flush=True)
        try:
            out[key] = fn()
        except Exception as exc:
            out[key] = {"error": f"{type(exc).__name__}: {exc}"}
        print(json.dumps(out[key], indent=2), flush=True)

    cfg.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    path = cfg.RESULTS_DIR / f"micro_{stamp}.json"
    path.write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"\nWrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
