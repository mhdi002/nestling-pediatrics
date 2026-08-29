#!/usr/bin/env python3
"""
Fixed-concurrency smoke test: fire N simultaneous requests at each endpoint
class and report success rate and latency spread.

This is deliberately not Locust: it answers one narrow question -- "does the
service stay up and error-free under N truly simultaneous requests?" -- with
no ramp, no think time, and no dependencies beyond the stdlib.

Endpoint classes are tested separately because they fail for different
reasons: static reads are cheap, writes contend on the SQLite lock, chart
rendering is CPU-bound, and chat serialises behind the LLM.

Usage: concurrency_check.py [BASE_URL] [CONCURRENCY]
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8080").rstrip("/")
N = int(sys.argv[2]) if len(sys.argv) > 2 else 100
API_KEY = os.environ.get("NESTLING_API_KEY", "")


def _request(method: str, path: str, body: dict | None, timeout: float):
    url = f"{BASE}{path}"
    data = json.dumps(body).encode() if body is not None else None
    req = urllib.request.Request(url, data=data, method=method)
    req.add_header("Accept", "application/json")
    if data is not None:
        req.add_header("Content-Type", "application/json")
    if API_KEY:
        req.add_header("X-API-Key", API_KEY)
    started = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            resp.read()
            return resp.status, (time.perf_counter() - started) * 1000, ""
    except urllib.error.HTTPError as exc:
        # An HTTP error is a *served* response: the app stayed up.
        try:
            detail = exc.read()[:120].decode("utf-8", "replace")
        except Exception:
            detail = ""
        return exc.code, (time.perf_counter() - started) * 1000, detail
    except Exception as exc:  # connection refused, reset, timeout
        return 0, (time.perf_counter() - started) * 1000, f"{type(exc).__name__}: {exc}"


def run(label: str, method: str, path, body=None, timeout: float = 60.0, n: int = N):
    """Fire n requests simultaneously. `path`/`body` may be callables of the index."""
    def one(i: int):
        p = path(i) if callable(path) else path
        b = body(i) if callable(body) else body
        return _request(method, p, b, timeout)

    started = time.perf_counter()
    with ThreadPoolExecutor(max_workers=n) as pool:
        results = list(pool.map(one, range(n)))
    wall = time.perf_counter() - started

    codes: dict[int, int] = {}
    for code, _, _ in results:
        codes[code] = codes.get(code, 0) + 1
    lat = sorted(r[1] for r in results)
    ok = sum(c for code, c in codes.items() if 200 <= code < 300)
    transport_fail = codes.get(0, 0)
    errors = [d for code, _, d in results if (code == 0 or code >= 500) and d][:2]

    def pct(p: float) -> float:
        if not lat:
            return 0.0
        return lat[min(int(len(lat) * p), len(lat) - 1)]

    print(f"\n--- {label} (n={n}) ---")
    print(f"  wall clock : {wall:.2f}s   throughput: {n / wall:.1f} req/s")
    print(f"  success    : {ok}/{n} ({100.0 * ok / n:.1f}%)")
    print(f"  status codes: {dict(sorted(codes.items()))}")
    print(
        f"  latency ms : p50={pct(0.5):.0f} p95={pct(0.95):.0f} "
        f"p99={pct(0.99):.0f} max={max(lat):.0f}" if lat else "  latency: n/a"
    )
    if transport_fail:
        print(f"  !! {transport_fail} TRANSPORT failures (connection refused/reset/timeout)")
    for e in errors:
        print(f"  !! sample error: {e}")
    return {"label": label, "n": n, "ok": ok, "codes": codes, "transport_fail": transport_fail}


def main() -> int:
    print(f"target={BASE} concurrency={N} auth={'yes' if API_KEY else 'no'}")

    summary = []
    # 1. Liveness endpoint: pure read, no DB, no LLM.
    summary.append(run("GET /api/health", "GET", "/api/health", timeout=30))
    # 2. Static asset through nginx -- the canary from docs/PERFORMANCE.md.
    summary.append(run("GET / (static SPA)", "GET", "/", timeout=30))
    # 3. Concurrent writes: the SQLite write-lock contention case.
    summary.append(
        run(
            "POST /api/children (writes)",
            "POST",
            "/api/children",
            body=lambda i: {"name": f"LoadBaby{i}", "sex": "male", "gestational_age_weeks": 32},
            timeout=60,
        )
    )
    # 4. Read-back under concurrency.
    summary.append(run("GET /api/children (reads)", "GET", "/api/children", timeout=60))
    # 5. CPU-bound chart maths.
    summary.append(
        run(
            "GET /api/growth/curves (CPU)",
            "GET",
            "/api/growth/curves?sex=male&measure=weight&standard=who",
            timeout=90,
        )
    )

    total = sum(s["n"] for s in summary)
    total_ok = sum(s["ok"] for s in summary)
    total_transport = sum(s["transport_fail"] for s in summary)
    print("\n================ SUMMARY ================")
    print(f"  total requests : {total}")
    print(f"  succeeded      : {total_ok} ({100.0 * total_ok / total:.1f}%)")
    print(f"  transport fails: {total_transport}")
    print(f"  service up     : {'YES' if total_transport == 0 else 'NO -- refused/reset'}")
    print("=========================================")
    return 0 if total_transport == 0 and total_ok == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
