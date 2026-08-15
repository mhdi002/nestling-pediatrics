# Nestling performance harness

Load, ramp, micro-benchmark and profiling tools for the Nestling API.
Findings and the scaling plan live in [`../docs/PERFORMANCE.md`](../docs/PERFORMANCE.md).

Nothing in here is imported by the app, and `requirements-dev.txt` is not
installed into the runtime image.

## Install

```powershell
python -m pip install -r requirements-dev.txt
```

## Start the app under test

Measure the **app tier** first — keep models off and leave the LLM sidecar down,
otherwise every chat turn just measures GPU inference:

```powershell
$env:NESTLING_LOAD_MODELS = "0"
$env:NESTLING_USE_LLM     = "0"     # extractive RAG fallback, no sidecar
$env:NESTLING_USE_DENSE   = "0"     # skip bge-m3 download/embedding
$env:MPLBACKEND           = "Agg"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8080
```

Port 8080 is used because host port 8000 is often occupied on this machine.
To measure the LLM path instead, set `NESTLING_USE_LLM=1` and
`NESTLING_LLM_URL=http://127.0.0.1:8001` with the sidecar up, and expect a
completely different (much lower) ceiling — see `docs/PERFORMANCE.md`.

## Run a ramp

```powershell
python -m perf.run_ramp --host http://127.0.0.1:8080 --stages 50,200,500,1000 --duration 90s
```

Per stage this writes to `perf/results/<run-id>/`:

| File | Contents |
|---|---|
| `summary.md` | Ramp table + slowest endpoints per stage |
| `summary.json` | Same data, machine-readable |
| `stage-NNNN_stats.csv` | Locust per-endpoint stats (p50/p95/p99) |
| `stage-NNNN_failures.csv` | Failure kinds and counts |
| `stage-NNNN-resources.csv` | CPU/RSS samples for **server** and **locust** |
| `stage-NNNN-locust.log` | Raw Locust output |

The `limited_by` column is derived by comparing the server and Locust CPU series
plus failure kinds, so an exhausted test client is not mistaken for an app limit.

### Single stage, straight Locust

```powershell
locust -f perf/locustfile.py --headless -u 200 -r 20 -t 2m `
       --host http://127.0.0.1:8080 --csv perf/results/stage-200
```

Drop `--headless` for the web UI on <http://localhost:8089>.

### Going past the single-process client limit

One Locust process is CPU-bound well before 1000 users on a dev box. Use
worker processes on the same machine:

```powershell
python -m perf.run_ramp --stages 1000 --duration 90s --workers 8
# or directly:
locust -f perf/locustfile.py --headless -u 1000 -r 50 -t 2m --processes 8 --host http://127.0.0.1:8080
```

For a genuine 1000-user test, drive it from a second machine — a Windows dev box
sharing CPU between client and server cannot produce a trustworthy number.

### Reduce the spawn-time write burst

By default every simulated parent creates its own child row. To reuse a pool:

```powershell
python -m perf.seed --host http://127.0.0.1:8080 --count 250
```

The locustfile picks up `perf/results/seed_children.json` automatically.

## Micro-benchmarks (bottleneck isolation)

```powershell
python -m perf.micro_bench --threads 16 --iterations 40
python -m perf.micro_bench --only chart,sqlite --threads 32
```

Measures, in-process and without HTTP:

- SQLite journal mode / `synchronous` / `busy_timeout` as actually configured
- write throughput for shared-connection vs per-thread+WAL, under N threads
- matplotlib chart cost per render, serial vs concurrent (thread scaling)
- BM25 query cost and index build cost
- full `assistant.chat()` turn cost and RSS growth
- module-level caches with no eviction

## Profiling

```powershell
python -m perf.profile_hotpath --target chat  --iterations 20
python -m perf.profile_hotpath --target chart --iterations 20
python -m perf.profile_hotpath --target bm25  --iterations 50
```

Or sample the live server without restarting it:

```powershell
py-spy record -o perf/results/flame.svg --pid <uvicorn-pid> --duration 30
py-spy dump --pid <uvicorn-pid>          # what every thread is doing right now
```

## Configuration

Every knob is an environment variable; CLI flags override where offered.

| Variable | Default | Meaning |
|---|---|---|
| `PERF_HOST` | `http://127.0.0.1:8080` | Target base URL |
| `PERF_API_KEY` | *(empty)* | Sent as `X-API-Key` when `NESTLING_API_KEY` is set |
| `PERF_API_PREFIX` | `/api` | API mount prefix |
| `PERF_USERS` | `50` | Default user count for a single run |
| `PERF_SPAWN_RATE` | `10` | Users spawned per second |
| `PERF_DURATION` | `60s` | Per-stage duration |
| `PERF_STAGES` | `50,200,500,1000` | Ramp stages |
| `PERF_THINK_MIN` / `PERF_THINK_MAX` | `3` / `12` | Think time between requests (s) |
| `PERF_TIMEOUT` | `60` | Per-request timeout (s) |
| `PERF_STREAM_TIMEOUT` | `120` | SSE request timeout (s) |
| `PERF_W_CHAT` | `30` | Weight: `POST /api/chat` |
| `PERF_W_CHAT_STREAM` | `20` | Weight: `POST /api/chat/stream` (SSE) |
| `PERF_W_GROWTH` | `12` | Weight: growth submit + chart render |
| `PERF_W_GROWTH_CURVES` | `8` | Weight: JSON percentile curves |
| `PERF_W_DOSSIER` | `10` | Weight: child dossier read |
| `PERF_W_CHILD_LIST` | `5` | Weight: child list |
| `PERF_W_SCREENING` | `8` | Weight: ASQ / M-CHAT scoring |
| `PERF_W_SESSION_HISTORY` | `5` | Weight: session history read |
| `PERF_W_STATIC` | `10` | Weight: SPA static asset |
| `PERF_W_HEALTH` | `4` | Weight: health check |
| `PERF_SEED_FILE` | `results/seed_children.json` | Child-id pool file |
| `PERF_SESSION_PER_USER` | `1` | Distinct chat session per simulated user |
| `PERF_MONITOR_INTERVAL` | `1.0` | Resource sampling interval (s) |
| `PERF_SERVER_PROC_MATCH` | `uvicorn` | Server process selector: command-line substring, or `pid:1234` to pin one process tree exactly (use this when another uvicorn is running on the machine) |
| `PERF_RESULTS_DIR` | `perf/results` | Output directory |
| `PERF_LOCUST_WORKERS` | `1` | Locust worker processes |
| `PERF_BENCH_THREADS` | `16` | Micro-benchmark thread count |
| `PERF_BENCH_ITER` | `40` | Micro-benchmark iterations |

## Notes / gotchas

- Results under `perf/results/` are gitignored; the harness itself is committed.
- The load test writes real rows into `data/children/*.db` and real PNGs into
  `data/overlays/`. Point `NESTLING_CHILD_DB` / `NESTLING_CHAT_DB` at a scratch
  path, or delete the generated rows afterwards, before trusting a later run.
- Windows caps outbound ephemeral ports (~16k by default) and they linger in
  `TIME_WAIT`. At high user counts client-side `WSAEADDRINUSE (10048)` /
  `WSAENOBUFS (10055)` errors mean the *test machine* ran out, not the app.
  `netstat -an | Select-String TIME_WAIT | Measure-Object` confirms it.
