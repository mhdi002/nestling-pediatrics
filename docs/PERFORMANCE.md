# Nestling — performance, load testing and scaling to 1000 users

**Status: the current stack does not serve 1000 concurrent users, and no amount
of load balancing fixes that on its own.** Four architectural properties cap the
app at roughly **one CPU core of useful work per process**, and the SQLite data
layer prevents running more than one process safely. This document records what
was measured, how to reproduce it, what has to change in application code, and
the deployment topology that becomes viable once those changes land.

Harness: [`../perf/README.md`](../perf/README.md). Raw results: `perf/results/`
(gitignored).

---

## 1. Test conditions (read before trusting any number)

| | |
|---|---|
| Commit | `eaaef83`, **working tree dirty** — `assistant/**` and `app/**` were being edited by another agent *during* these runs |
| App under test | `uvicorn app.main:app` — single process, single worker, no `--reload` |
| Env | `NESTLING_LOAD_MODELS=0`, `NESTLING_USE_LLM=0`, `NESTLING_USE_DENSE=0`, `MPLBACKEND=Agg` |
| LLM sidecar | **down** — chat used the extractive RAG fallback |
| Machine | Windows 11, 16 logical CPUs, 16 GB RAM, running the app **and** Locust **and** two unrelated `uvicorn` servers from other agents |
| Load generator | Locust 2.46.3, 1 master + 4 worker processes |
| Think time | 3–12 s uniform between requests |
| Data at start of ramp | 257 chart PNGs; grew to 873. `children.db` 86 KB → 2.3 MB, `chat.db` 958 KB → 6.2 MB |

Honest limitations:

- **This is a baseline, not a verdict on final code.** `assistant/memory/*.py`,
  `assistant/rag/stores.py` and `assistant/tools/clinical.py` changed several
  times mid-session. Some early errors (`name '_CARE_ID_PREFIXES' is not
  defined`) came from the server importing a half-saved module, not from load.
  Re-run after the concurrent edits settle.
- **The app tier was measured, not the LLM path.** Everything in §6 about the
  vLLM sidecar is arithmetic from its configured limits, clearly labelled as
  such, because the sidecar was not running.
- Absolute latencies are worse than a dedicated server would show, since the
  load generator shared 16 cores with the app. **This does not affect the
  conclusions**: in every stage the server used ≤ 6.6 % of the machine and
  Locust ≤ 1.5 %, so neither was CPU-starved. The limit is serialisation.

---

## 2. Ramp results — baseline (`perf/results/baseline_run2/`)

90 s per stage. Every simulated parent creates a child on start, so this
includes a registration burst; §3 isolates steady state.

| Users | RPS | p50 ms | p95 ms | p99 ms | Err % | Server CPU p95 % (of 16 cores) | Server RSS max MB | Locust CPU p95 % | Limited by |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 50 | 11.4 | 610 | 14 000 | 19 000 | 0.0 | 5.3 | 205 | 1.4 | app |
| 200 | 9.9 | 12 000 | 29 000 | 32 000 | 0.0 | 6.6 | 205 | 0.8 | app |
| 500 | 20.0 | 8 200 | 39 000 | 54 000 | 0.06 | 5.0 | 205 | 0.3 | app |
| 1000 | 30.6 | 23 000 | 34 000 | 41 000 | 0.0 | 3.6 | 205 | 0.5 | app |

**App limit, not test-machine limit.** Server CPU p95 peaked at 6.6 % of 16
logical cores — about **one core** — while 39 of 40 request threads sat idle.
Locust never exceeded 1.5 % CPU, there were no client socket errors
(`WSAEADDRINUSE`/`WSAENOBUFS`), and no ephemeral-port exhaustion. The test
machine had capacity to spare in every stage.

Read the table as: **throughput is pinned in a 10–31 RPS band no matter how many
users arrive, and the extra load is absorbed entirely as latency** (p50 0.6 s →
23 s). Errors stay near zero only because Locust waits up to 60 s; real users
would have abandoned the request.

The most diagnostic single row: at 1000 users, **`GET /` — a static file — had
p50 34 s**. Serving a small file off disk cannot take 34 seconds unless the
whole process is starved. Memory was flat at 205 MB throughout, so this is not a
leak.

### Which endpoints degrade first

At 50 users, while cheap reads were still sub-100 ms:

| Endpoint | p50 ms | p95 ms | p99 ms |
|---|---:|---:|---:|
| `POST /api/chat` | 7 200 | 18 000 | 23 000 |
| `POST /api/chat/stream` (time to first token) | 8 100 | 16 000 | 24 000 |
| `POST /api/growth` (chart render) | 1 400 | 5 300 | 6 400 |
| `GET /api/growth/curves` | 28 | 880 | 1 400 |
| `GET /api/asq/{age}/questions` | 41 | 1 200 | 2 200 |

Order of degradation: **`POST /api/chat` and `POST /api/chat/stream` first, then
`POST /api/growth`, then everything else uniformly.** By 500 users the
distinction is gone — `GET /api/health` sits at p50 15 s — which is the
signature of one shared choke point rather than N slow endpoints.

### SSE specifics

`POST /api/chat/stream` runs the **entire** chat turn before emitting the first
token (`app/api/routes.py::chat_stream` builds `out` first, then chunks
`out["reply"]`). So time-to-first-token ≈ full turn latency (8.1 s p50 at 50
users) and the "streaming" is cosmetic: it replays an already-complete string.
That matters for proxy configuration (§7) and it means SSE gives users no
latency benefit today.

---

## 3. Steady state without the registration burst

At 1000 users, `POST /api/children` was 912 of the recorded requests — the
spawn-time burst crowded out real traffic. Re-running against a pre-seeded pool
of 200 children (`python -m perf.seed --count 200`) removes it:
see `perf/results/steady_seeded/summary.md`.

---

## 4. Ranked bottlenecks, with measured evidence

Ranked by how hard each one caps a 1000-user deployment.

### #1 — Child RAG index is fully rebuilt and rewritten to disk on every write

`ChildRAG.reindex_child()` (`assistant/rag/stores.py`) is called from
`ParentAssistant.refresh_child_index()`, which runs on **every** growth submit,
**every** ASQ score, **every** M-CHAT score and every growth recorded inside a
chat turn. It does three O(total-children) things:

1. `remaining = [d for d in self.store.docs if d.get("child_id") != child_id]`
   — scans every document of every child in the process.
2. `store.add(...)` → `_rebuild()` → `BM25Index.fit(...)` — re-tokenises the
   **entire corpus of all children**.
3. `store.save()` → `json.dumps(all_docs, indent=2)` + `write_text(...)` —
   rewrites the whole index file synchronously, inside the request.

Measured (`python -m perf.micro_bench --only reindex`, 12 docs per child):

| Other children in store | Total docs | Reindex ms (median) | Child retrieval ms | `docs.json` size |
|---:|---:|---:|---:|---:|
| 1 | 24 | 0.84 | 2.6 | 6 KB |
| 10 | 132 | 1.91 | 0.7 | 34 KB |
| 50 | 612 | 6.28 | 2.3 | 159 KB |
| 200 | 2 412 | 24.9 | 8.9 | 630 KB |
| 500 | 6 012 | 59.2 | 21.8 | 1.5 MB |
| **1 000** | **12 012** | **131.2** | **45.9** | **3.1 MB** |

**156× slowdown from 1 to 1000 children, linear in total corpus size.** At the
target scale of 1000 children, one growth submit costs 131 ms of GIL-held pure
Python plus a 3.1 MB synchronous file write, and every child-scoped retrieval
costs 46 ms. At 10 000 children, extrapolate to ~1.3 s and ~31 MB per write.

It is also **not thread-safe**: the read-modify-write of `self.store.docs` has
no lock, and concurrent `write_text()` calls target the same `docs.json`, so two
parents saving measurements at the same moment can silently lose documents or
leave a truncated index.

`VectorStore.search()` compounds it: with `filters={"child_id": ...}` it scores
BM25 across **all** documents and filters afterwards, so one parent's retrieval
pays for every other parent's data.

### #2 — Every request runs in a thread pool behind a process-global DB lock

Every route in `app/api/routes.py` is a plain `def`, so FastAPI runs it in the
AnyIO worker thread pool (default 40 threads). Both data stores share **one**
`sqlite3` connection created with `check_same_thread=False`, and every
connection-touching method is wrapped in a `threading.RLock`
(`assistant/memory/chat_memory.py`, `assistant/memory/child_db.py`). Correct,
but it makes the database a strictly serial resource for the whole process.

`cProfile` over 40 chat turns (`python -m perf.profile_hotpath --target chat`,
1.161 s total, 29 ms/turn):

| Function | Self time | Share | Calls per turn |
|---|---:|---:|---:|
| `sqlite3.Connection.commit` | 0.690 s | **59.4 %** | **13.75** |
| `sqlite3.Connection.execute` | 0.188 s | 16.2 % | **34.75** |
| `BM25Index.scores` | 0.118 s (cum) | 10.2 % | 1 |
| `json.loads` | — | — | **206** |

**76 % of a chat turn is SQLite**, spread over 13.75 separate commits and 34.75
statements — all serialised process-wide. `py-spy` sampling of the live server
under load agrees: the top self-time entries were
`child_db.py:125` (`commit` in `create_child`, 8.0 %),
`child_db.py:213` (`commit` in `add_event`, 7.3 %), the two matching `execute`
calls (5.3 % and 4.4 %), and `child_db.py:30` — the lock acquire itself — at
1.0 %.

Consequence, measured in isolation with no HTTP involved
(`python -m perf.micro_bench --only chat`):

| Threads | p50 per turn | Throughput | Speed-up |
|---:|---:|---:|---:|
| 1 (serial) | 25 ms | 37.2 turns/s | 1.0× |
| 16 | 383 ms | 41.1 turns/s | 1.04× |
| 32 | 741 ms | 38.4 turns/s | **1.03×** |

**32 threads buy 3 % more throughput.** Latency grows linearly with concurrency
because every turn queues for the same lock. Note the CPU cost of a turn is only
2.5–3.9 ms — the other ~22 ms is commit I/O, so this is not a compute problem.

### #3 — SQLite is configured for the slowest possible writes

Measured via `python -m perf.micro_bench --only pragmas`:

| Setting | Both `chat.db` and `children.db` |
|---|---|
| `journal_mode` | **`delete`** (rollback journal created and deleted per transaction) |
| `synchronous` | **`2` = FULL** (fsync on every commit) |
| `busy_timeout` | 5000 ms |
| Connection | one object shared by all threads, `check_same_thread=False` |

16 threads × 10 writes (`--only sqlite`):

| Configuration | Successful writes/s | p50 | p95 | Failed writes |
|---|---:|---:|---:|---:|
| Shared connection, `journal_mode=delete` (**today**) | 1 102 | 9.3 ms | 22.7 ms | **35 / 160 (21.9 %)** |
| Shared connection, WAL + `synchronous=NORMAL` | 3 363 | 3.1 ms | 6.7 ms | 23 / 160 (14.4 %) |
| **Per-thread connections, WAL + `synchronous=NORMAL`** | 529 | **0.19 ms** | 61 ms | **0 / 160 (0 %)** |

Two separate conclusions:

- **WAL + `synchronous=NORMAL` is a 3× write-throughput win** and cuts p50 write
  latency 3×.
- **A shared connection is a correctness problem, not just a slow one.** Without
  a lock, 14–22 % of concurrent writes fail with `sqlite3.InterfaceError`
  ("bad parameter or other API misuse"), `SystemError` ("error return without
  exception set"), `OperationalError` ("cannot commit — no transaction is
  active") and `DatabaseError` ("another row available"). An early HTTP run at
  only 60 users reproduced exactly this family — 103 `InterfaceError`, 10
  "cannot commit", 7 "error return without exception set" — **plus ~20
  `Unknown session_id` responses for sessions the API had just created
  successfully, i.e. silently lost writes**. The `RLock` added mid-session fixes
  this, but only for as long as every future DB method remembers the decorator.
  Per-connection-per-thread has no such footgun.

Note the trade-off the third row exposes: correctness costs tail latency with
SQLite (p95 61 ms, p99 184 ms as 16 connections contend for the file write
lock). That tail is a property of SQLite, and it is the reason Postgres is on the
critical path for 1000 users rather than a nice-to-have.

### #4 — Chart rendering is CPU-bound and gets *worse* with threads

`overlay_growth_on_chart()` (`assistant/tools/clinical.py`) now correctly avoids
`pyplot` global state (it builds `Figure` + `FigureCanvasAgg` directly, so the
earlier `plt.close("all")` hazard is gone). It remains expensive
(`--only chart`, 40 renders):

| Mode | p50 | p95 | CPU per chart | Throughput | Speed-up |
|---|---:|---:|---:|---:|---:|
| Serial | 104 ms | 129 ms | **37.9 ms** | 9.4 charts/s | 1.0× |
| 16 threads | 1 891 ms | 2 134 ms | 42.2 ms | 7.7 charts/s | **0.82×** |

**Threads make chart rendering slower.** matplotlib layout, tick computation and
legend bbox maths are pure Python holding the GIL; PNG encoding adds PIL work.
`py-spy` caught this directly — repeated samples of the live server showed
exactly one `active+gil` thread inside `savefig` (legend `_get_bbox_and_child_offsets`,
`ticker._raw_ticks`, `figure.draw`) while all ~40 other threads were parked, and
the aggregate profile attributed ~7.6 % of sampled time to
`get_text_width_height_descent`, `draw_text`, `_draw_text_glyphs_and_boxes` and
PIL's `_encode_tile`. **A single chart render starves the event loop**, which is
why `GET /api/health` and `GET /` degrade in lockstep with chart traffic.

Two secondary costs in the same function: it `glob()`s `data/overlays/` on every
render to unlink stale files (that directory reached 873 files during this
session, and the scan is O(directory)), and two threads plotting the same
child+measure can unlink each other's output, producing a 404 from
`GET /api/overlays/{file}`.

### #5 — BM25 medical retrieval: fine today, quadratic by construction

`--only bm25`: index **built once per process** (18.9 ms rebuild, 243 docs, so
no per-request rebuild here — good), query cost 3.9 ms p50, 257 queries/s on one
core, and 0.94× speed-up across 16 threads (GIL-bound, as expected for pure
Python). Not currently a bottleneck.

The latent issue is `BM25Index.scores()` in `assistant/rag/embeddings.py`: it
loops over every document and calls `Counter(toks)` **inside** the loop, so term
frequencies for the whole corpus are recomputed on every query. At 243 documents
that is 3.9 ms; at 10 000 documents it is ~160 ms per query. Precomputing the
per-document `Counter` at `fit()` time removes it.

### #6 — Caches and memory: currently healthy

- RSS was **flat at 205 MB** across all four stages and a sustained ~8 minute
  run. No leak observed at this duration.
- `assistant/rag/dense.py::_cache` is now a lock-protected LRU `OrderedDict`
  bounded by `NESTLING_DENSE_CACHE_SIZE` (default 2048 ≈ 8 MB per worker
  process for 1024-dim bge-m3 vectors). Previously unbounded; fixed.
- Watch item, not yet a problem: the child RAG store holds every child's
  documents in memory for the process lifetime with no eviction (§1 above), so
  memory grows with total children served, ~3 MB per 1000 children of index text.
- `assistant/refdata`, `assistant/agent/rules.py` and `runtime_translate` use
  `functools.lru_cache` on fixed config — bounded and fine.

---

## 5. What has to change in application code

These were the ranked fixes at measurement time. Status after the hardening pass:

| # | Fix | Status |
|---|---|---|
| 3 | SQLite `PRAGMA journal_mode=WAL; synchronous=NORMAL` | **Applied** in `child_db.py` / `chat_memory.py` (measured ~3× write throughput) |
| 11 | Precompute BM25 term frequencies in `fit()` | **Applied** in `embeddings.py` |
| 7 | Vision route off event loop | **Already applied** (`run_in_threadpool`) |
| 13 | `GET /api/ready` for LB probes | **Applied** |
| 1–2, 4–6, 8–10, 12, 14 | Per-child RAG, per-thread DB connections, chart process pool, LLM semaphore, etc. | Still required for true 1000-user capacity — see table below |

Original precise list (files that still need deeper work):

| # | File / function | Problem | Fix |
|---|---|---|---|
| 1 | `assistant/rag/stores.py` → `ChildRAG.reindex_child()` | Rebuilds BM25 over all children and rewrites `docs.json` on every write; O(total children); no lock | Per-child index (or one SQLite FTS5 table keyed by `child_id`); incremental update instead of full `fit()`; never write the whole corpus inside a request |
| 2 | `assistant/rag/stores.py` → `VectorStore.search()` | Scores every document then filters by `child_id` | Filter candidates **before** scoring, or keep per-child indexes |
| 3 | `assistant/memory/chat_memory.py`, `assistant/memory/child_db.py` → `__init__` | ~~`journal_mode=delete`, `synchronous=FULL`~~ | ~~`PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL`~~ **DONE** |
| 4 | Same files → shared `self.conn` + `RLock` | One serial DB for the whole process; also blocks `NESTLING_WEB_CONCURRENCY>1` | Per-thread connection via `threading.local()` (or a small pool), keep WAL. Measured: 0 failures vs 22 % with a bare shared connection |
| 5 | `assistant/memory/child_db.py` → `create_child`, `add_growth`, `add_screening` | Each does its own commit **and then** `add_event()` commits again — 2 fsyncs per logical write | One transaction per logical operation |
| 6 | `assistant/agent/orchestrator.py` → `chat()` | 13.75 commits and 34.75 statements per turn; `build_context` → `get_history()` reloads the full session and re-parses JSON (206 `json.loads` per turn) | Batch writes into one transaction; `SELECT` only the recent window with `LIMIT`; avoid re-parsing `meta_json`/`tool_calls_json` when unused |
| 7 | `app/api/routes.py` → `chat_vision` | ~~blocking work on event loop~~ | ~~Make it `def`, or `await run_in_threadpool(...)`~~ **DONE** |
| 8 | `assistant/tools/clinical.py` → `overlay_growth_on_chart()` | 37.9 ms GIL-held CPU per chart, negative thread scaling; starves the event loop | Move rendering to a `ProcessPoolExecutor` (or a separate worker service) so it does not hold the GIL; cache by `(child_id, measure, value)` — the filename is already content-addressed, so a cache hit is free |
| 9 | Same function | `OVERLAY_DIR.glob()` + `unlink` per render; two threads can delete each other's PNG | Derive the stale-file name deterministically, or sweep on a schedule instead of per request |
| 10 | `app/api/routes.py` → `child_dossier` | `sorted(OVERLAY_DIR.glob(...), key=lambda x: x.stat().st_mtime)` — a `stat()` per file in a directory that grew to 873 files | Track overlays in the DB, or bound the scan |
| 11 | `assistant/rag/embeddings.py` → `BM25Index.scores()` | ~~`Counter(toks)` recomputed per document per query~~ | ~~Precompute term frequencies in `fit()`~~ **DONE** |
| 12 | `assistant/llm/qwen_client.py` → `QwenClient.chat()` | 180 s timeout, no queue cap, no concurrency limit, no circuit breaker; blocking `urllib` | Add a semaphore sized to `VLLM_MAX_NUM_SEQS`, a short queue with fast 503, and a per-request timeout ≤ 30 s (§6) |
| 13 | `app/api/routes.py` | ~~No readiness endpoint distinct from `/api/health`~~ | ~~Add `GET /api/ready`~~ **DONE** |
| 14 | `assistant/config.py` | ~~`CHILD_DB_PATH` / `CHAT_DB_PATH` ignore settings~~ | ~~Read them from settings~~ **DONE earlier in hardening pass** |
| 15 | `app/main.py` | `StaticFiles` serves the SPA from the app process, competing with API traffic for the same GIL | Let nginx serve `web/` directly (§7) |

---

## 6. The LLM path (estimates — the sidecar was not running)

Everything measured above used `NESTLING_USE_LLM=0`. With the sidecar in the
loop, the app tier stops being the bottleneck and the GPU becomes it.

From `docker-compose.yml` (`llm` service) and `assistant/llm/qwen_client.py` —
**configuration facts, not estimates**:

- `VLLM_MAX_NUM_SEQS=1` — the sidecar serves **one request at a time**. This is
  the hard concurrency limit of the entire product when `NESTLING_USE_LLM=1`.
- `VLLM_MAX_MODEL_LEN=1536`, `VLLM_GPU_MEMORY_UTILIZATION=0.86`,
  `VLLM_ENFORCE_EAGER=1` (no CUDA graphs — slower decode), fp8 weights and KV
  cache, `VLLM_TENSOR_PARALLEL_SIZE=1`, sized for one 8 GB consumer GPU.
- `QwenClient` uses `urllib` with a **180 s** timeout, `max_tokens=320` for RAG
  answers, and **no semaphore, no queue bound and no circuit breaker**.

Arithmetic from those numbers (**not measured — verify before relying on it**):
a 4B fp8 model in eager mode on an 8 GB consumer GPU decodes on the order of
30–60 tok/s single-stream, so 320 output tokens ≈ **5–10 s per turn**, and with
`max_num_seqs=1` that is **~0.1–0.2 chat turns/s for the whole product**.

The failure mode is worse than the throughput number suggests. Because there is
no concurrency limit in the app, N concurrent chat requests each occupy an AnyIO
worker thread blocked in `urllib` while the GPU serves them strictly one at a
time. With 40 threads, the 40th request waits ~40 × 8 s ≈ 320 s — past its 180 s
timeout — so under load the system **times out from the back of an invisible
queue** while the GPU stays busy on work nobody is waiting for any more.

Capacity needed for 1000 users, at one chat turn per active user per 5 minutes
(3.3 turns/s):

| Setup | Est. turns/s | GPUs for 3.3 turns/s |
|---|---:|---:|
| Today: 4B fp8, `max_num_seqs=1`, 8 GB consumer GPU, eager | 0.1–0.2 | **17–33** |
| Same GPU, `max_num_seqs=16–32`, CUDA graphs on (`ENFORCE_EAGER=0`) | 0.6–1.2 | 3–6 |
| 24 GB datacentre GPU (L4/A10G), `max_num_seqs=32+`, continuous batching | 1.5–3 | **2–3** |

**A 1000-user product cannot run on one 4B model on one consumer GPU.** The
required changes, in order: raise `VLLM_MAX_NUM_SEQS` and disable eager mode to
get continuous batching; add an app-side semaphore matching it plus a bounded
queue that returns 503 fast; cut the client timeout to ≤ 30 s; then scale to 2–3
datacentre GPUs behind their own load balancer. Until then, keep
`NESTLING_USE_LLM=0` in production and serve extractive RAG, which is what the
measured numbers above describe.

---

## 6a. The LLM path, measured on a real deployment (2026-09-02)

The estimates in §6 predate two changes that make them out of date, and a live
deploy that replaces the arithmetic with numbers.

**What changed since §6.** The served model is now **openbmb/MiniCPM5-1B**, not a
4B. `VLLM_MAX_NUM_SEQS` and `VLLM_MAX_MODEL_LEN` are no longer pinned to `1` and
`1536` — `scripts/size_llm.py` derives them from the GPU and the checkpoint at
deploy time, and `scripts/size_llm.py` now also derives the precision from the
card's compute capability (fp8 on Ada/Hopper, bf16 below). The §6 failure-mode
analysis (an unbounded app-side queue in front of a one-at-a-time GPU) still
stands and is still the next thing to fix; what has changed is the concurrency
floor.

**The box.** A single **RTX 2080 Ti** (Turing, sm_75, 11 GiB), 16 vCPU, 44 GiB
RAM, Ubuntu 24.04, driver 580 / CUDA 13, vLLM `v0.28.0`. Deployed end to end by
`./deploy.sh --yes` with no manual step.

**Sizing, predicted vs. what vLLM then measured.** `size_llm.py` read the card
and the checkpoint and emitted `VLLM_MAX_MODEL_LEN=4096`, `VLLM_MAX_NUM_SEQS=73`,
`VLLM_QUANTIZATION=none`, `VLLM_KV_CACHE_DTYPE=auto` (bf16 → fp16 on this card).
vLLM's own KV-cache probe at startup reported **292,800 tokens / 71.5×
concurrency** against the predicted 300,903 / 73 — within 3 %. The precision
derivation is load-bearing, not cosmetic: the previous default pinned fp8, which
vLLM **refuses to start** on sm_75, so the sidecar would have silently failed and
the app would have fallen back to extractive RAG while reporting a healthy
deploy.

**Reasoning control genuinely reaches the model.** MiniCPM is a hybrid-reasoning
model; with thinking on, a 200-token reply is spent entirely inside `<think>`
and the parent sees nothing. Measured on this sidecar over `/v1/chat/completions`:

| `chat_template_kwargs` | finish | visible answer |
|---|---|---|
| *(none)* | `length` | truncated `<think>…`, no answer |
| `{"enable_thinking": false}` | `stop` | clean answer, 152 tokens |
| `{"enable_thinking": true}` | `length` | reasoning, truncated |

The app sends `enable_thinking:false`, and vLLM's chat template honours it — the
thing Ollama's shim silently ignored during local testing.

**Single-stream generation.** ~16 tok/s decode (eager mode, fp16). A grounded
RAG turn (≈200-token reply) returns in **8–12 s** unloaded.

**End-to-end conversation quality, on the deployed HTTP API.** Five generated,
seeded scenarios from `tests/scenarios.py` (no hand-written stories), 14 turns
each, driven through the real `/api/*` endpoints on the box:

| | |
|---|---|
| Scored checks passed | **69 / 70 (99 %)** |
| Cross-session recall (allergen, clinic, medication, condition) | 19 / 20 |
| Emergency escalation | 5 / 5 |
| Never recites its own prompt / never denies data it holds | 5 / 5, 4 / 5 |
| Turn latency | median **11.4 s**, p90 19.6 s, max 44.3 s |

The p90/max reflect `max_num_seqs`-serialised turns under the harness firing with
no think time; a real user with think time sees the median. This is a
single-user quality-and-latency measurement, **not** a concurrency test — the §6
and §7 scaling analysis is unchanged.

**Security, probed live.** A route-driven probe (enumerated from the server's own
`/openapi.json`, so it cannot miss an endpoint or test one that does not exist)
run against the deployed box as one account attacking another: anonymous access
refused on every route, cross-account access refused on every data route,
generated SQL/template injections stored as text with the database intact,
forged/`alg:none` tokens refused. It found **one real IDOR** — `POST
/api/sessions` bound a session to any `child_id` without the ownership guard
every sibling route had — which was fixed (`ade6c89`), redeployed, and
re-probed clean. See `tests/test_tenant_isolation.py`.

---

## 6b. Adaptive concurrency, measured on the deployment (2026-09-02)

§6 identified the app as capping concurrency below the GPU (AnyIO's 40-thread
default in front of a sidecar that batches many) and piling load into timeouts
with no admission control. `app/concurrency.py` (commit `af36730`) fixes both:
the app resolves one number — the chat concurrency — from `VLLM_MAX_NUM_SEQS`,
the value `scripts/size_llm.py` already derives from the card, raises the
worker pool to hold it, and gates admission with a bounded queue that sheds to
503 past capacity.

On the RTX 2080 Ti box the sidecar was sized to 73 sequences, so the app came
up with **`max_inflight=73`, `max_waiting=36`** — reported in `/api/health` —
with no manual setting. The same `loadtest.py` (on the box, so the client link
is not the variable) firing identical chat turns at rising concurrency:

| Concurrency | Completed 200/s | Median 200 latency | Shed 503 | Timeouts |
|---:|---:|---:|---:|---:|
| 1 (serial) | 0.09 | 11.1 s | 0 | 0 |
| 8 | 0.54 (6×) | 13.4 s | 0 | 0 |
| 24 | 1.56 (17×) | 11.8 s | 0 | 0 |
| 150 (overload) | 3.44 (38×) | 14.1 s | 76 | **0** |

Two things this shows, neither of them arithmetic:

- **Throughput scales with concurrency while latency stays flat** — 0.09 →
  3.44 turns/s, median holding around 12 s. The app is now using the GPU's
  continuous batching instead of serialising behind a 40-thread wall. On a
  bigger card `VLLM_MAX_NUM_SEQS` is larger and the app widens to it with no
  code change — the "adapt to the GPU" property, measured.
- **Overload sheds instead of collapsing.** At 150 concurrent against a
  109-deep gate, 74 turns completed, 76 were rejected with **503 in ~0.01 s**,
  and **nothing timed out or 500'd**. Peak in-flight reached exactly 73 — the
  app drove the GPU to its full sized batch and no further. This is the §6
  failure mode (timeout from the back of an invisible queue) replaced by a
  fast, honest Retry-After.

The §7 SQLite/state ceiling on the *app tier* is unchanged, and a single GPU
is still a single GPU: this widens the app to the card in front of it and
fails gracefully past it — it does not turn one 2080 Ti into a 1000-user
cluster. What it removes is the app being the bottleneck below the GPU, and
the pileup that turned a burst into a wall of timeouts.

---

## 7. Load balancing and horizontal scaling (implemented here)

### What changed

**`docker-compose.yml`** — added an `nginx` service in front of the app, removed
the app's published host port (traffic now enters through the proxy), and made
every tunable an environment variable with a documented default. Added
`deploy.replicas`, CPU/memory limits and reservations for both services.

**`docker/nginx/nestling.conf.template`** — rendered at container start by the
official nginx image's `envsubst` step, filtered to `^NESTLING_LB_` so nginx's
own `$host`/`$remote_addr` survive. It provides:

- `least_conn` upstream with `keepalive`, `max_fails=3 fail_timeout=10s`.
- **Correct SSE handling** for `/api/chat/stream`: `proxy_buffering off`,
  `proxy_request_buffering off`, `proxy_cache off`, `gzip off`,
  `X-Accel-Buffering: no`, `chunked_transfer_encoding on`, and a long
  `proxy_read_timeout` (default 300 s) because the app runs the whole turn
  before the first token.
- gzip for text/JSON, deliberately **not** for `text/event-stream`.
- Per-location timeouts: 300 s SSE, 120 s `/api/chat`, 30 s other API, 5 s
  health, 5 s connect.
- `limit_req` on chat only (default 2 r/s, burst 5) so one client cannot consume
  the scarce chat capacity. Health is never rate-limited or logged.
- `client_max_body_size` (default 10 m) ≥ the app's 8 MB upload cap.
- Browser-side caching for `/api/overlays/` — chart filenames already encode
  child, measure and value, so they are safe to cache; no `proxy_cache` is
  configured, so per-child images are never stored on the shared proxy.

**`Dockerfile`** — `CMD` is now env-driven, adding no dependency (uvicorn's own
multi-worker supervisor, already present via `uvicorn[standard]`):

```
NESTLING_PORT             listen port                  (default 8000)
NESTLING_WEB_CONCURRENCY  uvicorn worker processes     (default 1)
NESTLING_KEEPALIVE        keep-alive timeout, seconds  (default 15)
NESTLING_BACKLOG          accept queue depth           (default 2048)
```

**Healthchecks** — both services have them, driven by env
(`NESTLING_HC_*`, `NESTLING_LB_HC_*`). The app's uses `/api/health`, which is
acceptable now that `qwen_client` caches its readiness probe for
`NESTLING_LLM_READY_CACHE_SECONDS` (5 s default). It is still a liveness probe,
not a readiness probe — see fix #13.

### Running it

```bash
docker compose up -d --build              # nginx on :8080 → one app replica
docker compose up -d --scale nestling=3   # three replicas
docker compose restart nginx              # required: nginx resolves upstreams at start
docker compose --profile llm up -d        # add the GPU sidecar
```

### Blocking issue for more than 1 worker or replica

`NESTLING_WEB_CONCURRENCY` defaults to **1** and `NESTLING_REPLICAS` to **1**,
on purpose:

- **Multiple workers, one container**: each worker process opens its own
  connection to the same SQLite file. With `journal_mode=delete` a writer takes
  an exclusive lock for the whole transaction, so concurrent cross-process
  writes serialise on the filesystem and surface as `database is locked` once
  `busy_timeout` (5 s) expires. Fix #3 (WAL) is the prerequisite; then
  `NESTLING_WEB_CONCURRENCY=4` is reasonable and is the **single biggest
  available win**, because it turns the one-core ceiling into a four-core one.
- **Multiple replicas**: the app holds an in-process BM25 child index that it
  rewrites to a shared file. Two replicas will overwrite each other's
  `docs.json`, and neither sees the other's writes until reload. Replicas > 1
  is safe for **read-only** traffic only.

The nginx layer is therefore correct and ready, but it cannot be used to scale
writes until the data layer changes.

---

## 8. The state problem, and the migration path

SQLite plus multiple replicas does not work for shared writes. There is no
configuration that fixes it: a single-writer embedded database on a shared
volume is not a substitute for a database server. Concretely, with N replicas on
the current `nestling_children` volume you get lock contention, stale reads, and
a child RAG index that replicas overwrite in turn.

Recommended target, with rough effort. **Not attempted here.**

| Step | What | Effort | Why |
|---|---|---|---|
| 1 | WAL + `synchronous=NORMAL` + per-thread connections | **0.5–1 day** | Unblocks `NESTLING_WEB_CONCURRENCY>1`; measured 3× write throughput. Do this first regardless. |
| 2 | Per-child RAG index (or SQLite FTS5 keyed by `child_id`) | 2–3 days | Removes the O(total children) cost on every write (§4 #1) — the hard blocker for 1000 users |
| 3 | Chart rendering into a `ProcessPoolExecutor` + content-addressed cache | 1–2 days | Stops 38 ms of GIL-held work per chart from starving the event loop |
| 4 | **Postgres** for children, growth, screenings, events, chat sessions/messages/facts | **1–2 weeks** | Real concurrent writers, connection pooling, MVCC reads. Both stores are thin SQL wrappers with no ORM, so the surface is small: `assistant/memory/child_db.py` + `assistant/memory/chat_memory.py`, plus a migration for existing rows |
| 5 | **Redis** for chat session state, rolling summaries, readiness probes and the dense-embedding cache | 3–5 days | Lets any replica serve any user; today session context is re-read from SQLite on every turn |
| 6 | Shared object storage (S3/MinIO) or a volume for `data/overlays` + `data/uploads` | 2–3 days | Chart PNGs and uploads are written to local disk, so a replica can only serve files it generated |
| 7 | LLM: batching + semaphore + bounded queue + 2–3 datacentre GPUs | 1–2 weeks | §6 |

A pragmatic sequence: **1 → 3 → 2** buys a working multi-worker single box
(roughly 4–8× today's throughput) in under a week. Steps 4–6 are what make the
nginx layer in §7 meaningful, i.e. what actually enables horizontal scaling.

---

## 9. Verdict

**What this stack serves today** (single process, LLM off, measured): the app is
capped at about **one CPU core of work and roughly 10–30 requests/second** on a
16-core machine, and latency is already unacceptable at **50 concurrent users**
(p95 14 s, chat p50 7.2 s). Errors stay near zero — it does not crash, it just
queues without bound, which is arguably worse for a parent waiting on a fever
question. A realistic supportable figure is **tens of concurrent users**, not
hundreds. With the LLM sidecar in the path the ceiling drops to
`VLLM_MAX_NUM_SEQS=1`, i.e. **one chat turn at a time**.

**What 1000 concurrent users requires**, none of which is optional:

1. Fix #1 (per-child index) — otherwise every write costs 131 ms at 1000
   children and grows from there.
2. WAL + per-thread connections, then `NESTLING_WEB_CONCURRENCY` sized to CPU.
3. Chart rendering out of the request path (process pool + cache).
4. Postgres for chat/child data and Redis for sessions — the actual precondition
   for more than one replica.
5. Shared storage for overlays and uploads.
6. For the LLM: continuous batching, an app-side concurrency limit with
   backpressure, and 2–3 datacentre-class GPUs. One 4B model on one consumer GPU
   is not a 1000-user configuration.
7. Then the nginx layer in §7 with `NESTLING_REPLICAS` ≥ 3 and a re-run of this
   ramp against a *dedicated* load generator machine.

Until items 1–4 land, adding replicas behind the load balancer would make
correctness worse, not throughput better.

---

## 10. Reproducing

```powershell
python -m pip install -r requirements-dev.txt

$env:NESTLING_LOAD_MODELS="0"; $env:NESTLING_USE_LLM="0"; $env:NESTLING_USE_DENSE="0"
python -m uvicorn app.main:app --host 127.0.0.1 --port 8099 --log-level warning

python -m perf.run_ramp --host http://127.0.0.1:8099 --stages 50,200,500,1000 `
    --duration 90s --workers 4 --server-match "pid:<uvicorn-pid>"
python -m perf.micro_bench --threads 16 --iterations 40
python -m perf.profile_hotpath --target chat --iterations 40 --sort tottime
py-spy record --pid <uvicorn-pid> --duration 30 --format speedscope -o perf/results/p.json
python -m perf.analyze_profile perf/results/p.json
```

Full option list: [`../perf/README.md`](../perf/README.md).

Note: the load test writes into `data/children/*.db` and `data/overlays/`, and
`assistant/config.py` currently ignores the settings fields that would let you
redirect them (fix #14). Back those up before a run.

---

## 11. Repo hygiene

`.gitignore` now excludes agent scratch files (`_*.py`, `_*.txt`, `_*.json`,
`_*.log`, `_*.png`, `_*.md`, `_*.csv`, `_*.html`), `docs/debug-*.log`,
`.pytest_cache/`, `**/__pycache__/`, `extracted/`, and `perf/results/`.

`__init__.py` and `__main__.py` are explicitly re-included — the `_*.py` pattern
would otherwise silently ignore every package marker in the repo.

`.env` is ignored and is **not** tracked; no credential-looking path is tracked.

**123 files are already committed that the new rules would ignore** (64 scratch
files at the repo root and under `docs/`, plus 59 files under `extracted/`).
Ignore rules do not affect tracked files, so removing them needs
`git rm --cached`, which was deliberately not run — that is the owner's call, and
`extracted/` may be wanted in history since it is generated from PDFs that are
themselves gitignored.
