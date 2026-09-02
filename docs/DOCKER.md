# Nestling Docker (server deploy)

Two ways to bring the stack up: the unattended `deploy.sh` (recommended on a
fresh GPU host — it installs Docker + the NVIDIA toolkit, fetches the model,
**sizes the sidecar for the actual GPU**, and starts everything), or `docker
compose` directly when you want to drive it yourself.

## Recommended: `./deploy.sh`

```bash
./deploy.sh --yes
```

See the README ("Full stack, one command") for what it does. The key thing
this doc adds is *what it writes*: `deploy.sh` runs `scripts/size_llm.py`
against the real card and appends the derived values to `.env` —
`VLLM_MAX_MODEL_LEN`, `VLLM_MAX_NUM_SEQS`, **`VLLM_QUANTIZATION`**,
**`VLLM_KV_CACHE_DTYPE`**, `VLLM_LIMIT_MM_IMAGE`, `NESTLING_LLM_MAX_CONCURRENCY`
and the nginx rate limits. `docker compose` then reads `.env`, so the compose
defaults below are only the fallback when nothing sized the host.

Modes: `--mode full` (GPU, default), `--mode app` (no sidecar, extractive RAG).
Model source: `--model-source auto|hub|hf|ms` (default `auto` tries the Docker
Hub registry, then Hugging Face, then ModelScope — see below).

## App only (no LLM sidecar)

```bash
docker compose up --build -d nestling
docker compose exec nestling python docs/_e2e_docker_scenarios.py
```

Chat still answers, via **extractive RAG** — no generation, no memory-grounded
paraphrase. Good for a host with no GPU.

## Full stack — MiniCPM vLLM sidecar

A local `llm` sidecar based on `vllm/vllm-openai` loads **openbmb/MiniCPM5-1B**
from the host Hugging Face cache.

### Weights must be present first

`deploy.sh` fetches them. To do it by hand, the cache dir is
`${NESTLING_HF_CACHE_HOST:-$HOME/.cache/huggingface}` (on Windows,
`C:/Users/mhf/.cache/huggingface`), and there are three fetchers — use whichever
your network can reach:

```bash
scripts/fetch_model_registry.sh     # Docker Hub OCI image (needs a pairing in config/model_images.txt)
scripts/fetch_model_modelscope.sh   # ModelScope (config/model_mirrors.txt) — reachable where the HF CDN is blocked
# or the HF CLI:  hf download openbmb/MiniCPM5-1B
```

The registry and ModelScope routes exist because the Hugging Face LFS CDN is
throttled or unreachable on some networks — the metadata call succeeds and the
download dies partway. `--model-source auto` in `deploy.sh` tries all three.

### Start

```bash
docker compose --profile llm up --build -d
docker compose --profile llm logs -f llm
```

- Web UI / API via nginx: `http://localhost:8080`
- App direct: `http://localhost:8000`
- OpenAI-compatible API (text + vision target): `http://localhost:8001/v1`

### GPU precision — set it, or the sidecar may not start

The compose default is `fp8` weights + `fp8` KV cache, which needs **Ada
(sm_89) or newer**. On Turing/Ampere (e.g. RTX 2080 Ti, sm_75; RTX 3090,
sm_86) vLLM refuses to start with fp8, and the app silently falls back to
extractive RAG while health checks pass. `deploy.sh` avoids this by deriving
precision from the card. If you run compose **by hand** on a pre-Ada card,
set it yourself first (or just run the sizer and source its output):

```bash
# derive everything for this card and write it to .env:
snap=$(ls -d "$HOME/.cache/huggingface/hub/models--openbmb--MiniCPM5-1B/snapshots"/*)
python3 scripts/size_llm.py --snapshot "$snap" >> .env
docker compose --profile llm up --build -d
```

or minimally, for a Turing/Ampere card:

```bash
VLLM_QUANTIZATION=none VLLM_KV_CACHE_DTYPE=auto docker compose --profile llm up -d
```

Requires the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html)
and a working `nvidia-smi`. Verify CUDA after start:

```bash
docker compose --profile llm logs llm | grep -iE "gpu|CUDA|vllm"
```

### Adaptive concurrency

The app widens its request concurrency to match what the sidecar batches, so a
bigger GPU serves more parents at once with no code change. It reads
`VLLM_MAX_NUM_SEQS` (what `size_llm.py` derived) as the adaptive default,
raises the worker-thread pool to hold it, and sheds overflow with a fast `503`
+ `Retry-After` instead of queuing into timeouts. Occupancy is visible in the
health endpoint:

```bash
curl -s http://localhost:8080/api/health | python3 -m json.tool
# -> "capacity": { "chat_gate_enabled": true, "max_inflight": 73, "max_waiting": 36, ... }
```

Pin it with `NESTLING_LLM_MAX_CONCURRENCY` (0 = adapt). See `app/concurrency.py`
and `docs/PERFORMANCE.md` §6b for the measured numbers.

### Fallback behaviour

- If the `llm` profile is not running (no GPU / incomplete weights / wrong
  precision), Nestling still serves chat with **extractive RAG**
  (`NESTLING_USE_LLM=0` or an unreachable LLM). CPU-friendly; no generation.
- MiniCPM5-1B is **text-only**. `/api/chat/vision` still returns `200` and
  falls back to caption + care-notes guidance — the capability probe sends a
  1-pixel image, sees the refusal, and reports vision unavailable rather than
  pretending to have looked. Point `NESTLING_VISION_LLM_URL` at a
  vision-capable sidecar to enable real photo analysis.

## Env vars

Defaults below are the compose fallback; `deploy.sh` overrides the derived ones
in `.env`.

| Variable | Default | Purpose |
|---|---|---|
| `NESTLING_LLM_URL` | `http://llm:8000` | App → LLM base URL |
| `NESTLING_VISION_LLM_URL` | `http://llm:8000` | App → vision base URL (same service) |
| `NESTLING_USE_LLM` | `1` | `0` = force extractive fallback |
| `NESTLING_LLM_USE_MODEL_ID` | `0` | `0` = load from local snapshot path (offline); `1` = HF model id |
| `NESTLING_LLM_MODEL` | `openbmb/MiniCPM5-1B` | OpenAI served model name |
| `NESTLING_VISION_MODEL` | `openbmb/MiniCPM5-1B` | Vision target model name (same endpoint) |
| `NESTLING_HF_CACHE_HOST` | `~/.cache/huggingface` | Host HF cache mount (read-only) |
| `VLLM_MAX_MODEL_LEN` | `1536` (derived ↑) | vLLM context window; sized from GPU |
| `VLLM_MAX_NUM_SEQS` | `1` (derived ↑) | vLLM batch size; also the app's concurrency default |
| `VLLM_QUANTIZATION` | `fp8` (derived per card) | `none` on pre-Ada GPUs — **see precision note** |
| `VLLM_KV_CACHE_DTYPE` | `fp8` (derived per card) | `auto` on pre-Ampere GPUs |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.86` | vLLM GPU memory target |
| `VLLM_LIMIT_MM_IMAGE` | `0` (derived) | images per prompt; raised when card + model allow |
| `NESTLING_LLM_MAX_CONCURRENCY` | `0` (adapt) | in-flight chat turns; 0 = match `VLLM_MAX_NUM_SEQS` |
| `NESTLING_CHAT_ACQUIRE_TIMEOUT_S` | `20` | wait before shedding a queued turn (503) |
| `NESTLING_WORKER_THREADS` | `0` (derive) | AnyIO pool; 0 = sized from concurrency |
| `NESTLING_LB_HOST_PORT` | `8080` | nginx host port |
| `NESTLING_LLM_HOST_PORT` | `8001` | llm sidecar host port |
| `NVIDIA_VISIBLE_DEVICES` | `all` | GPU selection |

## Stop

```bash
docker compose --profile llm down
```
