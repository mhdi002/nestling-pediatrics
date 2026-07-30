# Nestling Docker (server deploy)

## App only

```bash
docker compose up --build -d nestling
docker compose exec nestling python docs/_e2e_docker_scenarios.py
```

## Full stack — single Qwen vLLM service

Uses a local `llm` sidecar based on `vllm/vllm-openai` that loads Qwen from the host Hugging Face cache.

### Required local path

- `C:/Users/mhf/.cache/huggingface`
  (or set `NESTLING_HF_CACHE_HOST`)

### Start

```bash
docker compose --profile llm up --build -d
docker compose --profile llm logs -f llm
```

- Nestling API: `http://localhost:8000`
- Unified OpenAI API: `http://localhost:8001/v1` (text + vision endpoint target)

### GPU (preferred for Qwen3.5-4B)

- Compose requests one NVIDIA GPU (`gpus: all` + `deploy.resources.reservations.devices`).
- Requires [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) and a working `nvidia-smi` on the host.
- Verify CUDA inside the container after start:

```bash
docker compose --profile llm logs llm | findstr /i "gpu CUDA vllm"
nvidia-smi
```

- **Weights must be complete** before `llm` can start. If logs say `no model weights found`, resume the download on the host:

```powershell
# Only if weights missing (do not re-download if already cached):
# hf download Qwen/Qwen3.5-4B
docker compose --profile llm up --build -d llm
```

### Notes and fallback

- Text endpoint default: `NESTLING_LLM_URL=http://llm:8000`
- Vision endpoint default: `NESTLING_VISION_LLM_URL=http://llm:8000` (same service)
- If the LLM profile is not running (no GPU / incomplete weights), Nestling still serves chat with **extractive RAG** (`NESTLING_USE_LLM=0` or unreachable LLM). This is CPU-friendly fallback — generative Qwen answers require the GPU `llm` service.
- If the selected served model cannot process image inputs with this stack, `/api/chat/vision` still returns `200` and falls back to caption/RAG guidance (`mode` values like `caption+rag_extractive` or `rag_fallback:*`).

## Env vars

| Variable | Default | Purpose |
|---|---|---|
| `NESTLING_LLM_URL` | `http://llm:8000` | App → LLM base URL |
| `NESTLING_VISION_LLM_URL` | `http://llm:8000` | App → vision base URL (same service) |
| `NESTLING_USE_LLM` | `1` | `0` = force extractive fallback |
| `NESTLING_LLM_USE_MODEL_ID` | `0` | `0` = load from local snapshot path (offline); `1` = HF model id |
| `NESTLING_LLM_MODEL` | `Qwen/Qwen3.5-4B` | OpenAI served model name |
| `NESTLING_VISION_MODEL` | `Qwen/Qwen3.5-4B` | Vision target model name (same endpoint) |
| `NESTLING_HF_CACHE_HOST` | `C:/Users/mhf/.cache/huggingface` | Host HF cache mount (read-only) |
| `NESTLING_LLM_MODEL_PATH` | `/models/models--Qwen--Qwen3.5-4B` | In-container model cache root |
| `VLLM_MAX_MODEL_LEN` | `8192` | vLLM model context window |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.90` | vLLM GPU memory target |
| `VLLM_TENSOR_PARALLEL_SIZE` | `1` | vLLM tensor parallel size |
| `NVIDIA_VISIBLE_DEVICES` | `all` | GPU selection |

## Stop

```bash
docker compose --profile llm down
```
