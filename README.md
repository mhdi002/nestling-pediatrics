# Nestling — Pediatrics Parent Assistant

Dual-RAG parent assistant: deterministic growth/screening tools + BM25 retrieval + optional LLM generation.

## Models

| Role | Model |
|---|---|
| Medical generation (text) | Qwen/Qwen3.5-4B (local HF cache via vLLM) |
| Vision endpoint (`/api/chat/vision`) | Same Qwen/Qwen3.5-4B OpenAI endpoint |
| Tool calling (optional) | Salesforce/xLAM-1b-fc-r |
| Fallback | Extractive BM25 snippets |

Growth and screening scores always come from deterministic tools.

## Docker

### App only (fast)

```bash
docker compose up --build -d nestling
docker compose exec nestling python docs/_e2e_docker_scenarios.py
```

### Full stack (single Qwen service)

```bash
docker compose --profile llm up --build -d
docker compose --profile llm logs -f llm
```

See `docs/DOCKER.md` for exact env vars and compose layout.
