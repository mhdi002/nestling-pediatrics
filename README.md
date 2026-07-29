# Nestling — Pediatrics Parent Assistant

Dual-RAG parent assistant: INTERGROWTH/WHO growth tools, ASQ/M-CHAT, chat memory, FastAPI + web UI. **Run and deploy with Docker.**

## Models

| Role | Model |
|------|--------|
| RAG + vision (preferred) | [prism-ml/Bonsai-27B-gguf](https://huggingface.co/prism-ml/Bonsai-27B-gguf) — **Q1_0** (~3.9 GB) + **mmproj Q8_0** vision (~0.63 GB) via PrismML llama-server |
| Tool calling (optional) | [Salesforce/xLAM-1b-fc-r](https://huggingface.co/Salesforce/xLAM-1b-fc-r) |
| RAG fallback | [PleIAs/Pleias-RAG-1B](https://huggingface.co/PleIAs/Pleias-RAG-1B) or extractive BM25 |

Growth numbers always come from deterministic equation tools — never from the LLM.

## Docker (required for deploy)

### App only (fast — extractive RAG, no 27B download)

```bash
docker compose up --build -d nestling
docker compose exec nestling python docs/_e2e_docker_scenarios.py
```

Open http://localhost:8000

### Full stack — Bonsai 1-bit + vision

```bash
docker compose --profile llm up --build -d
docker compose --profile llm logs -f bonsai
# wait until /health is ready (~GB download + model load), then:
docker compose exec nestling python docs/_e2e_docker_scenarios.py
```

See [docs/DOCKER.md](docs/DOCKER.md). GPU: `BONSAI_NGL=99`. CPU-only: `BONSAI_NGL=0`.

Chat UI: attach a photo (📷) for rash/wound guidance via `/api/chat/vision`.

## Anti-hallucination rules

1. Growth numbers **only** from equation tools.
2. Screening scores **only** from scoring tools.
3. Medical answers cite retrieved chunks; vision is educational, not a diagnosis.
4. Child history comes from SQLite + child RAG, never invented.
