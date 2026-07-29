# Nestling Docker (server deploy)

## App only

```bash
docker compose up --build -d nestling
docker compose exec nestling python docs/_e2e_docker_scenarios.py
```

Open http://localhost:8000 — chat, growth, photo attach (📷).

## Full stack — Bonsai-27B Q1_0 + vision mmproj

Uses `llama-cpp-python` + Hugging Face weights (GitHub not required).

```bash
# Optional but strongly recommended (faster HF download):
# export HF_TOKEN=hf_...

docker compose --profile llm up --build -d
docker compose --profile llm logs -f bonsai
```

First start downloads:

- `Bonsai-27B-Q1_0.gguf` (~3.9 GB)
- `Bonsai-27B-mmproj-Q8_0.gguf` (~0.63 GB)

Weights persist in the `bonsai_models` volume.

When `/api/health` shows `"bonsai":{"ready":true}`, vision chat uses the model.

## Stop

```bash
docker compose --profile llm down
```
