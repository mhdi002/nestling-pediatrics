# Bonsai LLM/vision sidecar — no GitHub required at build time.
# Uses llama-cpp-python (PyPI) + HuggingFace model download at runtime.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MODELS_DIR=/models \
    PIP_NO_CACHE_DIR=1

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates \
    build-essential cmake g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/bonsai

# Prefer CPU wheels; rebuild from source only if needed
RUN pip install --upgrade pip \
    && pip install "huggingface_hub>=0.24" "fastapi" "uvicorn[standard]" "python-multipart" \
    && (pip install llama-cpp-python \
        || CMAKE_ARGS="-DGGML_NATIVE=OFF" pip install --no-binary=:all: llama-cpp-python)

COPY docker/bonsai/server.py /opt/bonsai/server.py
COPY docker/bonsai/entrypoint.sh /entrypoint.sh
RUN sed -i 's/\r$//' /entrypoint.sh && chmod +x /entrypoint.sh

EXPOSE 8080
VOLUME ["/models"]
ENTRYPOINT ["/entrypoint.sh"]
