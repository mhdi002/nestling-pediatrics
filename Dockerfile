# Nestling — pediatric parent assistant
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=1000 \
    MPLBACKEND=Agg

WORKDIR /app

# Matplotlib / Pillow runtime libs + curl for healthcheck
RUN apt-get update && apt-get install -y --no-install-recommends \
    libfreetype6 \
    libpng16-16 \
    libjpeg62-turbo \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-core.txt requirements.txt ./

# Default image is slim (no torch). Build with --build-arg INSTALL_ML=1 for optional xLAM.
ARG INSTALL_ML=0
RUN if [ "$INSTALL_ML" = "1" ]; then \
      pip install -r requirements.txt ; \
    else \
      pip install -r requirements-core.txt ; \
    fi

COPY . .

# Persistable seed so empty named volumes do not wipe knowledge
RUN mkdir -p /opt/nestling-seed \
    && cp -a /app/data/knowledge /opt/nestling-seed/knowledge \
    && sed -i 's/\r$//' scripts/entrypoint.sh 2>/dev/null || true \
    && chmod +x scripts/entrypoint.sh scripts/docker_test.sh || true \
    && chmod +x scripts/entrypoint.py || true

EXPOSE 8000

ENTRYPOINT ["python", "/app/scripts/entrypoint.py"]
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
