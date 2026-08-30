# Nestling — pediatric parent assistant
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=1000 \
    MPLBACKEND=Agg

WORKDIR /app

# Debian package mirror. Defaults to deb.debian.org; override when that host is
# unreachable from the build network (it is blocked in some regions), e.g.
#   docker compose build --build-arg DEBIAN_MIRROR=https://cloudflaremirrors.com/debian
# or set NESTLING_DEBIAN_MIRROR in .env and let compose pass it through.
ARG DEBIAN_MIRROR=""

# Matplotlib / Pillow runtime libs + curl for healthcheck.
# Retried because a single transient mirror failure otherwise fails the build.
RUN set -eux; \
    if [ -n "$DEBIAN_MIRROR" ]; then \
      . /etc/os-release; \
      printf 'deb %s %s main\ndeb %s %s-updates main\n' \
        "$DEBIAN_MIRROR" "$VERSION_CODENAME" \
        "$DEBIAN_MIRROR" "$VERSION_CODENAME" > /etc/apt/sources.list; \
      rm -f /etc/apt/sources.list.d/debian.sources; \
    fi; \
    for i in 1 2 3; do \
      apt-get update && break || { echo "apt-get update failed (attempt $i)"; sleep 5; }; \
    done; \
    apt-get install -y --no-install-recommends \
      libfreetype6 \
      libpng16-16 \
      libjpeg62-turbo \
      curl; \
    rm -rf /var/lib/apt/lists/*

COPY requirements-core.txt requirements.txt ./

# Default image is slim (no torch). Build with --build-arg INSTALL_ML=1 for optional xLAM.
ARG INSTALL_ML=0
# Python package index. Empty means PyPI; override when pypi.org is unreachable
# from the build network. deploy.sh probes config/pypi_mirrors.txt and passes a
# working one through as NESTLING_PIP_INDEX_URL.
ARG PIP_INDEX_URL=""
RUN set -eux; \
    if [ -n "$PIP_INDEX_URL" ]; then \
      host="$(printf '%s' "$PIP_INDEX_URL" | sed -e 's|^https\?://||' -e 's|/.*$||')"; \
      pip config set global.index-url "$PIP_INDEX_URL"; \
      pip config set global.trusted-host "$host"; \
    fi; \
    if [ "$INSTALL_ML" = "1" ]; then \
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

# Serving concurrency is env-driven, never baked in:
#   NESTLING_PORT              listen port                      (default 8000)
#   NESTLING_WEB_CONCURRENCY   uvicorn worker processes         (default 1)
#   NESTLING_KEEPALIVE         keep-alive timeout, seconds      (default 15)
#   NESTLING_BACKLOG           accept queue depth               (default 2048)
#
# WEB_CONCURRENCY defaults to 1 on purpose. Each worker is a separate process
# with its own SQLite connection, and the chat/child databases still run in
# `journal_mode=delete`, so concurrent cross-process writes serialise on the
# file lock and can surface as "database is locked". Raise it only after the
# SQLite work in docs/PERFORMANCE.md ("Blocking issue for >1 worker") lands, or
# after moving to Postgres. Read-heavy deployments can raise it sooner.
ENTRYPOINT ["python", "/app/scripts/entrypoint.py"]
CMD ["sh", "-c", "exec uvicorn app.main:app \
  --host 0.0.0.0 \
  --port ${NESTLING_PORT:-8000} \
  --workers ${NESTLING_WEB_CONCURRENCY:-1} \
  --backlog ${NESTLING_BACKLOG:-2048} \
  --timeout-keep-alive ${NESTLING_KEEPALIVE:-15}"]
