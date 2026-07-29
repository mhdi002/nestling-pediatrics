#!/usr/bin/env bash
# Build, smoke-test Nestling via docker compose, then tear down.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

echo "==> docker compose build"
docker compose build

echo "==> docker compose up -d"
docker compose up -d

cleanup() {
  echo "==> docker compose down"
  docker compose down
}
trap cleanup EXIT

echo "==> waiting for health"
ok=0
for i in $(seq 1 60); do
  if curl -fsS "http://localhost:8000/api/health" >/tmp/nestling_health.json 2>/dev/null; then
    cat /tmp/nestling_health.json
    echo
    ok=1
    break
  fi
  sleep 2
done
if [ "$ok" -ne 1 ]; then
  echo "Health check failed"
  docker compose logs --tail=80
  exit 1
fi

echo "==> create child"
CHILD_JSON=$(curl -fsS -X POST "http://localhost:8000/api/children" \
  -H "Content-Type: application/json" \
  -d '{"name":"DockerBaby","sex":"male","gestational_age_weeks":32}')
echo "$CHILD_JSON"
CHILD_ID=$(python -c "import json,sys; print(json.load(sys.stdin)['child_id'])" <<<"$CHILD_JSON")

echo "==> growth"
curl -fsS -X POST "http://localhost:8000/api/growth" \
  -H "Content-Type: application/json" \
  -d "{\"child_id\":\"$CHILD_ID\",\"sex\":\"male\",\"measure\":\"weight\",\"weeks\":40,\"value\":3.2}"
echo

echo "==> docker smoke tests passed"
