"""Pre-create child records so a ramp stage does not start with a write burst.

    python -m perf.seed --host http://127.0.0.1:8080 --count 200

Writes ``perf/results/seed_children.json``; the locustfile picks it up
automatically and assigns each simulated parent one of the pooled children.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import uuid

import httpx

from perf import config as cfg


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Seed synthetic children for load tests")
    parser.add_argument("--host", default=cfg.HOST)
    parser.add_argument("--count", type=int, default=cfg.SEED_CHILDREN or 200)
    parser.add_argument("--api-key", default=cfg.API_KEY)
    parser.add_argument("--out", default=str(cfg.SEED_POOL))
    args = parser.parse_args(argv)

    headers = {"X-API-Key": args.api_key} if args.api_key else {}
    ids: list[str] = []
    with httpx.Client(base_url=args.host, headers=headers, timeout=cfg.TIMEOUT) as client:
        for _ in range(args.count):
            preterm = random.random() < 0.3
            body = {
                "name": f"seed-{uuid.uuid4().hex[:8]}",
                "sex": random.choice(["male", "female"]),
                "date_of_birth": f"20{random.randint(21, 25)}-{random.randint(1, 12):02d}-15",
                "gestational_age_weeks": round(random.uniform(28.0, 34.0), 1)
                if preterm
                else round(random.uniform(37.0, 41.0), 1),
                "notes": "seeded load-test child",
            }
            resp = client.post(f"{cfg.API_PREFIX}/children", json=body)
            if resp.status_code != 200:
                print(f"seed failed: HTTP {resp.status_code} {resp.text[:200]}", file=sys.stderr)
                return 1
            ids.append(resp.json()["child_id"])

    out = cfg.PERF_DIR / args.out if not str(args.out).startswith(("/", "\\")) else args.out
    out = cfg.SEED_POOL if str(args.out) == str(cfg.SEED_POOL) else out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"host": args.host, "child_ids": ids}, indent=2), encoding="utf-8")
    print(f"seeded {len(ids)} children -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
