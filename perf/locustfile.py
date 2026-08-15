"""Locust harness modelling realistic Nestling parent traffic.

Run headless (single machine)::

    locust -f perf/locustfile.py --headless -u 200 -r 20 -t 2m \
           --host http://127.0.0.1:8080 --csv perf/results/stage-200

Every simulated parent gets its own child record and its own chat session, so
SQLite session/message tables and the per-session memory path see real
contention rather than one shared row.
"""

from __future__ import annotations

import json
import random
import sys
import time
import uuid
from pathlib import Path

from locust import HttpUser, between, events, task

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from perf import config as cfg  # noqa: E402

# Parent questions that exercise different orchestrator intents (chat / medical /
# growth / reassurance) rather than hitting one cached path repeatedly.
PARENT_QUESTIONS = [
    "My baby has had a fever of 38.5 for two days, what should I do?",
    "When should my 9 month old start finger foods?",
    "How much milk does a 6 month old need per day?",
    "My toddler is not walking yet at 15 months, should I worry?",
    "What are the red flags for infant dehydration?",
    "How do I know if my newborn is getting enough breast milk?",
    "My child has a rash on the belly after a fever, is it serious?",
    "How many hours should a 4 month old sleep?",
    "Is it normal for a 2 year old to have tantrums every day?",
    "What vaccines are due at 12 months?",
    "My baby spits up after every feed, what can I do?",
    "How do I introduce allergenic foods safely?",
]

# Per-measure plausible ranges — the API validates these, and out-of-range values
# would show up as HTTP 400s that look like server errors in the report.
MEASURE_RANGES = {
    "weight": (2.5, 15.0),
    "length": (46.0, 88.0),
    "head_circumference": (32.0, 52.0),
}
MEASURES = list(MEASURE_RANGES)
SEXES = ["male", "female"]
ASQ_DOMAINS = [
    "communication",
    "gross_motor",
    "fine_motor",
    "problem_solving",
    "personal_social",
]
ASQ_ANSWERS = ["Yes", "Sometimes", "Not yet"]
ASQ_FORM_AGES = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 27, 30, 33, 36]
STATIC_PATHS = ["/", "/index.html"]

_SEED_CHILDREN: list[str] = []


def _load_seed_pool() -> list[str]:
    """Optional pre-seeded child ids (see perf/seed.py) to avoid a create burst."""
    path = cfg.SEED_POOL
    if not path.is_file():
        return []
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    ids = data.get("child_ids") if isinstance(data, dict) else data
    return [str(i) for i in (ids or [])]


@events.test_start.add_listener
def _on_test_start(environment, **_kwargs):
    global _SEED_CHILDREN
    _SEED_CHILDREN = _load_seed_pool()
    if _SEED_CHILDREN:
        print(f"[perf] using pre-seeded child pool: {len(_SEED_CHILDREN)} ids")
    else:
        print("[perf] no seed pool found — each user creates its own child")


class NestlingParent(HttpUser):
    """One simulated parent using the app from a phone."""

    wait_time = between(cfg.THINK_MIN, cfg.THINK_MAX)
    host = cfg.HOST

    def on_start(self) -> None:
        self.client.timeout = cfg.TIMEOUT
        if cfg.API_KEY:
            self.client.headers.update({"X-API-Key": cfg.API_KEY})
        self.child_id: str | None = None
        self.session_id: str | None = None
        self.overlay: str | None = None
        self._bootstrap()

    # --- setup -------------------------------------------------------------

    def _bootstrap(self) -> None:
        if _SEED_CHILDREN:
            self.child_id = random.choice(_SEED_CHILDREN)
        else:
            self.child_id = self._create_child()
        if cfg.SESSION_PER_USER:
            self.session_id = self._create_session()

    def _create_child(self) -> str | None:
        tag = uuid.uuid4().hex[:8]
        # Mix of term and preterm so both WHO and INTERGROWTH chart paths are used.
        preterm = random.random() < 0.3
        body = {
            "name": f"perf-{tag}",
            "sex": random.choice(SEXES),
            "date_of_birth": f"20{random.randint(21, 25)}-{random.randint(1, 12):02d}-15",
            "gestational_age_weeks": round(random.uniform(28.0, 34.0), 1)
            if preterm
            else round(random.uniform(37.0, 41.0), 1),
            "notes": "synthetic load-test child",
        }
        with self.client.post(
            f"{cfg.API_PREFIX}/children", json=body, name="POST /api/children", catch_response=True
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return None
            try:
                return resp.json().get("child_id")
            except ValueError:
                resp.failure("non-JSON body")
                return None

    def _create_session(self) -> str | None:
        body = {"child_id": self.child_id, "title": None}
        with self.client.post(
            f"{cfg.API_PREFIX}/sessions", json=body, name="POST /api/sessions", catch_response=True
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return None
            try:
                return resp.json().get("session_id")
            except ValueError:
                resp.failure("non-JSON body")
                return None

    # --- tasks -------------------------------------------------------------

    @task(cfg.W_CHAT)
    def chat_turn(self) -> None:
        """Heaviest path: router + chat memory + BM25 RAG (+ LLM when the sidecar is up)."""
        body = {
            "session_id": self.session_id,
            "message": random.choice(PARENT_QUESTIONS),
            "child_id": self.child_id,
            "ui_lang": "en",
        }
        with self.client.post(
            f"{cfg.API_PREFIX}/chat", json=body, name="POST /api/chat", catch_response=True
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:120]}")
                return
            try:
                payload = resp.json()
            except ValueError:
                resp.failure("non-JSON body")
                return
            if not payload.get("reply"):
                resp.failure("empty reply")
            elif payload.get("session_id"):
                self.session_id = payload["session_id"]

    @task(cfg.W_CHAT_STREAM)
    def chat_stream_turn(self) -> None:
        """SSE path — also records time-to-first-token as its own metric."""
        body = {
            "session_id": self.session_id,
            "message": random.choice(PARENT_QUESTIONS),
            "child_id": self.child_id,
            "ui_lang": "en",
        }
        start = time.perf_counter()
        first_token_ms: float | None = None
        events_seen = 0
        with self.client.post(
            f"{cfg.API_PREFIX}/chat/stream",
            json=body,
            name="POST /api/chat/stream",
            stream=True,
            timeout=cfg.STREAM_TIMEOUT,
            catch_response=True,
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}")
                return
            try:
                for raw in resp.iter_lines(decode_unicode=True):
                    if not raw:
                        continue
                    if raw.startswith("event:"):
                        events_seen += 1
                        kind = raw.split(":", 1)[1].strip()
                        if kind == "token" and first_token_ms is None:
                            first_token_ms = (time.perf_counter() - start) * 1000.0
                        elif kind == "error":
                            resp.failure("SSE error event")
                            return
                        elif kind == "done":
                            break
            except Exception as exc:  # network/timeout mid-stream
                resp.failure(f"stream aborted: {type(exc).__name__}")
                return
            if events_seen == 0:
                resp.failure("no SSE events received")
                return
        if first_token_ms is not None:
            # Surfaced in Locust stats so we can see queueing delay before the
            # first byte, which the total stream duration hides.
            self.environment.events.request.fire(
                request_type="SSE",
                name="POST /api/chat/stream [time-to-first-token]",
                response_time=first_token_ms,
                response_length=0,
                exception=None,
                context={},
            )

    @task(cfg.W_GROWTH)
    def submit_growth(self) -> None:
        """Growth measurement + matplotlib chart render (CPU-bound, PNG written to disk)."""
        if not self.child_id:
            return
        measure = random.choice(MEASURES)
        low, high = MEASURE_RANGES[measure]
        body = {
            "sex": random.choice(SEXES),
            "measure": measure,
            "value": round(random.uniform(low, high), 2),
            "child_id": self.child_id,
            "age_months": round(random.uniform(1.0, 23.0), 1),
        }
        with self.client.post(
            f"{cfg.API_PREFIX}/growth", json=body, name="POST /api/growth", catch_response=True
        ) as resp:
            if resp.status_code != 200:
                resp.failure(f"HTTP {resp.status_code}: {resp.text[:120]}")
                return
            try:
                payload = resp.json()
            except ValueError:
                resp.failure("non-JSON body")
                return
            if payload.get("plot_error"):
                resp.failure(f"chart render failed: {str(payload['plot_error'])[:100]}")
                return
            if payload.get("overlay_filename"):
                self.overlay = payload["overlay_filename"]

        if self.overlay:
            self.client.get(
                f"{cfg.API_PREFIX}/overlays/{self.overlay}",
                name="GET /api/overlays/{file}",
            )

    @task(cfg.W_GROWTH_CURVES)
    def growth_curves(self) -> None:
        """JSON percentile curves — pure WHO/INTERGROWTH equation CPU, no image."""
        self.client.get(
            f"{cfg.API_PREFIX}/growth/curves",
            params={"sex": random.choice(SEXES), "measure": random.choice(MEASURES)},
            name="GET /api/growth/curves",
        )

    @task(cfg.W_DOSSIER)
    def read_dossier(self) -> None:
        if not self.child_id:
            return
        self.client.get(
            f"{cfg.API_PREFIX}/children/{self.child_id}/dossier",
            name="GET /api/children/{id}/dossier",
        )

    @task(cfg.W_CHILD_LIST)
    def list_children(self) -> None:
        self.client.get(f"{cfg.API_PREFIX}/children", name="GET /api/children")

    @task(cfg.W_SCREENING)
    def screening(self) -> None:
        """ASQ or M-CHAT scoring; with child_id it also writes a screening row."""
        if random.random() < 0.5:
            age = random.choice(ASQ_FORM_AGES)
            self.client.get(
                f"{cfg.API_PREFIX}/asq/{age}/questions", name="GET /api/asq/{age}/questions"
            )
            body = {
                "age_months": age,
                "child_id": self.child_id,
                "domain_answers": {
                    dom: [random.choice(ASQ_ANSWERS) for _ in range(6)] for dom in ASQ_DOMAINS
                },
            }
            with self.client.post(
                f"{cfg.API_PREFIX}/asq/score",
                json=body,
                name="POST /api/asq/score",
                catch_response=True,
            ) as resp:
                if resp.status_code != 200:
                    resp.failure(f"HTTP {resp.status_code}: {resp.text[:120]}")
        else:
            self.client.get(f"{cfg.API_PREFIX}/mchat/questions", name="GET /api/mchat/questions")
            body = {
                "answers": {str(q): random.choice(["yes", "no"]) for q in range(1, 21)},
                "child_id": self.child_id,
            }
            with self.client.post(
                f"{cfg.API_PREFIX}/mchat/score",
                json=body,
                name="POST /api/mchat/score",
                catch_response=True,
            ) as resp:
                if resp.status_code != 200:
                    resp.failure(f"HTTP {resp.status_code}: {resp.text[:120]}")

    @task(cfg.W_SESSION_HISTORY)
    def session_history(self) -> None:
        if not self.session_id:
            return
        self.client.get(
            f"{cfg.API_PREFIX}/sessions/{self.session_id}", name="GET /api/sessions/{id}"
        )

    @task(cfg.W_STATIC)
    def static_asset(self) -> None:
        self.client.get(random.choice(STATIC_PATHS), name="GET / (static SPA)")

    @task(cfg.W_HEALTH)
    def health(self) -> None:
        self.client.get(f"{cfg.API_PREFIX}/health", name="GET /api/health")
