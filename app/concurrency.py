"""Adaptive request concurrency for the chat path.

The sidecar is the scarce resource, and vLLM already serves several requests
at once by continuous batching -- up to VLLM_MAX_NUM_SEQS, which
scripts/size_llm.py derives from the actual GPU (about 70 on an 11 GiB 2080
Ti, ~200 on a 24 GiB card). The app, though, ran every chat turn in the AnyIO
default thread pool of 40 and had no admission control at all, so two things
went wrong at load:

  * the app capped concurrency below what the GPU could serve -- a bigger card
    bought nothing, because the 41st turn waited for a thread while the GPU
    sat half idle;
  * nothing shed load, so a burst past capacity filled the thread pool with
    turns all blocked on the GPU, and requests timed out from the back of an
    invisible queue while the GPU stayed busy on work nobody was waiting for.

This module fixes both from one number. The effective concurrency is resolved
from what the sidecar was actually sized to (below), the worker thread pool is
raised to hold it, and an admission gate lets that many turns run, queues a
small slack beyond it, and returns 503 fast once even the slack is full. Raise
the GPU and the whole thing widens on its own; run app-only and it disables
itself.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass

log = logging.getLogger(__name__)

# A hard ceiling on derived concurrency. A very large card could size vLLM for
# several hundred sequences; matching that thread-for-thread would spend the
# host's cores on context-switching before the GPU was the limit. Past this,
# raising it further is a deliberate operator choice, not an automatic one.
_CONCURRENCY_CEILING = 256

# Threads kept aside for everything that is not a chat turn -- health checks,
# login, listing children, growth charts -- so admission control on chat never
# makes the rest of the API unresponsive.
_NON_CHAT_THREAD_RESERVE = 8


def resolve_chat_concurrency(settings) -> int:
    """How many chat turns may hit the sidecar at once, adapted to the host.

    Resolution order, most explicit first:

      1. NESTLING_LLM_MAX_CONCURRENCY, when set > 0 -- an operator override.
      2. VLLM_MAX_NUM_SEQS from the environment -- what the sidecar was
         actually sized to for this GPU. This is the adaptive path: deploy.sh
         writes it from scripts/size_llm.py, so the app widens to match the
         card without a second knob to keep in sync.
      3. a conservative floor, when neither is present (a hand-started stack,
         or app-only) -- enough to serve a few parents at once, not so much
         that an un-sized deploy melts.

    Returns 0 when generation is off entirely, which disables the gate: an
    extractive-RAG turn is CPU work that does not need protecting from the GPU
    it never calls.
    """
    if not getattr(settings, "nestling_use_llm", True):
        return 0

    override = int(getattr(settings, "nestling_llm_max_concurrency", 0) or 0)
    if override > 0:
        return min(override, _CONCURRENCY_CEILING)

    sized = _int_env("VLLM_MAX_NUM_SEQS")
    if sized and sized > 0:
        return min(sized, _CONCURRENCY_CEILING)

    return int(getattr(settings, "nestling_chat_concurrency_floor", 8) or 8)


def resolve_worker_threads(settings, concurrency: int) -> int:
    """Size the AnyIO thread pool to hold the admitted turns plus a reserve.

    An explicit NESTLING_WORKER_THREADS wins. Otherwise the pool has to be at
    least the concurrency (turns that are running) plus the queue slack (turns
    waiting inside a worker thread for their admission) plus a reserve for the
    rest of the API -- anything less and admitted turns would starve for a
    thread to run in, reintroducing the cap this removes.
    """
    override = int(getattr(settings, "nestling_worker_threads", 0) or 0)
    if override > 0:
        return override
    if concurrency <= 0:
        # Gate disabled (app-only): keep the framework default rather than
        # inflating a pool nothing will use.
        return 40
    slack = _resolve_queue_slack(settings, concurrency)
    return concurrency + slack + _NON_CHAT_THREAD_RESERVE


def _resolve_queue_slack(settings, concurrency: int) -> int:
    """How many turns may wait for admission before load is shed.

    A little slack absorbs the normal jitter of arrivals without a 503; too
    much just moves the timeout pileup behind the gate instead of removing it.
    Defaults to half the concurrency, bounded so it is neither zero nor huge.
    """
    override = int(getattr(settings, "nestling_chat_queue_slack", 0) or 0)
    if override > 0:
        return override
    return max(4, min(concurrency, concurrency // 2 or 4))


class GateBusy(Exception):
    """The admission gate is full; the caller maps this to 503 + Retry-After."""

    def __init__(self, retry_after: float, inflight: int, waiting: int):
        super().__init__("chat capacity reached")
        self.retry_after = retry_after
        self.inflight = inflight
        self.waiting = waiting


@dataclass
class GateStats:
    enabled: bool
    max_inflight: int
    max_waiting: int
    inflight: int
    waiting: int


class ChatGate:
    """Admission control for chat turns.

    Lets `max_inflight` turns run at once, allows `max_waiting` more to wait
    briefly for a slot, and rejects anything beyond that immediately. A waiting
    turn that is not admitted within `acquire_timeout` is also rejected, so a
    slot that never frees (a wedged sidecar) sheds load instead of hanging
    every caller for the full request timeout.
    """

    def __init__(self, max_inflight: int, max_waiting: int, acquire_timeout: float):
        self.enabled = max_inflight > 0
        self.max_inflight = max_inflight
        self.max_waiting = max_waiting
        self.acquire_timeout = acquire_timeout
        self._sem = threading.BoundedSemaphore(max_inflight) if self.enabled else None
        self._lock = threading.Lock()
        self._occupancy = 0  # admitted-and-running plus waiting-for-a-slot

    @contextmanager
    def admit(self):
        """Run the body with a slot held, or raise GateBusy if full.

        Disabled gates are a no-op, so the extractive path pays nothing.
        """
        if not self.enabled:
            yield
            return

        with self._lock:
            if self._occupancy >= self.max_inflight + self.max_waiting:
                inflight = min(self._occupancy, self.max_inflight)
                waiting = self._occupancy - inflight
                raise GateBusy(self.acquire_timeout, inflight, waiting)
            self._occupancy += 1

        admitted = self._sem.acquire(timeout=self.acquire_timeout)
        if not admitted:
            with self._lock:
                self._occupancy -= 1
            raise GateBusy(self.acquire_timeout, self.max_inflight, self.max_waiting)

        try:
            yield
        finally:
            self._sem.release()
            with self._lock:
                self._occupancy -= 1

    def stats(self) -> GateStats:
        with self._lock:
            occ = self._occupancy
        inflight = min(occ, self.max_inflight) if self.enabled else 0
        waiting = max(0, occ - self.max_inflight) if self.enabled else 0
        return GateStats(
            enabled=self.enabled,
            max_inflight=self.max_inflight,
            max_waiting=self.max_waiting,
            inflight=inflight,
            waiting=waiting,
        )


def build_gate(settings) -> ChatGate:
    concurrency = resolve_chat_concurrency(settings)
    slack = _resolve_queue_slack(settings, concurrency) if concurrency > 0 else 0
    timeout = float(getattr(settings, "nestling_chat_acquire_timeout_s", 20.0) or 20.0)
    gate = ChatGate(concurrency, slack, timeout)
    log.info(
        "chat admission gate: enabled=%s inflight=%d waiting=%d timeout=%.1fs",
        gate.enabled, gate.max_inflight, gate.max_waiting, gate.acquire_timeout,
    )
    return gate


def apply_thread_limit(settings, concurrency: int) -> int | None:
    """Raise the AnyIO worker-thread pool to hold the admitted turns.

    Returns the value set, or None if it could not be (no running loop). Must
    be called from inside the event loop -- the limiter lives in a run-scoped
    variable -- so app startup is where this belongs.
    """
    threads = resolve_worker_threads(settings, concurrency)
    try:
        import anyio.to_thread

        limiter = anyio.to_thread.current_default_thread_limiter()
        limiter.total_tokens = threads
    except Exception as exc:  # pragma: no cover - depends on the loop being up
        log.warning("could not raise worker thread pool to %d: %s", threads, exc)
        return None
    log.info("worker thread pool sized to %d (chat concurrency %d)", threads, concurrency)
    return threads


def _int_env(name: str) -> int | None:
    raw = os.environ.get(name)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError:
        return None
