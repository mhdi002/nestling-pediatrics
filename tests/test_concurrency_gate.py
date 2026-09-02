"""The adaptive chat concurrency gate.

Two things are under test: that the effective concurrency is resolved from
the host the way the deploy intends (explicit knob, else the sidecar's sizing,
else a floor), and that the gate itself admits, queues and sheds exactly as
claimed under real threads. The load behaviour is asserted with a barrier and
a slow body, not with sleeps timed to pass.
"""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import pytest

from app.concurrency import (
    ChatGate,
    GateBusy,
    build_gate,
    resolve_chat_concurrency,
    resolve_worker_threads,
)


def _settings(**over):
    base = dict(
        nestling_use_llm=True,
        nestling_llm_max_concurrency=0,
        nestling_chat_concurrency_floor=8,
        nestling_chat_queue_slack=0,
        nestling_chat_acquire_timeout_s=20.0,
        nestling_worker_threads=0,
    )
    base.update(over)
    return SimpleNamespace(**base)


# ---------------------------------------------------------------------------
# Resolving the number from the host
# ---------------------------------------------------------------------------


def test_an_explicit_override_wins(monkeypatch):
    monkeypatch.setenv("VLLM_MAX_NUM_SEQS", "73")
    assert resolve_chat_concurrency(_settings(nestling_llm_max_concurrency=12)) == 12


def test_it_adapts_to_the_sidecar_sizing_when_not_overridden(monkeypatch):
    """The adaptive path: widen to whatever the GPU was sized for."""
    monkeypatch.setenv("VLLM_MAX_NUM_SEQS", "73")
    assert resolve_chat_concurrency(_settings()) == 73


def test_a_bigger_gpu_gives_more_concurrency(monkeypatch):
    monkeypatch.setenv("VLLM_MAX_NUM_SEQS", "8")
    small = resolve_chat_concurrency(_settings())
    monkeypatch.setenv("VLLM_MAX_NUM_SEQS", "198")
    big = resolve_chat_concurrency(_settings())
    assert big > small


def test_it_is_capped_so_a_huge_card_does_not_spawn_unbounded_threads(monkeypatch):
    monkeypatch.setenv("VLLM_MAX_NUM_SEQS", "100000")
    assert resolve_chat_concurrency(_settings()) <= 256


def test_it_falls_back_to_the_floor_with_no_sizing(monkeypatch):
    monkeypatch.delenv("VLLM_MAX_NUM_SEQS", raising=False)
    assert resolve_chat_concurrency(_settings(nestling_chat_concurrency_floor=8)) == 8


def test_generation_off_disables_the_gate(monkeypatch):
    monkeypatch.setenv("VLLM_MAX_NUM_SEQS", "73")
    assert resolve_chat_concurrency(_settings(nestling_use_llm=False)) == 0


def test_a_non_numeric_sizing_is_ignored(monkeypatch):
    monkeypatch.setenv("VLLM_MAX_NUM_SEQS", "not-a-number")
    assert resolve_chat_concurrency(_settings(nestling_chat_concurrency_floor=8)) == 8


# ---------------------------------------------------------------------------
# The worker-thread pool holds the admitted turns
# ---------------------------------------------------------------------------


def test_the_pool_holds_every_admitted_and_queued_turn_plus_a_reserve():
    threads = resolve_worker_threads(_settings(), concurrency=73)
    # 73 running + slack waiting + a reserve for non-chat routes.
    assert threads > 73


def test_an_explicit_pool_size_wins():
    assert resolve_worker_threads(_settings(nestling_worker_threads=50), 73) == 50


def test_a_disabled_gate_keeps_the_framework_default():
    assert resolve_worker_threads(_settings(), concurrency=0) == 40


# ---------------------------------------------------------------------------
# The gate under real threads
# ---------------------------------------------------------------------------


def test_a_disabled_gate_is_a_transparent_no_op():
    gate = ChatGate(0, 0, 0.0)
    ran = []
    with gate.admit():
        ran.append(1)
    assert ran == [1]
    assert gate.stats().enabled is False


def test_it_admits_up_to_the_limit_concurrently():
    gate = ChatGate(max_inflight=3, max_waiting=0, acquire_timeout=5.0)
    start = threading.Barrier(4, timeout=5)  # 3 workers + this thread
    release = threading.Event()
    peak = {"n": 0}
    lock = threading.Lock()

    def worker():
        with gate.admit():
            with lock:
                peak["n"] += 1
            start.wait()
            release.wait(5)
            with lock:
                peak["n"] -= 1

    ts = [threading.Thread(target=worker) for _ in range(3)]
    for t in ts:
        t.start()
    start.wait()  # all three are inside the gate at once
    assert gate.stats().inflight == 3
    release.set()
    for t in ts:
        t.join(5)


def test_it_sheds_once_inflight_and_queue_are_full():
    # One slot, no queue: the second concurrent caller is rejected at once.
    gate = ChatGate(max_inflight=1, max_waiting=0, acquire_timeout=5.0)
    holding = threading.Event()
    release = threading.Event()

    def holder():
        with gate.admit():
            holding.set()
            release.wait(5)

    t = threading.Thread(target=holder)
    t.start()
    assert holding.wait(5)
    with pytest.raises(GateBusy):
        with gate.admit():
            pass
    release.set()
    t.join(5)


def test_a_queued_caller_is_admitted_when_a_slot_frees():
    gate = ChatGate(max_inflight=1, max_waiting=1, acquire_timeout=5.0)
    holding = threading.Event()
    release = threading.Event()
    second_ran = threading.Event()

    def holder():
        with gate.admit():
            holding.set()
            release.wait(5)

    def waiter():
        with gate.admit():  # must wait for holder, then get the slot
            second_ran.set()

    h = threading.Thread(target=holder)
    w = threading.Thread(target=waiter)
    h.start()
    assert holding.wait(5)
    w.start()
    time.sleep(0.2)  # w is now queued, not yet admitted
    assert not second_ran.is_set()
    release.set()
    assert second_ran.wait(5)
    h.join(5)
    w.join(5)


def test_a_queued_caller_gives_up_after_the_timeout():
    # A slot that never frees must not hang the queued caller forever.
    gate = ChatGate(max_inflight=1, max_waiting=1, acquire_timeout=0.3)
    release = threading.Event()

    def holder():
        with gate.admit():
            release.wait(5)

    h = threading.Thread(target=holder)
    h.start()
    time.sleep(0.2)
    t0 = time.time()
    with pytest.raises(GateBusy):
        with gate.admit():
            pass
    waited = time.time() - t0
    assert 0.25 <= waited < 3.0, waited
    release.set()
    h.join(5)


def test_occupancy_returns_to_zero_after_a_rejection():
    """A shed request must not leak a slot, or the gate wedges shut."""
    gate = ChatGate(max_inflight=1, max_waiting=0, acquire_timeout=1.0)
    release = threading.Event()
    holding = threading.Event()

    def holder():
        with gate.admit():
            holding.set()
            release.wait(5)

    h = threading.Thread(target=holder)
    h.start()
    assert holding.wait(5)
    for _ in range(5):
        with pytest.raises(GateBusy):
            with gate.admit():
                pass
    release.set()
    h.join(5)
    # The slot the holder used is released; the gate now admits again.
    with gate.admit():
        pass
    assert gate.stats().inflight == 0


def test_an_exception_in_the_body_still_frees_the_slot():
    gate = ChatGate(max_inflight=1, max_waiting=0, acquire_timeout=1.0)
    with pytest.raises(ValueError):
        with gate.admit():
            raise ValueError("boom")
    # Slot was freed despite the error.
    with gate.admit():
        pass
    assert gate.stats().inflight == 0


def test_build_gate_from_settings(monkeypatch):
    monkeypatch.setenv("VLLM_MAX_NUM_SEQS", "10")
    gate = build_gate(_settings())
    assert gate.enabled
    assert gate.max_inflight == 10
    assert gate.max_waiting >= 1
