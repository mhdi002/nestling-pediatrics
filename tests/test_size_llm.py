"""Sidecar sizing must follow the GPU and the model, not a fixed default."""

from __future__ import annotations

import json

import pytest

from scripts import size_llm


QWEN_4B = {
    "num_hidden_layers": 36,
    "hidden_size": 2560,
    "num_attention_heads": 32,
    "num_key_value_heads": 8,
    "head_dim": 128,
    "max_position_embeddings": 32768,
}


@pytest.fixture
def snapshot(tmp_path):
    (tmp_path / "config.json").write_text(json.dumps(QWEN_4B), encoding="utf-8")
    # ~8 GiB of weights, the size the 4B model actually occupies.
    (tmp_path / "model.safetensors").write_bytes(b"\0" * 1024)
    return tmp_path


def _plan(monkeypatch, snapshot, gpu_gib, weights_gib=8):
    monkeypatch.setattr(size_llm, "gpu_memory_bytes", lambda: gpu_gib * 1024**3)
    monkeypatch.setattr(size_llm, "weights_bytes", lambda p: weights_gib * 1024**3)
    return size_llm.plan(snapshot, 0.90, 1536, 1)


def test_a_24gb_card_gets_real_concurrency(monkeypatch, snapshot):
    """The default of one sequence serialised every chat on a 3090."""
    p = _plan(monkeypatch, snapshot, 24)
    assert p["max_num_seqs"] > 1, p
    assert p["max_model_len"] >= 1536, p


def test_a_bigger_card_never_gets_less_than_a_smaller_one(monkeypatch, snapshot):
    small = _plan(monkeypatch, snapshot, 12)
    large = _plan(monkeypatch, snapshot, 48)
    budget_small = small["max_model_len"] * small["max_num_seqs"]
    budget_large = large["max_model_len"] * large["max_num_seqs"]
    assert budget_large >= budget_small


def test_context_never_exceeds_what_the_model_supports(monkeypatch, snapshot):
    p = _plan(monkeypatch, snapshot, 640)
    assert p["max_model_len"] <= QWEN_4B["max_position_embeddings"]


def test_a_card_the_weights_barely_fit_keeps_the_floor(monkeypatch, snapshot):
    p = _plan(monkeypatch, snapshot, 9)
    assert p["max_model_len"] == 1536
    assert p["max_num_seqs"] == 1
    assert "keeping defaults" in p["reason"]


def test_no_gpu_reading_keeps_the_defaults(monkeypatch, snapshot):
    monkeypatch.setattr(size_llm, "gpu_memory_bytes", lambda: None)
    p = size_llm.plan(snapshot, 0.90, 1536, 1)
    assert (p["max_model_len"], p["max_num_seqs"]) == (1536, 1)


def test_a_missing_model_config_keeps_the_defaults(monkeypatch, tmp_path):
    monkeypatch.setattr(size_llm, "gpu_memory_bytes", lambda: 24 * 1024**3)
    p = size_llm.plan(tmp_path, 0.90, 1536, 1)
    assert (p["max_model_len"], p["max_num_seqs"]) == (1536, 1)


def test_kv_cost_uses_kv_heads_not_attention_heads():
    """Grouped-query attention is why this model's cache is affordable."""
    cost = size_llm.kv_bytes_per_token(QWEN_4B)
    expected = 2 * 36 * 8 * 128 * 2
    assert cost == expected


def test_context_window_covers_what_the_app_actually_sends():
    """The sizing constant must not fall below the app's own prompt caps.

    If settings grow the prompt, a window sized for the old cap would silently
    truncate; this fails instead.
    """
    from assistant.settings import get_settings

    s = get_settings()
    # The sidecar sees the context, the query, a system prompt, and must have
    # room to generate the reply. Four characters per token is the usual
    # English ratio and is deliberately conservative here.
    chars = s.llm_prompt_context_chars + s.llm_prompt_query_chars
    prompt_tokens = chars / 4
    needed = prompt_tokens + s.llm_max_tokens_rag
    assert size_llm.DEFAULT_CONTEXT_TOKENS >= needed * 1.5, (
        f"context {size_llm.DEFAULT_CONTEXT_TOKENS} too small for "
        f"~{needed:.0f} tokens of prompt+reply"
    )


def test_concurrency_beats_context_on_a_big_card(monkeypatch, snapshot):
    """Sizing a 24 GiB card for full 32k context gave two sequences."""
    p = _plan(monkeypatch, snapshot, 24)
    assert p["max_num_seqs"] >= 10, p
    assert p["max_model_len"] == size_llm.DEFAULT_CONTEXT_TOKENS


def test_proxy_admission_follows_sidecar_capacity():
    """A fixed burst of 5 turned away the sixth caller on a 22-sequence card."""
    assert size_llm.lb_limits(22)["lb_chat_burst"] == 22
    assert size_llm.lb_limits(3)["lb_chat_burst"] == 3
    # Sustained rate is throughput: one batch clears per generation.
    assert size_llm.lb_limits(120)["lb_chat_rps"] >= size_llm.lb_limits(12)["lb_chat_rps"]
    # Never zero, or the proxy would admit nothing at all.
    assert size_llm.lb_limits(1)["lb_chat_rps"] >= 1


def test_a_small_card_keeps_the_compose_defaults(monkeypatch, snapshot):
    """Emitting nothing leaves docker-compose's own limits in force."""
    p = _plan(monkeypatch, snapshot, 9)
    assert "lb_chat_burst" not in p, "would tighten the proxy on a small GPU"
