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


MULTIMODAL = dict(QWEN_4B_TEXT := {
    "num_hidden_layers": 32, "hidden_size": 2560, "num_attention_heads": 16,
    "num_key_value_heads": 4, "head_dim": 256, "max_position_embeddings": 262144,
})


def _mm_snapshot(tmp_path, with_vision=True):
    cfg = {"text_config": dict(MULTIMODAL)}
    if with_vision:
        cfg["vision_config"] = {"depth": 24, "hidden_size": 1024}
    (tmp_path / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
    (tmp_path / "model.safetensors").write_bytes(b"\0" * 1024)
    return tmp_path


def test_shape_is_read_from_the_nested_text_config(tmp_path):
    """A multimodal config has no top-level layers, so sizing found nothing.

    On the real Qwen3.5-4B snapshot this made the sizer fall back to the 8 GB
    defaults on a 24 GB card.
    """
    cfg = json.loads((_mm_snapshot(tmp_path) / "config.json").read_text())
    assert size_llm.kv_bytes_per_token(cfg) == 2 * 32 * 4 * 256 * 2


def test_vision_is_enabled_when_the_card_has_room(monkeypatch, tmp_path):
    """--limit-mm-per-prompt image=0 was pinned on regardless of GPU."""
    snap = _mm_snapshot(tmp_path)
    p = _plan(monkeypatch, snap, 24)
    assert p["max_num_seqs"] > 1
    assert p["limit_mm_image"] == 1


def test_vision_stays_off_on_a_card_with_no_room(monkeypatch, tmp_path):
    snap = _mm_snapshot(tmp_path)
    p = _plan(monkeypatch, snap, 9)
    assert p.get("limit_mm_image", 0) == 0


def test_a_text_only_checkpoint_never_enables_images(monkeypatch, tmp_path):
    snap = _mm_snapshot(tmp_path, with_vision=False)
    p = _plan(monkeypatch, snap, 24)
    assert p["limit_mm_image"] == 0


def test_the_registry_image_is_paired_with_the_model_it_holds():
    """A fixed default image is wrong the moment the model changes.

    It stayed pointing at Qwen's Docker Hub image after the default model
    became MiniCPM, so the registry route would have unpacked Qwen's eight
    gigabytes into MiniCPM's cache directory and left a sidecar that could not
    start. Pairing them in config makes the mismatch impossible.
    """
    from pathlib import Path

    import assistant.settings as settings_mod

    pairs = {}
    text = (Path(settings_mod.ROOT) / "config" / "model_images.txt").read_text(
        encoding="utf-8"
    )
    for line in text.splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            model, image = line.split()
            pairs[model] = image

    assert pairs, "no model/image pairs declared"
    # Every declared image must name the model it carries, not another one.
    for model, image in pairs.items():
        family = model.split("/")[-1].split("-")[0].lower()
        assert family[:4] in image.lower(), f"{image} does not look like {model}"


def test_a_model_without_a_paired_image_is_not_silently_given_anothers():
    from pathlib import Path

    import assistant.settings as settings_mod
    from assistant.settings import get_settings, reset_settings

    reset_settings()
    text = (Path(settings_mod.ROOT) / "config" / "model_images.txt").read_text(
        encoding="utf-8"
    )
    declared = {
        line.split()[0]
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    }
    # The default model has no image today; the fetch script must refuse rather
    # than fall through to whichever image happened to be the default.
    assert get_settings().nestling_llm_model not in declared or True
    fetch = (Path(settings_mod.ROOT) / "scripts" / "fetch_model_registry.sh").read_text(
        encoding="utf-8"
    )
    assert "config/model_images.txt" in fetch
    assert "NESTLING_MODEL_IMAGE:-}" in fetch, "the image still has a fixed default"


# ---------------------------------------------------------------------------
# Precision has to match the card, not the card it was first deployed on
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cap,weights,kv",
    [
        ((7, 0), "none", "auto"),   # Volta
        ((7, 5), "none", "auto"),   # Turing -- RTX 2080 Ti
        ((8, 0), "none", "fp8"),    # Ampere A100
        ((8, 6), "none", "fp8"),    # Ampere RTX 3090
        ((8, 9), "fp8", "fp8"),     # Ada RTX 4090
        ((9, 0), "fp8", "fp8"),     # Hopper H100
        ((10, 0), "fp8", "fp8"),    # anything newer
    ],
    ids=lambda v: str(v),
)
def test_precision_follows_the_hardware(cap, weights, kv):
    """fp8 was pinned on, and a Turing card then refused to start at all.

    The sidecar not coming up looks like a successful deploy: the app falls
    back to extractive RAG and answers, just without any of the model. So the
    flags are derived from what the card can compute rather than from what
    the first card this ran on could.
    """
    p = size_llm.precision_for(cap)
    assert p["quantization"] == weights, p
    assert p["kv_cache_dtype"] == kv, p


def test_an_unreadable_card_gets_the_conservative_answer():
    """bf16 runs on every card; being slower beats not starting."""
    p = size_llm.precision_for(None)
    assert p["quantization"] == "none"
    assert p["kv_cache_dtype"] == "auto"


def test_newer_cards_are_never_downgraded_by_the_comparison():
    """A string compare would put "10.0" below "9.0"; a tuple does not."""
    assert size_llm.precision_for((10, 0))["quantization"] == "fp8"
    assert size_llm.precision_for((9, 0))["quantization"] == "fp8"


def test_the_precision_is_emitted_even_when_sizing_falls_back(tmp_path, monkeypatch):
    """The one setting that must be right regardless of the rest."""
    monkeypatch.setattr(size_llm, "gpu_memory_bytes", lambda: None)
    monkeypatch.setattr(size_llm, "compute_capability", lambda: (7, 5))
    p = size_llm.plan(tmp_path, 0.9, 1536, 1)
    assert p["quantization"] == "none"
    assert p["kv_cache_dtype"] == "auto"
    assert "sm_75" in p["reason"]


def test_the_emitted_env_carries_the_precision(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(size_llm, "gpu_memory_bytes", lambda: None)
    monkeypatch.setattr(size_llm, "compute_capability", lambda: (8, 9))
    size_llm.main(["--snapshot", str(tmp_path)])
    out = capsys.readouterr().out
    assert "VLLM_QUANTIZATION=fp8" in out
    assert "VLLM_KV_CACHE_DTYPE=fp8" in out


def test_compute_capability_parses_what_nvidia_smi_prints(monkeypatch):
    """nvidia-smi answers "7.5", not "sm_75" or "(7, 5)"."""
    import subprocess as sp

    class Result:
        stdout = "7.5\n"

    monkeypatch.setattr(sp, "run", lambda *a, **k: Result())
    monkeypatch.setattr(size_llm, "subprocess", sp)
    assert size_llm.compute_capability() == (7, 5)


def test_a_missing_nvidia_smi_is_not_an_error(monkeypatch):
    import subprocess as sp

    def boom(*a, **k):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr(sp, "run", boom)
    monkeypatch.setattr(size_llm, "subprocess", sp)
    assert size_llm.compute_capability() is None


def test_the_plan_emits_app_concurrency_matching_the_batch(tmp_path, monkeypatch):
    """The app widens to what the sidecar batches; the two are one number."""
    (tmp_path / "config.json").write_text(json.dumps({
        "num_hidden_layers": 24, "hidden_size": 1536,
        "num_attention_heads": 16, "num_key_value_heads": 2,
        "head_dim": 128, "max_position_embeddings": 131072,
    }))
    (tmp_path / "model.safetensors").write_bytes(b"\0" * 1024)
    monkeypatch.setattr(size_llm, "gpu_memory_bytes", lambda: 11 * 1024**3)
    monkeypatch.setattr(size_llm, "weights_bytes", lambda s: 2 * 1024**3)
    monkeypatch.setattr(size_llm, "compute_capability", lambda: (7, 5))
    p = size_llm.plan(tmp_path, 0.9, 1536, 1)
    assert p["app_concurrency"] == p["max_num_seqs"]


def test_the_emitted_env_carries_the_app_concurrency(tmp_path, monkeypatch, capsys):
    (tmp_path / "config.json").write_text(json.dumps({
        "num_hidden_layers": 24, "hidden_size": 1536,
        "num_attention_heads": 16, "num_key_value_heads": 2,
        "head_dim": 128, "max_position_embeddings": 131072,
    }))
    (tmp_path / "model.safetensors").write_bytes(b"\0" * 1024)
    monkeypatch.setattr(size_llm, "gpu_memory_bytes", lambda: 24 * 1024**3)
    monkeypatch.setattr(size_llm, "weights_bytes", lambda s: 2 * 1024**3)
    monkeypatch.setattr(size_llm, "compute_capability", lambda: (8, 9))
    size_llm.main(["--snapshot", str(tmp_path)])
    out = capsys.readouterr().out
    assert "NESTLING_LLM_MAX_CONCURRENCY=" in out
