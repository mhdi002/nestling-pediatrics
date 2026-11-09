"""History / prompt budget math from CompactSettings (settings-driven)."""

from __future__ import annotations

from dataclasses import dataclass

from assistant.settings import get_settings


@dataclass(frozen=True)
class ModelLimits:
    context_length: int
    max_output_tokens: int


@dataclass(frozen=True)
class HistoryBudget:
    window: int
    output_reserve: int
    history: int
    trigger: int
    keep: int
    newest_limit: int
    max_rows: int


def model_limits() -> ModelLimits:
    s = get_settings()
    return ModelLimits(
        context_length=int(s.rlm_context_length),
        max_output_tokens=int(s.rlm_max_output_tokens),
    )


def compute_budget(limits: ModelLimits | None = None) -> HistoryBudget:
    s = get_settings()
    limits = limits or model_limits()
    window = limits.context_length
    output_reserve = min(
        limits.max_output_tokens,
        int(window * float(s.rlm_compact_max_output_ratio)),
    )
    history = int(max(0, window - output_reserve) * float(s.rlm_compact_history_share))
    trigger = int(history * float(s.rlm_compact_trigger_ratio))
    keep = int(history * float(s.rlm_compact_keep_recent_ratio))
    newest_limit = int(history * float(s.rlm_compact_newest_limit_ratio))
    max_rows = max(4, int((history / 1000) * float(s.rlm_compact_max_rows_per_1k_history)))
    return HistoryBudget(
        window=window,
        output_reserve=output_reserve,
        history=history,
        trigger=trigger,
        keep=keep,
        newest_limit=newest_limit,
        max_rows=max_rows,
    )


def prompt_limits(limits: ModelLimits | None = None) -> tuple[int, int]:
    """Return (compact_trigger_tokens, input_limit_tokens)."""
    s = get_settings()
    limits = limits or model_limits()
    input_limit = limits.context_length - min(limits.max_output_tokens, 1024)
    trigger = min(
        int(limits.context_length * float(s.rlm_compact_trigger_ratio_in_turn)),
        input_limit,
    )
    return trigger, input_limit
