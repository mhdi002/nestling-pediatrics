"""Hybrid intent router: declarative YAML rules + legacy regex classifier."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from assistant.agent.rules import extract_slots_from_rules, match_intents_from_rules


class IntentDecision(BaseModel):
    intents: list[str] = Field(default_factory=list)
    slots: dict[str, Any] = Field(default_factory=dict)
    confidence: float = 0.5
    source: Literal["rules", "regex", "hybrid", "llm"] = "hybrid"
    rationale: str = ""


def route_message(
    user_message: str,
    *,
    prior_slots: dict | None = None,
    en_message: str | None = None,
) -> IntentDecision:
    """
    Primary path: YAML rules + regex classifier (hybrid).
    LLM structured routing plugs in when the text sidecar is available.
    """
    from assistant.agent.intents import (
        GROWTH_COMPUTE_RE,
        SHOW_CHART_RE,
        classify_intent,
    )
    from assistant.agent.slots import extract_growth_slots

    msg = user_message or ""
    en = en_message or msg

    rule_intents = match_intents_from_rules(msg) | match_intents_from_rules(en)
    rule_slots = extract_slots_from_rules(msg)
    rule_slots.update({k: v for k, v in extract_slots_from_rules(en).items() if k not in rule_slots})

    regex_intents = classify_intent(en, prior_slots=prior_slots) | classify_intent(
        msg, prior_slots=prior_slots
    )
    growth_slots = extract_growth_slots(en)
    for k, v in extract_growth_slots(msg).items():
        growth_slots.setdefault(k, v)

    # Prefer regex for clinical safety (battle-tested); enrich with rules.
    intents = set(regex_intents) | set(rule_intents)
    slots = dict(growth_slots)
    for k, v in rule_slots.items():
        slots.setdefault(k, v)

    # Optional LLM boost when text LLM is up
    source: Literal["rules", "regex", "hybrid", "llm"] = "hybrid"
    rationale = f"regex={sorted(regex_intents)} rules={sorted(rule_intents)}"
    try:
        llm_decision = _try_llm_route(en, prior_slots=prior_slots)
        if llm_decision and llm_decision.confidence >= 0.7:
            intents |= set(llm_decision.intents)
            for k, v in (llm_decision.slots or {}).items():
                if v is not None and v != "":
                    slots.setdefault(k, v)
            source = "llm"
            rationale = llm_decision.rationale or rationale
    except Exception:
        pass

    # Care / feeding questions must never pick up growth from LLM or rules alone.
    show_chart = bool(SHOW_CHART_RE.search(msg) or SHOW_CHART_RE.search(en))
    growth_compute = bool(GROWTH_COMPUTE_RE.search(msg) or GROWTH_COMPUTE_RE.search(en))
    if (
        ("medical" in intents or "screening" in intents)
        and not show_chart
        and not growth_compute
        and not slots.get("want_overlay")
    ):
        intents.discard("growth")
        intents.discard("growth_analysis")

    conf = 0.85 if regex_intents else (0.6 if rule_intents else 0.4)
    if source == "llm":
        conf = max(conf, 0.7)
    return IntentDecision(
        intents=sorted(intents),
        slots=slots,
        confidence=conf,
        source=source,
        rationale=rationale,
    )


def _try_llm_route(message: str, *, prior_slots: dict | None = None) -> IntentDecision | None:
    """JSON-schema route via text model when available; otherwise None."""
    from assistant.llm.qwen_client import get_qwen, llm_enabled

    if not llm_enabled():
        return None
    client = get_qwen()
    if not client.ready:
        return None
    schema_hint = (
        'Return ONLY JSON: {"intents":["growth"|"medical"|"history"|"screening"|"help"|'
        '"growth_analysis"|"reassure"|"slot_update"|"chat"],'
        '"slots":{},"confidence":0.0,"rationale":"..."}'
    )
    prior = ""
    if prior_slots:
        prior = f" Prior slots: { {k: prior_slots[k] for k in list(prior_slots)[:12]} }"
    text = client.answer_with_context(
        query=f"{message}{prior}\n{schema_hint}",
        context="Classify the parent message for a pediatric assistant. Never invent numbers.",
        system=(
            "You are an intent router. Output valid JSON only. "
            "Use medical for feeding/nutrition/food/eat AND skin/wound/scar/cut/injury/"
            "bruise/burn/rash/redness questions (EN or FA: غذا، تغذیه، بخوره، زخم، جراحت). "
            "Also use medical for walking/motor milestones (can't walk, crawling, cruising) "
            "and for speech/talking. Short follow-ups (dada/mama, 'is that good', 'yes she says…') "
            "continue a care topic only when they do NOT introduce a new domain. "
            "Prefer growth only when measurements or explicit chart requests are present. "
            "Never classify scar/wound or walking questions as chat or growth_analysis."
        ),
    )
    import json
    import re

    raw = (text or "").strip()
    try:
        data = json.loads(raw)
    except Exception:
        m = re.search(r"\{.*\}", raw, re.S)
        if not m:
            return None
        data = json.loads(m.group(0))
    return IntentDecision(
        intents=list(data.get("intents") or []),
        slots=dict(data.get("slots") or {}),
        confidence=float(data.get("confidence") or 0.5),
        source="llm",
        rationale=str(data.get("rationale") or "llm"),
    )
