"""Hybrid intent router: declarative YAML rules + legacy regex classifier."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal

from pydantic import BaseModel, Field

from assistant.agent.rules import extract_slots_from_rules, match_intents_from_rules
from assistant.settings import get_settings

log = logging.getLogger(__name__)

# Confidence tiers for the hybrid router, highest-trust source first.
REGEX_CONFIDENCE = 0.85
RULES_CONFIDENCE = 0.6
FALLBACK_CONFIDENCE = 0.4
DEFAULT_LLM_CONFIDENCE = 0.5
# How many prior slots are shown to the routing LLM as context.
PRIOR_SLOTS_IN_PROMPT = 12


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
    from assistant.agent.urgency import INTENT as URGENT_INTENT, is_urgent

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
    min_llm_confidence = get_settings().nestling_router_llm_min_confidence
    try:
        llm_decision = _try_llm_route(en, prior_slots=prior_slots)
        if llm_decision and llm_decision.confidence >= min_llm_confidence:
            intents |= set(llm_decision.intents)
            for k, v in (llm_decision.slots or {}).items():
                if v is not None and v != "":
                    slots.setdefault(k, v)
            source = "llm"
            rationale = llm_decision.rationale or rationale
    except Exception as exc:
        # The deterministic rules already produced a decision; the LLM is a bonus.
        log.warning("LLM intent routing failed, keeping the rule-based decision: %s", exc)

    # A parent reporting an emergency is recognised by the LLM above when the
    # sidecar is up, and by the structural test below whether it is or not.
    # The app runs on one GPU that can be down, and an emergency check that
    # only works while the model is up is not an emergency check -- so this
    # runs on every turn, and it reads the parent's own words as well as the
    # translated English, because the translation is an outside HTTP call that
    # fails at the same times as everything else.
    if is_urgent(msg, en_text=en):
        intents.add(URGENT_INTENT)

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

    conf = (
        REGEX_CONFIDENCE
        if regex_intents
        else (RULES_CONFIDENCE if rule_intents else FALLBACK_CONFIDENCE)
    )
    if source == "llm":
        conf = max(conf, min_llm_confidence)
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
        'Return ONLY JSON: {"intents":["urgent"|"growth"|"medical"|"history"|"screening"|'
        '"help"|"growth_analysis"|"reassure"|"slot_update"|"chat"],'
        '"slots":{},"confidence":0.0,"rationale":"..."}'
    )
    prior = ""
    if prior_slots:
        shown = {k: prior_slots[k] for k in list(prior_slots)[:PRIOR_SLOTS_IN_PROMPT]}
        prior = f" Prior slots: {shown}"
    text = client.answer_with_context(
        query=f"{message}{prior}\n{schema_hint}",
        context="Classify the parent message for a pediatric assistant. Never invent numbers.",
        system=(
            "You are an intent router. Output valid JSON only. "
            "Use urgent ONLY when the parent is REPORTING that the child is in "
            "danger right now -- not breathing, choking, convulsing, unconscious "
            "or unresponsive, gone blue or floppy, or bleeding heavily. A "
            "question ABOUT such a sign ('what are the danger signs', 'what "
            "should I do if she has a fit') is medical, never urgent; so is an "
            "ordinary symptom ('she has a mild fever'). "
            "Use medical for feeding/nutrition/food/eat AND skin/wound/scar/cut/injury/"
            "bruise/burn/rash/redness questions (EN or FA: غذا، تغذیه، بخوره، زخم، جراحت). "
            "Also use medical for walking/motor milestones (can't walk, crawling, cruising) "
            "and for speech/talking. Short follow-ups (dada/mama, 'is that good', 'yes she says…') "
            "continue a care topic only when they do NOT introduce a new domain. "
            "Prefer growth only when measurements or explicit chart requests are present. "
            "Never classify scar/wound or walking questions as chat or growth_analysis."
        ),
    )
    raw = (text or "").strip()
    data = _loads_object(raw)
    if data is None:
        m = re.search(r"\{.*\}", raw, re.S)
        data = _loads_object(m.group(0)) if m else None
    if data is None:
        return None
    try:
        confidence = float(data.get("confidence") or DEFAULT_LLM_CONFIDENCE)
    except (TypeError, ValueError):
        confidence = DEFAULT_LLM_CONFIDENCE
    return IntentDecision(
        intents=[str(i) for i in (data.get("intents") or []) if i],
        slots=dict(data.get("slots") or {}),
        confidence=confidence,
        source="llm",
        rationale=str(data.get("rationale") or "llm"),
    )


def _loads_object(raw: str) -> dict | None:
    """Parse `raw` as a JSON object, or None when it is not one."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    return data if isinstance(data, dict) else None
