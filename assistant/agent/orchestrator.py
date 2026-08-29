#!/usr/bin/env python3
"""
Parent assistant orchestrator.

Models:
  - Local Qwen (OpenAI-compatible vLLM sidecar) → RAG answers + normal chat + vision
  - Salesforce/xLAM-1b-fc-r (optional) → tool / function calling

Deterministic clinical tools compute all growth/screening numbers (no LLM math).
Full chat session memory persists multi-turn slots + messages.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from assistant.config import EN_DIR, KNOWLEDGE_DIR
from assistant.memory.chat_memory import ChatMemory
from assistant.memory.child_db import ChildMemoryDB
from assistant.rag.stores import ChildRAG, MedicalRAG
from assistant.refdata import clinical_bounds, weeks_per_month
from assistant.runtime_translate import translate_en_to_fa, translate_for_models
from assistant.settings import get_settings
from assistant.tools.clinical import TOOL_SPECS, _term_age_months_from_weeks, dispatch_tool
from assistant.tools.who_term_equations import classify_maturity
from assistant.agent.slots import extract_growth_slots
from assistant.agent.router import route_message
from assistant.agent.intents import (  # noqa: F401
    AFFIRM_RE,
    ANALYZE_GROWTH_RE,
    BARE_SHOW_RE,
    CONCERN_RE,
    GROWTH_COMPUTE_RE,
    HELP_RE,
    HISTORY_RE,
    MEASURE_EXPLAIN_RE,
    MEDICAL_FOLLOWUP_RE,
    MEDICAL_RE,
    SCREEN_RE,
    SHOW_CHART_RE,
    TALK_WORRY_RE,
    classify_intent,
)

log = logging.getLogger(__name__)


def _bounds() -> dict:
    return clinical_bounds()


# Heuristics for reading a stored `weeks` column that has no unit attached.
# WHO rows store postnatal weeks of life; INTERGROWTH rows store postmenstrual age.
# Below this age a weeks value is unambiguous: it cannot be a plausible PMA.
AMBIGUOUS_WEEKS_MIN = float(_bounds().get("intergrowth_weeks_min", 27))
AMBIGUOUS_WEEKS_MAX = float(_bounds().get("intergrowth_weeks_max", 64))
# A postnatal reading this old means the row cannot be an INTERGROWTH PMA point.
LIFE_WEEKS_TODDLER_MONTHS = 10.0
# If the PMA reading lands this far below the life-weeks reading, the stored value
# was postnatal weeks (subtracting GA would report a much younger child).
PMA_IMPLAUSIBLE_RATIO = 0.75

TASK_INSTRUCTION = """
You are an expert in composing functions for a pediatric parent assistant.
You are given a question and a set of possible functions.
Based on the question, make one or more function/tool calls to achieve the purpose.
If none of the functions can be used, return an empty tool_calls list.
If required parameters are missing, do not invent them — omit the call.
Never invent growth percentiles, z-scores, or screening scores; always use tools.
""".strip()

FORMAT_INSTRUCTION = """
The output MUST strictly adhere to the following JSON format, and NO other text MUST be included.
{
 "tool_calls": [
 {"name": "func_name1", "arguments": {"argument1": "value1", "argument2": "value2"}}
]
}
""".strip()


def _infer_last_topic(intents: set[str], query: str = "") -> str:
    """Free-text topic slug for multi-turn continuation (not a closed enum)."""
    from assistant.agent.intents import topic_slug_from_query

    q = query or ""
    if "medical" in intents or "screening" in intents:
        return topic_slug_from_query(q)
    if "growth_analysis" in intents:
        return "growth_analysis"
    if "growth" in intents:
        return "growth"
    if "history" in intents:
        return "history"
    if "help" in intents:
        return "help"
    if "reassure" in intents:
        return "reassure"
    if "chat" in intents:
        return "chat"
    return topic_slug_from_query(q) or next(iter(sorted(intents)), "chat")


DAYS_PER_MONTH = 365.25 / 12


def _age_months_from_dob(date_of_birth: str | None) -> float | None:
    """
    Chronological age in months from ISO date_of_birth (YYYY-MM-DD).

    Uses the UTC date so age matches the UTC timestamps written to the DBs.
    A future date_of_birth yields None rather than a negative age.
    """
    if not date_of_birth:
        return None
    from datetime import datetime, timezone

    raw = str(date_of_birth).strip()[:10]
    try:
        dob = datetime.strptime(raw, "%Y-%m-%d").date()
    except (TypeError, ValueError):
        return None
    today = datetime.now(timezone.utc).date()
    if dob > today:
        return None
    return max(0.0, (today - dob).days / DAYS_PER_MONTH)


def resolve_known_age_months(slots: dict, child_id: str | None, db: ChildMemoryDB) -> float | None:
    """
    Chronological age (months) for medical/feeding/speech RAG.

    Prefer last plotted chronological age, explicit age_months, then DOB.
    Weeks are ambiguous (PMA vs life-weeks); only convert with chart awareness and
    never treat postnatal weeks-of-life (age_months * wpm) as PMA minus GA.
    """
    wpm = weeks_per_month()

    # 1) Last successful growth plot (authoritative chronological months)
    if slots.get("last_age_months") is not None:
        try:
            return float(slots["last_age_months"])
        except (TypeError, ValueError):
            pass

    # 2) Explicit chronological months from parent / prior slots
    if slots.get("age_months") is not None:
        try:
            return float(slots["age_months"])
        except (TypeError, ValueError):
            pass

    child: dict = {}
    if child_id:
        child = db.get_child(child_id) or {}
        dob_age = _age_months_from_dob(child.get("date_of_birth"))
        if dob_age is not None:
            return float(dob_age)

        hist = db.growth_history(child_id) or []
        if hist:
            last = hist[-1]
            if last.get("age_months") is not None:
                try:
                    return float(last["age_months"])
                except (TypeError, ValueError):
                    pass

    weeks = slots.get("weeks")
    ga = slots.get("gestational_age_weeks")
    if ga is None and child.get("gestational_age_weeks") is not None:
        ga = child.get("gestational_age_weeks")
    chart = slots.get("last_chart_standard") or slots.get("chart_standard")

    preterm_threshold = float(_bounds().get("preterm_ga_threshold_weeks", 37))
    if weeks is not None:
        try:
            w = float(weeks)
        except (TypeError, ValueError):
            w = None
        if w is not None:
            # INTERGROWTH PMA → chronological only when chart says preterm PMA
            if chart == "intergrowth_preterm" and ga is not None:
                try:
                    return max(0.0, w - float(ga)) / wpm
                except (TypeError, ValueError):
                    pass
            # WHO / term / unknown: weeks are postnatal or near-term PMA band
            if chart == "who_term" or (ga is not None and float(ga) >= preterm_threshold):
                return float(_term_age_months_from_weeks(w, float(ga) if ga is not None else None))
            # Ambiguous weeks without preterm chart: do NOT subtract GA (avoids
            # turning 59 life-weeks ≈ 13.5m into ~7m when GA≈28–32).
            if w < AMBIGUOUS_WEEKS_MIN:
                return w / wpm
            return float(_term_age_months_from_weeks(w, None))

    if child_id:
        hist = db.growth_history(child_id) or []
        if hist:
            last = hist[-1]
            if last.get("weeks") is not None:
                cga = child.get("gestational_age_weeks")
                try:
                    lw = float(last["weeks"])
                except (TypeError, ValueError):
                    return None
                # Stored WHO points use life-weeks (age_months*wpm); INTERGROWTH uses PMA.
                # Without an age_months column, prefer life-weeks interpretation when
                # weeks/wpm is a plausible toddler age and PMA−GA would be much younger.
                life_m = lw / wpm
                if cga is not None:
                    pma_chrono = max(0.0, lw - float(cga)) / wpm
                    # If life-weeks reading is ~12m+ and PMA reading is << that, stored
                    # weeks were almost certainly postnatal (WHO save), not PMA.
                    if (
                        life_m >= LIFE_WEEKS_TODDLER_MONTHS
                        and pma_chrono < life_m * PMA_IMPLAUSIBLE_RATIO
                    ):
                        return life_m
                    if slots.get("last_chart_standard") == "intergrowth_preterm" or (
                        float(cga) < preterm_threshold
                        and AMBIGUOUS_WEEKS_MIN <= lw <= AMBIGUOUS_WEEKS_MAX
                    ):
                        return pma_chrono
                return float(_term_age_months_from_weeks(lw, float(cga) if cga is not None else None))
    return None


def convert_to_xlam_tool(tools: list[dict]) -> list[dict]:
    out = []
    for t in tools:
        out.append(
            {
                "name": t["name"],
                "description": t["description"],
                "parameters": {k: v for k, v in t["parameters"].get("properties", {}).items()},
            }
        )
    return out


def build_xlam_prompt(query: str, tools: list[dict] | None = None) -> str:
    tools = tools or TOOL_SPECS
    xlam_tools = convert_to_xlam_tool(tools)
    prompt = f"[BEGIN OF TASK INSTRUCTION]\n{TASK_INSTRUCTION}\n[END OF TASK INSTRUCTION]\n\n"
    prompt += f"[BEGIN OF AVAILABLE TOOLS]\n{json.dumps(xlam_tools)}\n[END OF AVAILABLE TOOLS]\n\n"
    prompt += f"[BEGIN OF FORMAT INSTRUCTION]\n{FORMAT_INSTRUCTION}\n[END OF FORMAT INSTRUCTION]\n\n"
    prompt += f"[BEGIN OF QUERY]\n{query}\n[END OF QUERY]\n\n"
    return prompt


class XLAMToolCaller:
    """Salesforce/xLAM-1b-fc-r — the only tool-calling model allowed."""

    def __init__(self, enabled: bool = False):
        self.enabled = enabled
        self._model = None
        self._tok = None
        self.model_id = get_settings().nestling_tool_model

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if device == "cuda" else torch.float32
        kwargs = {
            "trust_remote_code": True,
            "torch_dtype": dtype,
        }
        if device == "cuda":
            kwargs["device_map"] = "auto"
        else:
            kwargs["device_map"] = "cpu"
        self._model = AutoModelForCausalLM.from_pretrained(self.model_id, **kwargs)
        self.enabled = True

    def propose(self, query: str, *, user_message: str | None = None, slots: dict | None = None) -> list[dict]:
        current = user_message if user_message is not None else query
        intents = classify_intent(current)
        if not self.enabled or self._model is None:
            return rule_based_tool_calls(query, slots=slots, user_message=current, intents=intents)
        # For growth/screening intents, give xLAM a clean focused query + slots
        focused = current
        if slots:
            focused = f"{current}\nKnown slots: {json.dumps({k: slots[k] for k in slots if k != 'want_overlay'})}"
        content = build_xlam_prompt(focused)
        messages = [{"role": "user", "content": content}]
        # transformers>=5 may return BatchEncoding; tokenize then **kwargs to generate
        prompt = self._tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        device = next(self._model.parameters()).device
        # Avoid BatchEncoding.device (raises empty AttributeError on transformers 5.x)
        enc = self._tok(prompt, return_tensors="pt")
        input_ids = enc["input_ids"].to(device)
        attention_mask = enc["attention_mask"].to(device)
        outputs = self._model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=get_settings().nestling_tool_max_new_tokens,
            do_sample=False,
            eos_token_id=self._tok.eos_token_id,
        )
        in_len = input_ids.shape[1]
        text = self._tok.decode(outputs[0][in_len:], skip_special_tokens=True)
        calls = parse_tool_calls(text)
        # Safety: never run growth tools unless this turn intends growth
        if "growth" not in intents:
            calls = [c for c in calls if c.get("name") not in {"growth_percentile", "overlay_growth_on_chart"}]
        if "screening" not in intents:
            calls = [c for c in calls if c.get("name") not in {"score_asq_questionnaire", "score_mchat"}]
        if "history" not in intents:
            calls = [c for c in calls if c.get("name") != "get_child_summary"]
        # Merge remembered slots into growth tool args
        if slots:
            for c in calls:
                if c.get("name") in {"growth_percentile", "overlay_growth_on_chart"}:
                    args = dict(c.get("arguments") or {})
                    for k in ("sex", "measure", "weeks", "value", "child_id"):
                        if k in slots and k not in args:
                            args[k] = slots[k]
                    c["arguments"] = args
        return calls


def _tool_calls_from_json(raw: str) -> list[dict] | None:
    """Parsed tool_calls list, or None when `raw` is not a tool-call object."""
    try:
        data = json.loads(raw)
    except (json.JSONDecodeError, TypeError):
        return None
    if not isinstance(data, dict):
        return None
    calls = data.get("tool_calls")
    return calls if isinstance(calls, list) else []


def parse_tool_calls(text: str) -> list[dict]:
    text = (text or "").strip()
    calls = _tool_calls_from_json(text)
    if calls is not None:
        return calls
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return []
    return _tool_calls_from_json(m.group(0)) or []


def interpret_track_status(track_status: str | None, centile: float | None, *, fa: bool = False) -> str:
    """Parent-friendly reading of a growth point — not a diagnosis."""
    status = (track_status or "").lower()
    c = None
    try:
        if centile is not None:
            c = float(centile)
    except (TypeError, ValueError):
        c = None

    if status in {"within_10_90"} or (c is not None and 10 <= c <= 90):
        if fa:
            return (
                "از نظر صدک رشد، فعلاً در بازه معمول (حدود صدک ۱۰ تا ۹۰) است — "
                "یعنی مسیر رشدش در محدوده رایج همسالان است. این جایگزین ویزیت پزشک نیست."
            )
        return (
            "On the growth chart, this point sits in the usual range (about the 10th–90th centile) — "
            "so yes, that looks like a typical track for now. This is not a diagnosis."
        )
    if status in {"outer_centile_monitor"} or (c is not None and ((3 <= c < 10) or (90 < c <= 97))):
        if fa:
            return (
                "نزدیک لبه‌های نمودار است (بیرون بازه ۱۰–۹۰، اما هنوز شدید نیست). "
                "ارزش پیگیری در ویزیت بعدی را دارد؛ نگران فوری معمولاً لازم نیست مگر علائم دیگر باشد."
            )
        return (
            "This point is toward the outer part of the chart (outside 10–90, but not extreme). "
            "Worth watching at the next visit; usually not an emergency unless there are other worries."
        )
    if status in {"below_3rd_investigate"} or (c is not None and c < 3):
        if fa:
            return (
                "زیر صدک ۳ است — این را با متخصص کودکان در میان بگذارید تا مسیر رشد را بررسی کنند. "
                "من تشخیص نمی‌گذارم."
            )
        return (
            "This is below the 3rd centile — please discuss it with your pediatrician so they can review the growth pattern. "
            "I cannot diagnose."
        )
    if status in {"above_97th_investigate"} or (c is not None and c > 97):
        if fa:
            return (
                "بالای صدک ۹۷ است — بهتر است با متخصص کودکان بررسی شود. من تشخیص نمی‌گذارم."
            )
        return (
            "This is above the 97th centile — best reviewed with your pediatrician. I cannot diagnose."
        )
    if fa:
        return "نتیجه رشد را دارم؛ اگر عدد یا نمودار را بفرستید دقیق‌تر توضیح می‌دهم."
    return "I have a growth result on file — send the number or chart details if you want a clearer read."


def latest_growth_snapshot(db: ChildMemoryDB, child_id: str | None, slots: dict) -> dict | None:
    """Prefer last computed result in session slots; else child's latest saved growth."""
    if slots.get("last_centile") is not None or slots.get("last_track_status"):
        return {
            "measure": slots.get("last_measure") or slots.get("measure"),
            "value": slots.get("last_value") if slots.get("last_value") is not None else slots.get("value"),
            "centile": slots.get("last_centile"),
            "z_score": slots.get("last_z_score"),
            "track_status": slots.get("last_track_status"),
            "age_months": slots.get("last_age_months"),
            "weeks": slots.get("weeks"),
            "chart_standard": slots.get("last_chart_standard") or slots.get("chart_standard"),
            "sex": slots.get("sex"),
        }
    if not child_id:
        return None
    hist = db.growth_history(child_id) or []
    if not hist:
        return None
    g = hist[-1]
    return {
        "measure": g.get("measure"),
        "value": g.get("value"),
        "centile": g.get("centile"),
        "z_score": g.get("z_score"),
        "track_status": g.get("track_status"),
        "weeks": g.get("weeks"),
        "sex": slots.get("sex"),
        "chart_standard": slots.get("chart_standard"),
    }


def hydrate_slots_from_child(db: ChildMemoryDB, child_id: str | None, slots: dict) -> dict:
    """Fill missing growth slots from the selected child's latest saved measurement."""
    if not child_id:
        return slots
    child = db.get_child(child_id) or {}
    if child.get("sex") and "sex" not in slots:
        slots["sex"] = child["sex"]
    if child.get("gestational_age_weeks") is not None:
        slots.setdefault("gestational_age_weeks", child["gestational_age_weeks"])
        maturity = classify_maturity(child["gestational_age_weeks"])
        if maturity == "term":
            slots.setdefault("chart_standard", "who_term")
        elif maturity == "preterm":
            slots.setdefault("chart_standard", "intergrowth_preterm")

    dob_age = _age_months_from_dob(child.get("date_of_birth"))
    if dob_age is not None:
        slots.setdefault("age_months", float(dob_age))
        slots.setdefault("last_age_months", float(dob_age))

    history = db.growth_history(child_id) or []
    if not history:
        return slots
    by_measure: dict[str, dict] = {}
    for g in history:
        by_measure[str(g.get("measure"))] = g
    preferred = None
    if slots.get("measure") and slots["measure"] in by_measure:
        preferred = by_measure[slots["measure"]]
    else:
        preferred = by_measure.get("weight") or history[-1]
    if not preferred:
        return slots
    slots.setdefault("measure", preferred.get("measure"))
    if preferred.get("value") is not None:
        slots.setdefault("value", float(preferred["value"]))
    if preferred.get("weeks") is not None:
        slots.setdefault("weeks", float(preferred["weeks"]))
    if preferred.get("age_months") is not None:
        slots.setdefault("age_months", float(preferred["age_months"]))
        slots.setdefault("last_age_months", float(preferred["age_months"]))
    elif preferred.get("weeks") is not None and "age_months" not in slots:
        # Prefer chronological months; avoid GA-subtracting WHO life-weeks.
        try:
            lw = float(preferred["weeks"])
            wpm = weeks_per_month()
            cga = (
                float(slots["gestational_age_weeks"])
                if slots.get("gestational_age_weeks") is not None
                else (
                    float(child["gestational_age_weeks"])
                    if child.get("gestational_age_weeks") is not None
                    else None
                )
            )
            life_m = lw / wpm
            preterm_threshold = float(_bounds().get("preterm_ga_threshold_weeks", 37))
            if (
                cga is not None
                and float(cga) < preterm_threshold
                and AMBIGUOUS_WEEKS_MIN <= lw <= AMBIGUOUS_WEEKS_MAX
            ):
                pma_chrono = max(0.0, lw - float(cga)) / wpm
                # WHO saves store life-weeks; INTERGROWTH stores PMA.
                if (
                    life_m >= LIFE_WEEKS_TODDLER_MONTHS
                    and pma_chrono < life_m * PMA_IMPLAUSIBLE_RATIO
                ):
                    chrono = life_m
                else:
                    chrono = pma_chrono
            else:
                chrono = float(_term_age_months_from_weeks(lw, cga))
            slots["age_months"] = chrono
            slots.setdefault("last_age_months", chrono)
        except (TypeError, ValueError):
            pass
    slots["want_overlay"] = True
    return slots


def rule_based_tool_calls(
    query: str,
    slots: dict | None = None,
    *,
    user_message: str | None = None,
    intents: set[str] | None = None,
) -> list[dict]:
    """
    Deterministic router. Growth tools fire only when the *current* turn
    intends growth computation — never because prior chat mentioned weight.
    """
    current = user_message if user_message is not None else query
    intents = intents if intents is not None else classify_intent(current)
    slots = dict(slots or {})
    slots.update(extract_growth_slots(current))
    q = current.lower()
    calls: list[dict] = []

    if "screening" in intents:
        if "m-chat" in q or "mchat" in q or "autism screen" in q:
            m = re.search(r"ANSWERS:\s*(\{.*\})", current, re.S)
            if m:
                calls.append({"name": "score_mchat", "arguments": {"answers": json.loads(m.group(1))}})
        if "asq" in q and "ANSWERS:" in current:
            m = re.search(r"ANSWERS:\s*(\{.*\})", current, re.S)
            if m:
                calls.append(
                    {
                        "name": "score_asq_questionnaire",
                        "arguments": {"domain_answers": json.loads(m.group(1))},
                    }
                )

    if "history" in intents and slots.get("child_id"):
        calls.append({"name": "get_child_summary", "arguments": {"child_id": slots["child_id"]}})

    if "growth" in intents:
        # Prefer overlay whenever we have a measurement — agent calls tools for the parent.
        has_age = "weeks" in slots or "age_months" in slots
        growth_ready = "sex" in slots and "measure" in slots and has_age
        if growth_ready and "value" in slots:
            args = {
                "sex": slots["sex"],
                "measure": slots["measure"],
                "value": slots["value"],
            }
            # Prefer chronological age_months. Only pass weeks as PMA when months absent
            # (slot weeks often = age_months * wpm and must not be treated as PMA).
            if "age_months" in slots:
                args["age_months"] = slots["age_months"]
            elif "weeks" in slots:
                args["weeks"] = slots["weeks"]
            if slots.get("gestational_age_weeks") is not None:
                args["gestational_age_weeks"] = slots["gestational_age_weeks"]
            if slots.get("chart_standard"):
                args["chart_standard"] = slots["chart_standard"]
            if slots.get("child_id"):
                args["child_id"] = slots["child_id"]
            # Default: always produce chart overlay for parents (no need to say "overlay")
            calls.append({"name": "overlay_growth_on_chart", "arguments": args})
        elif growth_ready:
            args = {
                "sex": slots["sex"],
                "measure": slots["measure"],
            }
            if "age_months" in slots:
                args["age_months"] = slots["age_months"]
            elif "weeks" in slots:
                args["weeks"] = slots["weeks"]
            if slots.get("gestational_age_weeks") is not None:
                args["gestational_age_weeks"] = slots["gestational_age_weeks"]
            if slots.get("chart_standard"):
                args["chart_standard"] = slots["chart_standard"]
            calls.append({"name": "growth_percentile", "arguments": args})
    return calls


def _parse_growth_args(query: str) -> dict | None:
    slots = extract_growth_slots(query)
    if "sex" in slots and "measure" in slots and "weeks" in slots:
        return slots
    return None


class ParentAssistant:
    def __init__(
        self,
        db: ChildMemoryDB | None = None,
        chat_memory: ChatMemory | None = None,
        use_xlam: bool | None = None,
        use_llm: bool | None = None,
        *,
        use_pleias: bool | None = None,
    ):
        self.db = db or ChildMemoryDB()
        self.chat_memory = chat_memory or ChatMemory()
        self.medical = MedicalRAG()
        self.child_rag = ChildRAG()
        # xLAM loads only when NESTLING_LOAD_MODELS=1.
        # Generative RAG uses the vLLM/Qwen sidecar when NESTLING_USE_LLM is on.
        load = bool(get_settings().nestling_load_models)
        if use_llm is None:
            if use_pleias is not None:
                use_llm = bool(use_pleias)
            else:
                from assistant.llm.qwen_client import llm_enabled

                use_llm = llm_enabled()
        self.use_llm = bool(use_llm)
        self.use_pleias = self.use_llm  # backward compat for callers/tests
        self.tool_caller = XLAMToolCaller(enabled=False)
        if use_xlam if use_xlam is not None else load:
            self.tool_caller.load()
        self.medical.load()
        # Rebuild if volume index is missing curated feeding guidance (stale named volume).
        try:
            feed_docs = sum(
                1 for d in self.medical.store.docs if "feeding" in str(d.get("id", "")).lower()
            )
            if feed_docs == 0:
                chunks = KNOWLEDGE_DIR / "chunks.json"
                if chunks.is_file():
                    n = self.refresh_medical_index()
                    if n:
                        self.medical.load()
        except Exception as exc:
            log.warning("Could not refresh the medical RAG index at startup: %s", exc)
        self.child_rag.load()

    def refresh_medical_index(self):
        return self.medical.build_from_chunks()

    def refresh_child_index(self, child_id: str):
        docs = self.db.timeline_documents(child_id)
        self.child_rag.reindex_child(docs)
        return len(docs)

    def ask_medical(self, query: str) -> dict:
        return self.medical.answer(query, use_llm=self.use_llm)

    def ask_child(self, child_id: str, query: str) -> dict:
        return self.child_rag.answer(query, child_id=child_id, use_llm=self.use_llm)

    def analyze_parent_photo(
        self,
        image_bytes: bytes,
        *,
        mime: str = "image/png",
        prompt: str = "",
        ui_lang: str | None = None,
    ) -> dict:
        """Vision + RAG path for parent-sent photos. Never diagnoses."""
        from assistant.llm.qwen_client import get_qwen, llm_enabled
        from assistant.parent_voice import medical_chat_answer
        from assistant.runtime_translate import ensure_pure_lang, translate_for_models

        caption = (prompt or "").strip()
        neutral = "What can you tell me about this photo?"
        reply_lang, en_prompt = translate_for_models(
            caption or neutral,
            ui_lang=ui_lang,
        )
        rag_q = en_prompt
        rag = self.ask_medical(rag_q)
        context = rag.get("context") or rag.get("answer") or ""
        vision_text = ""
        mode = "rag_only"
        model = rag.get("model")
        if llm_enabled():
            try:
                client = get_qwen()
                if client.vision_ready:
                    user_q = (
                        f"Parent note: {caption}. " if caption else "Parent sent a photo without text. "
                    ) + (
                        "Describe likely benign possibilities and red flags for urgent care. "
                        "Educational guidance only."
                    )
                    grounded_prompt = (
                        f"{user_q}\n\nCare notes context:\n{context}\n\n"
                        "Keep response concise and parent-friendly."
                    )
                    vision_text = client.analyze_image(
                        image_bytes=image_bytes,
                        prompt=grounded_prompt,
                        mime=mime,
                        max_tokens=get_settings().llm_max_tokens_vision,
                    )
                    mode = "vision+rag"
                    model = get_settings().nestling_vision_model
            except Exception as exc:
                log.warning("Vision analysis failed, falling back to RAG: %s", exc)
                vision_text = ""
                mode = f"rag_fallback:{exc}"
        if not vision_text:
            if caption:
                vision_text = (
                    "Vision model is not ready, so I am using your description only "
                    f"({caption}) here is guidance from our care notes. "
                    f"{rag.get('answer') or ''}"
                )
                mode = "caption+rag_extractive"
            else:
                vision_text = (
                    "I received your photo. Vision service is not ready right now, so please "
                    "describe what you see or what you would like help with. "
                    f"Here is calm guidance from our care notes. {rag.get('answer') or ''}"
                )
                mode = "rag_no_vision"
        spoken = medical_chat_answer(vision_text, fa=(reply_lang == "fa"))
        if reply_lang == "fa":
            spoken = ensure_pure_lang(spoken, "fa")
        else:
            spoken = ensure_pure_lang(spoken, "en")
        return {
            "ok": True,
            "reply": spoken,
            "reply_lang": reply_lang,
            "mode": mode,
            "model": model,
            "medical_rag": rag,
            "disclaimer": "Educational only — not a diagnosis. Seek urgent care for breathing trouble, lethargy, or rapidly spreading infection signs.",
        }

    def run_tools(
        self,
        query: str,
        slots: dict | None = None,
        *,
        user_message: str | None = None,
        intents: set[str] | None = None,
    ) -> dict:
        current = user_message if user_message is not None else query
        intents = intents if intents is not None else classify_intent(current)
        if self.tool_caller.enabled:
            calls = self.tool_caller.propose(query, user_message=current, slots=slots)
        else:
            calls = rule_based_tool_calls(
                query, slots=slots, user_message=current, intents=intents
            )
        results = []
        for call in calls:
            name = call["name"]
            args = call.get("arguments", {})
            results.append(
                {"name": name, "arguments": args, "result": dispatch_tool(name, args, db=self.db)}
            )
        return {
            "query": query,
            "tool_calls": results,
            "tool_model": self.tool_caller.model_id if self.tool_caller.enabled else "deterministic_router",
            "intents": sorted(intents),
        }

    def record_growth_and_overlay(
        self,
        child_id: str,
        sex: str,
        measure: str,
        weeks: float | None,
        value: float,
        age_months: float | None = None,
    ) -> dict:
        child = self.db.get_child(child_id) or {}
        ga = child.get("gestational_age_weeks")
        overlay = dispatch_tool(
            "overlay_growth_on_chart",
            {
                "sex": sex,
                "measure": measure,
                "weeks": weeks,
                "value": value,
                "age_months": age_months,
                "gestational_age_weeks": ga,
                "child_id": child_id,
            },
            db=self.db,
        )
        if overlay.get("ok"):
            store_weeks = overlay.get("weeks")
            if store_weeks is None and overlay.get("age_months") is not None:
                store_weeks = float(overlay["age_months"]) * weeks_per_month()
            self.db.add_growth(
                child_id,
                weeks=store_weeks if store_weeks is not None else (weeks or 0),
                measure=overlay["measure"],
                value=value,
                z_score=overlay.get("z_score"),
                centile=overlay.get("centile"),
                track_status=overlay.get("track_status"),
                age_months=(
                    float(overlay["age_months"])
                    if overlay.get("age_months") is not None
                    else age_months
                ),
            )
            self.refresh_child_index(child_id)
        return overlay

    def run_asq_session(self, child_id: str, age_months: int, domain_answers: dict) -> dict:
        result = dispatch_tool("score_asq_questionnaire", {"domain_answers": domain_answers}, db=self.db)
        if result.get("ok"):
            self.db.add_screening(child_id, "ASQ", domain_answers, result, age_months=age_months)
            self.refresh_child_index(child_id)
        asq_path = EN_DIR / "asq" / f"{age_months}m.json"
        return {
            "instrument": "ASQ",
            "age_months": age_months,
            "result": result,
            "questionnaire_available": asq_path.exists(),
            "parent_report": self._asq_parent_report(result) if result.get("ok") else result.get("detail"),
        }

    def run_mchat_session(self, child_id: str, answers: dict) -> dict:
        result = dispatch_tool("score_mchat", {"answers": answers}, db=self.db)
        if result.get("ok"):
            self.db.add_screening(child_id, "M-CHAT-R", answers, result)
            self.refresh_child_index(child_id)
        return {
            "instrument": "M-CHAT-R",
            "result": result,
            "parent_report": result.get("summary") or result.get("detail"),
        }

    def _asq_parent_report(self, result: dict) -> str:
        lines = ["ASQ results:"]
        for dom, res in result.get("domains", {}).items():
            flag = "BELOW cutoff — discuss with clinician" if res["below_cutoff"] else "above cutoff"
            lines.append(f"- {dom}: {res['total']}/{res['max']} ({flag})")
        if result.get("needs_referral"):
            lines.append(
                "One or more domains are below the cutoff. Please follow up with your pediatric clinician."
            )
        else:
            lines.append("No domain scored below the cutoff in this session.")
        return "\n".join(lines)

    def start_session(self, child_id: str | None = None) -> str:
        return self.chat_memory.create_session(child_id=child_id)

    def _children_named_in(
        self, owner_user_id: str | None, *messages: str
    ) -> list[dict]:
        """
        Children whose stored name appears in any of `messages`.

        The roster is read from the DB every turn rather than matched against a
        list in code, so adding a child needs no rule change. Matching both the
        raw and the translated message matters because transliteration mangles
        Persian names ("مونیکا" does not reliably come back as "Monica").
        """
        min_chars = get_settings().nestling_child_name_min_chars
        haystacks = [(m or "").lower() for m in messages if m]
        hits: dict[str, dict] = {}
        for row in self.db.list_children(owner_user_id=owner_user_id):
            name = str(row.get("name") or "").strip()
            if len(name) < min_chars:
                continue
            if not any(name.lower() in hay for hay in haystacks):
                continue
            for match in self.db.find_children_by_name(name, owner_user_id=owner_user_id):
                hits[match["child_id"]] = match
        return list(hits.values())

    def chat(
        self,
        session_id: str,
        user_message: str,
        child_id: str | None = None,
        ui_lang: str | None = None,
        owner_user_id: str | None = None,
    ) -> dict:
        """
        Full multi-turn chat with persistent memory.
        Persian messages are translated to English for the agent; replies return in the parent language.
        """
        session = self.chat_memory.get_session(session_id)
        if not session:
            session_id = self.chat_memory.create_session(child_id=child_id)
            session = self.chat_memory.get_session(session_id)

        if child_id:
            self.chat_memory.set_child(session_id, child_id)
        elif session.get("child_id"):
            child_id = session["child_id"]

        reply_lang, en_message = translate_for_models(user_message, ui_lang=ui_lang)
        self.chat_memory.add_message(session_id, "user", user_message)

        # A parent can name the child instead of picking one in the UI
        # ("how is Monica doing?"). Scope to the session owner so a name can
        # never reach another account's record.
        owner_user_id = owner_user_id or (session.get("owner_user_id") or None)
        ambiguous_children: list[dict] = []
        if not child_id:
            named = self._children_named_in(owner_user_id, user_message, en_message)
            if len(named) == 1:
                child_id = named[0]["child_id"]
                self.chat_memory.set_child(session_id, child_id)
            elif len(named) > 1:
                # Never guess between siblings — the reply asks which one.
                ambiguous_children = named

        new_slots = extract_growth_slots(en_message)
        for k, v in extract_growth_slots(user_message).items():
            new_slots.setdefault(k, v)

        if child_id:
            new_slots["child_id"] = child_id
            child = self.db.get_child(child_id) or {}
            if child.get("gestational_age_weeks") is not None:
                new_slots["gestational_age_weeks"] = child["gestational_age_weeks"]
                maturity = classify_maturity(child["gestational_age_weeks"])
                if maturity == "term":
                    new_slots.setdefault("chart_standard", "who_term")
                elif maturity == "preterm":
                    new_slots.setdefault("chart_standard", "intergrowth_preterm")
            if child.get("sex") and "sex" not in new_slots:
                new_slots["sex"] = child["sex"]

        session_slots = dict(session.get("slots") or {})
        if child_id:
            session_slots.setdefault("child_id", child_id)
        slots = self.chat_memory.merge_slots(session_id, new_slots)
        # Hybrid router (YAML rules + regex; optional LLM when sidecar is up)
        decision = route_message(
            user_message, prior_slots=session_slots, en_message=en_message
        )
        intents = set(decision.intents)
        for k, v in (decision.slots or {}).items():
            if v is not None and v != "" and k not in slots:
                slots = self.chat_memory.merge_slots(session_id, {k: v})
        # Resume an interrupted request. When the previous turn asked for a
        # missing slot, the user's answer is often a bare word ("boy", "term",
        # "37") that carries no intent of its own, so the router classifies it
        # as small talk and the original request is silently dropped. If we
        # were waiting on something and this turn supplied a slot, put the
        # pending intent back.
        pending_intent = session_slots.get("pending_intent")
        if pending_intent:
            supplied = {k for k, v in (new_slots or {}).items() if v not in (None, "")}
            supplied |= {k for k, v in (decision.slots or {}).items() if v not in (None, "")}
            if supplied and not (intents & {"medical", "screening"}):
                intents.add(pending_intent)
                intents.discard("chat")
                intents.discard("reassure")
                intents.discard("slot_update")

        # The collapses below can drop "history"; remember that the parent asked
        # for a child's status so a name-resolved turn still pulls the record.
        history_requested = "history" in intents

        show_chart = bool(SHOW_CHART_RE.search(user_message) or SHOW_CHART_RE.search(en_message))
        bare_show = bool(BARE_SHOW_RE.search(user_message) or BARE_SHOW_RE.search(en_message))
        if bare_show and child_id and (self.db.growth_history(child_id) or []):
            intents.add("growth")
            intents.discard("chat")
            intents.discard("history")
        # Prefer medical/concern over bare help if both somehow appear
        if "medical" in intents and "help" in intents:
            intents.discard("help")
        if "growth" in intents and "help" in intents:
            intents.discard("help")
        # Medical / screening always beats vague chat + leftover growth analysis
        if "medical" in intents or "screening" in intents:
            intents.discard("chat")
            intents.discard("reassure")
            if not show_chart and not bare_show and not GROWTH_COMPUTE_RE.search(
                en_message
            ) and not GROWTH_COMPUTE_RE.search(user_message):
                intents.discard("growth_analysis")
                intents.discard("growth")
        if "growth" in intents:
            intents.discard("slot_update")
            intents.discard("chat")
            intents.discard("history")
        if "reassure" in intents and "medical" not in intents:
            intents = {"reassure"}
        if "growth_analysis" in intents and "medical" not in intents:
            intents = {"growth_analysis"}
            # Analysis explains existing results — never replot unless they asked to show the chart
        # Never run chart tools on care/speech turns
        if (
            ("medical" in intents or "screening" in intents)
            and not show_chart
            and not bare_show
            and not SHOW_CHART_RE.search(user_message)
            and not SHOW_CHART_RE.search(en_message)
            and not GROWTH_COMPUTE_RE.search(user_message)
            and not GROWTH_COMPUTE_RE.search(en_message)
        ):
            intents.discard("growth")
            intents.discard("growth_analysis")
            intents.discard("chat")

        if history_requested and child_id and not (intents & {"growth", "medical", "screening"}):
            intents.add("history")
            intents.discard("chat")
            intents.discard("reassure")

        # Re-plot from saved child measurements when parent asks to show the chart
        if "growth" in intents and child_id and (
            show_chart
            or bare_show
            or slots.get("want_overlay")
        ):
            slots = hydrate_slots_from_child(self.db, child_id, dict(slots))
            slots = self.chat_memory.merge_slots(session_id, slots)

        turn_growth = extract_growth_slots(en_message)
        for k, v in extract_growth_slots(user_message).items():
            turn_growth.setdefault(k, v)
        new_measurement_turn = "value" in turn_growth

        # Inject rolling conversation memory into the tool/RAG turn (was unused before).
        _s = get_settings()
        mem_ctx = self.chat_memory.build_context(
            session_id,
            window=_s.nestling_history_window,
            summary_trigger=_s.nestling_summary_trigger_turns,
        )
        # Long-term memory: what earlier sessions already established about
        # THIS child. Scoped to the session owner so one account's history can
        # never surface in another's chat.
        owner_user_id = session.get("owner_user_id")
        child_ctx = ""
        if child_id and _s.nestling_child_memory_enabled:
            child_ctx = self.db.child_context_text(child_id, owner_user_id=owner_user_id)
        mem_ctx["child_context"] = child_ctx
        ctx_parts = []
        if child_ctx:
            ctx_parts.append(f"[CHILD_MEMORY]\n{child_ctx}")
        if mem_ctx.get("summary"):
            ctx_parts.append(f"[SESSION_SUMMARY]\n{mem_ctx['summary']}")
        if mem_ctx.get("recent_text"):
            ctx_parts.append(f"[RECENT_CHAT]\n{mem_ctx['recent_text']}")
        contextual_query = en_message
        if ctx_parts:
            contextual_query = "\n\n".join(ctx_parts) + f"\n\n[CURRENT_USER]\n{en_message}"

        # Persist typed facts from slots (provenance for later recall)
        for fact_key in (
            "sex",
            "measure",
            "weeks",
            "age_months",
            "value",
            "gestational_age_weeks",
            "chart_standard",
            "child_id",
        ):
            if slots.get(fact_key) is not None:
                self.chat_memory.upsert_fact(
                    session_id, fact_key, slots[fact_key], provenance="slot"
                )

        tool_block = self.run_tools(
            contextual_query,
            slots=slots,
            user_message=en_message,
            intents=intents,
        )
        for tc in tool_block["tool_calls"]:
            res = tc.get("result") or {}
            if res.get("ok") and tc["name"] in {"growth_percentile", "overlay_growth_on_chart"}:
                # Remember last plotted result for follow-up analysis questions
                age_m = res.get("age_months")
                slot_update = {
                    "last_centile": res.get("centile"),
                    "last_z_score": res.get("z_score"),
                    "last_track_status": res.get("track_status"),
                    "last_measure": res.get("measure"),
                    "last_value": res.get("value"),
                    "last_chart_standard": res.get("chart_standard"),
                    "want_overlay": True,
                }
                if age_m is not None:
                    slot_update["last_age_months"] = age_m
                    slot_update["age_months"] = age_m
                slots = self.chat_memory.merge_slots(session_id, slot_update)
                if child_id and res.get("value") is not None and new_measurement_turn:
                    store_weeks = res.get("weeks")
                    if store_weeks is None and age_m is not None:
                        store_weeks = float(age_m) * weeks_per_month()
                    self.db.add_growth(
                        child_id,
                        weeks=store_weeks if store_weeks is not None else 0.0,
                        measure=res["measure"],
                        value=res["value"],
                        z_score=res.get("z_score"),
                        centile=res.get("centile"),
                        track_status=res.get("track_status"),
                        age_months=float(age_m) if age_m is not None else None,
                    )
                    self.refresh_child_index(child_id)

        out: dict[str, Any] = {
            "session_id": session_id,
            "child_id": child_id,
            "slots": slots,
            "intents": sorted(intents),
            "tools": tool_block,
            "ui_lang": reply_lang,
            "models": {
                "tool_calling": self.tool_caller.model_id,
                "tool_calling_loaded": bool(self.tool_caller.enabled),
                "rag": get_settings().nestling_llm_model,
                "rag_loaded": bool(self.use_llm),
            },
        }

        # Snapshot prior thread before we overwrite topic for this turn.
        prior_med = str(slots.get("last_medical_query") or "").strip()
        prior_topic = str(slots.get("last_topic") or "").lower()

        if "growth_analysis" in intents:
            snap = latest_growth_snapshot(self.db, child_id, slots)
            if snap and (snap.get("centile") is not None or snap.get("track_status")):
                out["growth_analysis"] = snap
            else:
                out["growth_analysis"] = {"missing": True}

        if "medical" in intents:
            # Soft follow-ups retrieve against last_medical_query; new concerns use
            # the current message. No closed topic taxonomy.
            from assistant.agent.intents import (
                AFFIRM_RE as _AFFIRM,
                ANALYZE_GROWTH_RE as _ANALYZE,
                MEDICAL_FOLLOWUP_RE as _MED_FU,
                _is_soft_followup,
            )

            memory_parts: list[str] = []
            if mem_ctx.get("child_context"):
                memory_parts.append(f"[CHILD_MEMORY]\n{mem_ctx['child_context']}")
            if mem_ctx.get("summary"):
                memory_parts.append(f"[SESSION_SUMMARY]\n{mem_ctx['summary']}")
            if mem_ctx.get("recent_text"):
                recent = mem_ctx["recent_text"]
                recent_cap = _s.nestling_memory_recent_chars
                if len(recent) > recent_cap:
                    recent = recent[-recent_cap:]
                memory_parts.append(f"[RECENT_CHAT]\n{recent}")

            followup_hit = bool(_MED_FU.search(en_message) or _MED_FU.search(user_message or ""))
            affirm = bool(_AFFIRM.search(en_message) or _AFFIRM.search(user_message or ""))
            analyze = bool(_ANALYZE.search(en_message) or _ANALYZE.search(user_message or ""))
            soft = _is_soft_followup(
                en_message,
                followup_hit=followup_hit,
                affirm=affirm,
                analyze=analyze,
                prior_query=prior_med,
            )
            continuing = bool(prior_med and soft)

            # CURRENT_USER drives MedicalRAG — soft follow-ups keep prior domain via
            # last_medical_query; hard switches use the new user message alone.
            if continuing:
                user_parts = [f"{prior_med}\nFollow-up: {en_message}"]
                if user_message and user_message.strip() and user_message.strip() != en_message.strip():
                    user_parts.append(f"Follow-up (original): {user_message.strip()}")
            else:
                user_parts = [en_message]
                if user_message and user_message.strip() and user_message.strip() != en_message.strip():
                    user_parts.append(user_message.strip())

            known_age = resolve_known_age_months(slots, child_id, self.db)
            if known_age is not None:
                # Persist so follow-ups and feeding bands stay consistent
                slots = self.chat_memory.merge_slots(
                    session_id,
                    {"age_months": known_age, "last_age_months": known_age},
                )
            user_block = "\n".join(user_parts)
            if known_age is not None:
                user_block += (
                    f"\nKnown chronological age: {known_age:.1f} months. "
                    f"Use ONLY this age for care guidance. "
                    f"Never invent a different age from care-note titles (e.g. do not say "
                    f"7-month-old if this age is ~13 months)."
                )
            if slots.get("sex"):
                user_block += f"\nKnown child sex: {slots['sex']}."
            ga = slots.get("gestational_age_weeks")
            if ga is not None:
                try:
                    if float(ga) < 37:
                        user_block += (
                            f"\nBorn preterm at {float(ga):.0f} weeks GA; "
                            f"use chronological age above for age-based care guidance."
                        )
                except (TypeError, ValueError):
                    pass

            med_query = ""
            if memory_parts:
                med_query = "\n\n".join(memory_parts) + "\n\n"
            med_query += f"[CURRENT_USER]\n{user_block}"
            out["medical_rag"] = self.ask_medical(med_query)

        # Persist thread topic so the *next* turn can continue medical care.
        # Soft follow-ups keep prior last_medical_query; new concerns replace it.
        from assistant.agent.intents import (
            AFFIRM_RE as _AFFIRM2,
            ANALYZE_GROWTH_RE as _ANALYZE2,
            MEDICAL_FOLLOWUP_RE as _MED_FU2,
            _NON_CARE_TOPICS,
            _is_soft_followup as _soft_fu,
            topic_slug_from_query,
        )

        topic = _infer_last_topic(intents, en_message)
        thread_slots: dict[str, Any] = {
            "last_intents": sorted(intents),
        }
        prior_care = prior_topic if prior_topic not in _NON_CARE_TOPICS else ""
        fu_hit = bool(_MED_FU2.search(en_message) or _MED_FU2.search(user_message or ""))
        soft_turn = _soft_fu(
            en_message,
            followup_hit=fu_hit,
            affirm=bool(_AFFIRM2.search(en_message)),
            analyze=bool(_ANALYZE2.search(en_message)),
            prior_query=prior_med,
        )
        query_cap = _s.nestling_medical_query_chars
        if "medical" in intents or "screening" in intents:
            if soft_turn and prior_med and prior_care:
                thread_slots["last_topic"] = prior_care
                thread_slots["last_medical_query"] = prior_med[:query_cap]
            else:
                thread_slots["last_topic"] = topic_slug_from_query(en_message) or topic
                thread_slots["last_medical_query"] = en_message.strip()[:query_cap]
        elif topic == "chat" and prior_care:
            # Don't wipe an open care thread on a stray chat miss.
            thread_slots["last_topic"] = prior_care
            if prior_med:
                thread_slots["last_medical_query"] = prior_med[:query_cap]
        else:
            thread_slots["last_topic"] = topic
        slots = self.chat_memory.merge_slots(session_id, thread_slots)
        out["slots"] = slots
        # Prefer deterministic child summary tool; skip noisy child-RAG dump when summary exists
        if child_id and "history" in intents:
            has_summary = any(
                tc.get("name") == "get_child_summary" and (tc.get("result") or {}).get("ok")
                for tc in tool_block.get("tool_calls") or []
            )
            if not has_summary:
                out["child_rag"] = self.ask_child(child_id, en_message)

        missing = []
        needs_ga = any(
            (tc.get("result") or {}).get("needs_gestational_age")
            for tc in tool_block.get("tool_calls") or []
        )
        if needs_ga:
            missing.append("gestational_age_weeks (or say preterm/term)")
        if "growth" in intents:
            for key in ("sex", "measure"):
                if key not in slots:
                    missing.append(key)
            if "weeks" not in slots and "age_months" not in slots:
                missing.append("age (weeks or months)")
            if "value" not in slots:
                missing.append("value")
        if ambiguous_children:
            missing.append("child (which one?)")
        out["ambiguous_children"] = [
            {"child_id": c["child_id"], "name": c.get("name")} for c in ambiguous_children
        ]
        out["missing_slots"] = missing
        out["needs_gestational_age"] = needs_ga
        # Remember what we were mid-way through so the next turn can resume it
        # (see the pending_intent block above); clear it once nothing is
        # outstanding, otherwise later unrelated turns would be hijacked.
        if missing and ("growth" in intents or needs_ga):
            self.chat_memory.merge_slots(session_id, {"pending_intent": "growth"})
        elif not missing and slots.get("pending_intent"):
            self.chat_memory.clear_slots(session_id, ["pending_intent"])
            slots.pop("pending_intent", None)
        out["memory"] = {
            "summary": mem_ctx.get("summary") or "",
            "recent_turns": len((mem_ctx.get("recent_text") or "").splitlines()),
            "facts": list((mem_ctx.get("facts") or {}).keys()),
            "child_context": mem_ctx.get("child_context") or "",
        }
        out["explain_measure"] = bool(
            MEASURE_EXPLAIN_RE.search(user_message) or MEASURE_EXPLAIN_RE.search(en_message)
        )

        assistant_text = self._format_reply(out, intents=intents, reply_lang=reply_lang)
        if reply_lang == "fa":
            # Prefer already-Persian clinical summaries; MT only the remaining English chrome.
            from assistant.runtime_translate import ensure_pure_lang

            assistant_text = ensure_pure_lang(assistant_text, "fa")
        else:
            from assistant.runtime_translate import ensure_pure_lang

            assistant_text = ensure_pure_lang(assistant_text, "en")
        out["reply"] = assistant_text
        out["reply_lang"] = reply_lang

        public_tools = []
        for tc in tool_block.get("tool_calls") or []:
            res = tc.get("result") or {}
            public_tools.append(
                {
                    "name": tc.get("name"),
                    "summary": res.get("summary") or res.get("detail"),
                    "overlay_filename": res.get("overlay_filename"),
                    "centile": res.get("centile"),
                    "z_score": res.get("z_score"),
                    "track_status": res.get("track_status"),
                    "maturity": res.get("maturity"),
                    "maturity_label_fa": res.get("maturity_label_fa"),
                    "chart_standard": res.get("chart_standard"),
                    "ok": res.get("ok"),
                }
            )
        out["tool_results"] = public_tools

        self.chat_memory.add_message(
            session_id,
            "assistant",
            assistant_text,
            tool_calls=public_tools,
            meta={"slots": slots, "missing_slots": missing, "intents": sorted(intents)},
        )
        self._remember_for_child(
            child_id,
            owner_user_id=owner_user_id,
            user_message=en_message or user_message,
            intents=intents,
            slots=slots,
            session_id=session_id,
        )
        out["history"] = self.chat_memory.get_history(session_id)
        return out

    def _remember_for_child(
        self,
        child_id: str | None,
        *,
        owner_user_id: str | None,
        user_message: str,
        intents: set[str],
        slots: dict,
        session_id: str,
    ) -> None:
        """
        Write one salient turn to the child's durable timeline.

        Only clinically meaningful intents are kept: everything the parent says
        would otherwise accumulate as noise and crowd out the facts a later
        session actually needs. The note records the parent's own words — the
        assistant reply is largely derived from them plus retrieved guidance,
        so storing the concern is what makes the follow-up possible.
        """
        settings = get_settings()
        if not child_id or not settings.nestling_child_memory_enabled:
            return
        if not (intents & settings.child_memory_intents):
            return
        text = (user_message or "").strip()
        if not text:
            return
        topic = str(slots.get("last_topic") or "").strip()
        summary = f"{topic}: {text}" if topic and topic not in {"chat", ""} else text
        self.db.remember_note(
            child_id,
            summary,
            payload={
                "intents": sorted(intents),
                "session_id": session_id,
                "topic": topic or None,
            },
            owner_user_id=owner_user_id,
        )

    def _format_reply(
        self, out: dict, intents: set[str] | None = None, reply_lang: str = "en"
    ) -> str:
        from assistant.runtime_translate import translate_en_to_fa
        from assistant.parent_voice import (
            child_summary_chat,
            growth_plot_chat,
            medical_chat_answer,
            open_chat_turn,
        )

        intents = intents or set(out.get("intents") or [])
        parts: list[str] = []
        fa = reply_lang == "fa"
        measure_q = bool(out.get("explain_measure"))
        slots = out.get("slots") or {}
        has_growth = bool(
            slots.get("last_centile") is not None
            or slots.get("value") is not None
            or out.get("growth_analysis")
        )
        active_medical = bool(
            slots.get("last_medical_query")
            and str(slots.get("last_topic") or "").lower()
            not in {
                "growth",
                "growth_analysis",
                "help",
                "chat",
                "history",
                "reassure",
                "slot_update",
                "",
            }
        ) or bool(set(slots.get("last_intents") or []) & {"medical", "screening"})

        options = [str(c.get("name") or "") for c in (out.get("ambiguous_children") or [])]
        if options:
            joined = "، ".join(options) if fa else ", ".join(options)
            parts.append(
                f"چند فرزند با این نام دارید: {joined}. کدام‌یک را می‌گویید؟"
                if fa
                else f"I have more than one child by that name: {joined}. Which one do you mean?"
            )
        elif out.get("needs_gestational_age"):
            parts.append(
                "برای انتخاب درست نمودار رشد (نارس / طبیعی) سن بارداری هنگام تولد را بگویید "
                "(مثلاً ۳۲ هفته)، یا بنویسید نارس / طبیعی."
                if fa
                else "To pick the right growth chart (preterm vs term), please share gestational age "
                "at birth (e.g. 32 weeks), or say preterm / term."
            )
        elif "reassure" in intents:
            parts.append(
                "بله — با توجه به حرف‌تان، فعلاً جای نگرانی فوری به نظر نمی‌رسد. "
                "کنار شما هستم؛ اگر چیزی عوض شد بگویید."
                if fa
                else "Yes — from what you've shared, that usually sounds okay for now. "
                "I'm here with you; if something changes, just tell me."
            )
        elif "growth_analysis" in intents:
            snap = out.get("growth_analysis") or {}
            if snap.get("missing"):
                parts.append(
                    "هنوز نتیجه رشدی برای با هم دیدن نداریم. وزن/قد و سن را بفرستید "
                    "یا بگویید نمودار را نشان بدهم."
                    if fa
                    else "We don't have a growth result to look at together yet. "
                    "Send weight/length and age, or ask me to show the chart."
                )
            else:
                measure = snap.get("measure") or "growth"
                value = snap.get("value")
                centile = snap.get("centile")
                age_m = snap.get("age_months")
                lead = "بگذارید با هم نگاه کنیم: " if fa else "Let's look at it together: "
                bits = []
                if value is not None:
                    bits.append(f"{measure} {value}")
                if age_m is not None:
                    try:
                        bits.append(
                            f"حدود {float(age_m):.1f} ماهگی"
                            if fa
                            else f"around {float(age_m):.1f} months"
                        )
                    except (TypeError, ValueError):
                        pass
                if centile is not None:
                    try:
                        bits.append(
                            f"حدود صدک {float(centile):.0f}"
                            if fa
                            else f"about the {float(centile):.0f}th centile"
                        )
                    except (TypeError, ValueError):
                        pass
                if bits:
                    lead += ", ".join(bits) + ". "
                parts.append(
                    lead
                    + interpret_track_status(
                        snap.get("track_status"), snap.get("centile"), fa=fa
                    )
                )
                parts.append("سوال دیگری هم دارید؟" if fa else "Anything else on your mind about this?")
        elif "slot_update" in intents and "medical" not in intents and "growth" not in intents:
            remembered = ", ".join(
                f"{k}={v}"
                for k, v in slots.items()
                if k
                not in {
                    "child_id",
                    "gestational_age_weeks",
                    "chart_standard",
                    "want_overlay",
                    "last_centile",
                    "last_z_score",
                    "last_track_status",
                    "last_measure",
                    "last_value",
                    "last_age_months",
                    "last_chart_standard",
                }
            )
            if fa:
                parts.append(
                    "باشه، یادداشت کردم"
                    + (f" ({remembered})" if remembered else "")
                    + ". هر سوالی دارید بپرسید."
                )
            else:
                parts.append(
                    "Got it — I've noted that"
                    + (f" ({remembered})" if remembered else "")
                    + ". Ask me anything next."
                )
        elif "help" in intents:
            parts.append(
                "سلام! من نستلینگ هستم — مثل یک دستیار والدین کنار شما. "
                "می‌توانیم درباره رشد، تغذیه، خواب یا نگرانی‌های تکاملی حرف بزنیم "
                "و نمودار را با هم بکشیم. ساده بگویید چه چیزی ذهنتان را مشغول کرده."
                if fa
                else "Hi — I'm Nestling, here to chat with you about your little one. "
                "We can talk through growth, feeding, sleep, or developmental worries, "
                "and plot charts together. Just say what's on your mind."
            )
        elif (
            "chat" in intents
            and not out.get("tools", {}).get("tool_calls")
            and not out.get("medical_rag")
            and not active_medical
        ):
            parts.append(open_chat_turn(fa=fa, has_growth=has_growth))

        if "growth" in intents and not (out.get("tools") or {}).get("tool_calls"):
            if measure_q or "measure" in (out.get("missing_slots") or []):
                parts.append(
                    "برای نمودار بگویید وزن، قد یا دور سر را دارید — مثلاً: پسر، وزن، ۴۰ هفته، ۳٫۲ کیلو."
                    if fa
                    else "For the chart, tell me weight, length, or head — e.g. boy, weight, 40 weeks, 3.2 kg."
                )

        if (
            out.get("missing_slots")
            and "growth" in intents
            and not (out.get("tools") or {}).get("tool_calls")
        ):
            missing = out["missing_slots"]
            ask = [m for m in missing if not (measure_q and m == "measure")]
            if ask:
                hints = {
                    "sex": "boy or girl" if not fa else "پسر یا دختر",
                    "measure": "weight, length, or head" if not fa else "وزن، قد یا دور سر",
                    "age (weeks or months)": "age" if not fa else "سن",
                    "value": "the number" if not fa else "عدد اندازه",
                }
                pretty = [hints.get(m, m) for m in ask]
                parts.append(
                    ("برای ادامه هنوز لازم دارم: " + "؛ ".join(pretty) + ".")
                    if fa
                    else ("To continue I still need: " + "; ".join(pretty) + ".")
                )

        for tc in out.get("tools", {}).get("tool_calls", []):
            res = tc.get("result") or {}
            name = tc.get("name")
            if name in {"overlay_growth_on_chart", "growth_percentile"} and res.get("ok"):
                parts.append(growth_plot_chat(res, fa=fa))
            elif name == "get_child_summary" and res.get("ok"):
                # Pass the structured result, not res["summary"]: that string is
                # the agent-facing digest and leaks English labels and raw
                # overlay filenames into parent-facing replies.
                parts.append(child_summary_chat(res, fa=fa))
            elif res.get("ok") is False and res.get("detail"):
                # Avoid double-speaking when we already asked for gestational age
                if out.get("needs_gestational_age") and res.get("needs_gestational_age"):
                    continue
                parts.append(
                    ("متأسفم، مشکلی پیش آمد: " if fa else "Sorry, something went wrong: ")
                    + str(res["detail"])
                )
            elif name not in {
                "overlay_growth_on_chart",
                "growth_percentile",
                "get_child_summary",
            }:
                if fa and res.get("summary_fa"):
                    parts.append(res["summary_fa"])
                elif res.get("summary"):
                    parts.append(res["summary"])

        if out.get("medical_rag"):
            ans = out["medical_rag"].get("answer", "")
            mode = (out["medical_rag"].get("mode") or "").lower()
            if fa and ans:
                ans = translate_en_to_fa(ans)
            parts.append(
                medical_chat_answer(
                    ans,
                    fa=fa,
                    from_llm="openai" in mode or "llm" in mode,
                )
            )

        if "screening" in intents and "medical" in intents:
            parts.append(
                "اگر دوست دارید از بخش غربالگری، ASQ مناسب سن را هم می‌توانید شروع کنید — جایگزین پزشک نیست."
                if fa
                else "If you'd like, you can also try the age-matched ASQ in screening — it doesn't replace your clinician."
            )

        seen: set[str] = set()
        uniq: list[str] = []
        for p in parts:
            if p and p not in seen:
                seen.add(p)
                uniq.append(p)
        if not uniq:
            if active_medical and not out.get("medical_rag"):
                uniq.append(
                    "بله — درباره همان موضوع مراقبتی ادامه می‌دهیم. کمی بیشتر بگویید "
                    "(مثلاً چه کلمه‌هایی می‌گوید یا چه چیزی نگران‌تان کرده)."
                    if fa
                    else "Happy to keep going on that care topic — tell me a bit more "
                    "(for example what words she says, or what still worries you)."
                )
            else:
                uniq.append(open_chat_turn(fa=fa, has_growth=has_growth))
        return "\n\n".join(uniq)


    def handle(self, query: str, child_id: str | None = None) -> dict:
        sid = self.start_session(child_id=child_id)
        return self.chat(sid, query, child_id=child_id)

    def close(self) -> None:
        """Release SQLite handles (needed on Windows before deleting temp DBs)."""
        for name, closer in (("child_db", self.db.close), ("chat_memory", self.chat_memory.close)):
            try:
                closer()
            except Exception as exc:
                log.warning("Error closing %s: %s", name, exc)
