#!/usr/bin/env python3
"""
Parent assistant orchestrator.

Models (ONLY these two — user-specified):
  - Salesforce/xLAM-1b-fc-r  → tool / function calling
  - PleIAs/Pleias-RAG-1B     → RAG answers with citations

Deterministic clinical tools compute all growth/screening numbers (no LLM math).
Full chat session memory persists multi-turn slots + messages.
"""

from __future__ import annotations

import json
import os
import re
from typing import Any

from assistant.config import EN_DIR, XLAM_MODEL_ID
from assistant.memory.chat_memory import ChatMemory
from assistant.memory.child_db import ChildMemoryDB
from assistant.rag.stores import ChildRAG, MedicalRAG
from assistant.runtime_translate import translate_en_to_fa, translate_for_models
from assistant.tools.clinical import TOOL_SPECS, dispatch_tool
from assistant.tools.who_term_equations import classify_maturity


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
        self.model_id = XLAM_MODEL_ID

    def load(self):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._tok = AutoTokenizer.from_pretrained(XLAM_MODEL_ID, trust_remote_code=True)
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
        self._model = AutoModelForCausalLM.from_pretrained(XLAM_MODEL_ID, **kwargs)
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
            max_new_tokens=512,
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


def parse_tool_calls(text: str) -> list[dict]:
    text = text.strip()
    try:
        data = json.loads(text)
        return data.get("tool_calls", [])
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            return []
        try:
            data = json.loads(m.group(0))
            return data.get("tool_calls", [])
        except Exception:
            return []


def extract_growth_slots(text: str) -> dict:
    """Extract any growth slots present in a message (partial OK). Soft parent language."""
    text = re.split(r"\[SESSION_SLOTS\]|\[RECENT_CHAT\]", text or "", maxsplit=1)[0]
    slots: dict[str, Any] = {}
    if re.search(r"\b(male|boy|پسر(?:م|ه)?)\b", text, re.I):
        slots["sex"] = "male"
    elif re.search(r"\b(female|girl|دختر(?:م|ه)?)\b", text, re.I):
        slots["sex"] = "female"
    # Prefer explicit body measures; avoid matching "hc" inside unrelated words
    if re.search(r"\b(head(?:\s*circumference)?|hc)\b|دور\s*سر", text, re.I):
        slots["measure"] = "head_circumference"
    elif re.search(r"\b(length|height|قد)\b", text, re.I):
        slots["measure"] = "length"
    elif re.search(r"\b(weight|وزن|کilo|کیلو)\b", text, re.I):
        slots["measure"] = "weight"

    # Age in months (chronological) — term WHO path
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:months?|mos?|ماه(?:ه|گی)?)",
        text,
        re.I,
    )
    if m:
        slots["age_months"] = float(m.group(1))
        # Also stash approx weeks for tools that still want a weeks field
        slots["weeks"] = float(m.group(1)) * 4.345

    # Weeks (PMA or chronological depending on maturity)
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:w(?:eeks?)?|هفته)", text, re.I)
    if m and "age_months" not in slots:
        slots["weeks"] = float(m.group(1))

    # "age 32" / "سن ۳۲" without unit — PMA weeks if ≥27, else months
    if "weeks" not in slots and "age_months" not in slots:
        m = re.search(
            r"\b(?:age|pma|pna|سن)\s*[:=]?\s*(\d+(?:\.\d+)?)\b",
            text,
            re.I,
        )
        if not m:
            # Bare number only when message is basically just the age
            m = re.search(r"^\s*(\d+(?:\.\d+)?)\s*$", text)
        if m:
            age_n = float(m.group(1))
            if age_n >= 27:
                slots["weeks"] = age_n
            else:
                slots["age_months"] = age_n
                slots["weeks"] = age_n * 4.345

    # Value with unit
    m = re.search(r"(\d+(?:\.\d+)?)\s*(kg|کیلو(?:گرم)?|cm|سانتی\s*متر)", text, re.I)
    if not m:
        m = re.search(r"value\s*[:=]\s*(\d+(?:\.\d+)?)", text, re.I)
    if not m:
        # Soft: "وزن ۳.۲" or "3.2 kilo"
        m = re.search(r"(?:weight|وزن)\s*[:=]?\s*(\d+(?:\.\d+)?)", text, re.I)
    if m:
        slots["value"] = float(m.group(1))
        if "measure" not in slots:
            unit = (m.group(2) if m.lastindex and m.lastindex >= 2 else "") or ""
            if re.search(r"cm|سانتی", unit, re.I):
                slots["measure"] = "length"
            else:
                slots["measure"] = "weight"

    # Always plot when parent shares a measurement (no need to say "overlay")
    if re.search(r"\b(overlay|chart|plot|نمودار|رسم)\b", text, re.I):
        slots["want_overlay"] = True
    if slots.get("value") is not None and slots.get("measure") and (
        slots.get("weeks") is not None or slots.get("age_months") is not None
    ):
        slots["want_overlay"] = True

    if re.search(r"\b(preterm|نارس)\b", text, re.I):
        slots["chart_standard"] = "intergrowth_preterm"
    elif re.search(r"\b(term|طبیعی|ترم)\b", text, re.I):
        slots["chart_standard"] = "who_term"
    return slots


MEASURE_EXPLAIN_RE = re.compile(
    r"\b(?:what(?:'s| is| do you mean(?: by)?)?)\s+(?:the\s+)?measur|"
    r"\bby the measur|"
    r"\bmeasur(?:e|es|ement)?\s+you mean|"
    r"\bmesure|"
    r"منظورت(?:ان)?\s*از\s*(?:اندازه|مقیاس|measure)|اندازه\s*یعنی",
    re.I,
)
SHOW_CHART_RE = re.compile(
    r"\b(?:show|see|open|display|plot|draw)\b.{0,40}\bchart\b|"
    r"\b(?:child(?:'s)?|baby(?:'s)?|my)\s+chart\b|"
    r"نمودار|چارت|چارتش|نشان\s*بده.*(?:نمودار|چارت)|(?:نمودار|چارت).{0,12}(?:نشون|نشان)",
    re.I,
)
BARE_SHOW_RE = re.compile(r"^\s*(?:show(?:\s+it)?|نمایش(?:\s*بده)?)\s*[!.]?\s*$", re.I)
AFFIRM_RE = re.compile(
    r"^\s*(?:so\s+)?(?:it'?s|its)\s+(?:ok(?:ay|ey)?|fine|alright|good)\b|"
    r"^\s*(?:ok(?:ay|ey)?|alright|thanks|thank you|got it)\s*[!.]?\s*$|"
    r"^\s*(?:پس\s*)?(?:خوبه|اوکی|باشه|ممنون)\s*[!.]?\s*$",
    re.I,
)
ANALYZE_GROWTH_RE = re.compile(
    r"\b(?:analy[sz]e|interpret|explain)\b(?:\s+\w+){0,4}\s*(?:that|this|it|chart|result|growth|number|centile)?|"
    r"^\s*analy[sz]e\s*[!.]?\s*$|"
    r"\bwhat does (?:that|this|it|the (?:chart|result|number)) mean\b|"
    r"\b(?:is|am i|are we).{0,30}(?:on|in)\s+(?:a\s+|the\s+)?(?:good\s+|right\s+|correct\s+)?track\b|"
    r"\b(?:good|right|correct)\s+track\b|"
    r"\bon\s+track\b|"
    r"\bis (?:he|she|my (?:baby|child|son|daughter)|the (?:baby|child))\s+"
    r"(?:ok|okay|okey|fine|normal|healthy|alright|good)\b|"
    r"\bis (?:my |the )?(?:baby|child|he|she|son|daughter).{0,30}"
    r"(?:ok|okay|okey|fine|normal|healthy|good|alright|growing well)\b|"
    r"\bhow is (?:my |the )?(?:baby|child|growth|weight)\b|"
    r"تحلیل|تفسیر|یعنی\s*چی|وضعیت\s*رشد|مسیر\s*خوب|روی\s*خط|"
    r"رشدش\s*خوبه|وزنش\s*خوبه|خوبه\s*\?|"
    r"حالش\s*خوبه|اوکی\s*هست",
    re.I,
)
TALK_WORRY_RE = re.compile(
    r"\b(?:talk(?:ing)?|speech|language)\b.{0,30}\b(?:worr|concern|delay|ability)|"
    r"\b(?:worr|concern).{0,30}\b(?:talk(?:ing)?|speech|language|ability)|"
    r"cant?\s+talk|can'?t\s+talk|chald|child\s+cant|"
    r"نگران.{0,20}(?:حرف|گفتار|صحبت)|(?:حرف|گفتار).{0,20}نگران",
    re.I,
)


HELP_RE = re.compile(
    r"^\s*(hi|hello|hey|salam|سلام|درود)(?:\s*[!.]*)?\s*$|"
    r"^\s*(?:hi|hello|hey)[,!.\s]+(?:how can you help(?: me)?|what can you do)\s*\??\s*$|"
    r"^\s*(how can you help(?: me)?|what can you do|who are you)\s*\??\s*$|"
    r"^\s*(help(?: me)?|کمک(?:\s*کن)?)\s*\??\s*$",
    re.I,
)
HISTORY_RE = re.compile(
    r"\b(last|previous|history|remember|remind(?:\s+me)?|summary|"
    r"child profile|show (?:my )?(?:child|baby)(?: profile| data| info| growth| record)?|"
    r"who (?:is|did) I select|what do you know about (?:my )?(?:child|baby)|"
    r"child(?:'s)? (?:data|info|information|record|profile)|"
    r"my child(?:'s)? (?:last |growth |result|summary|profile|data)|"
    r"what (?:was|were) my|"
    r"we just (?:plotted|measured|saved|checked))\b|"
    r"قبلی|تاریخچه|آخرین|یادت|یادآوری|پرونده(?:\s*فرزند)?|اطلاعات (?:فرزند|کودک)|"
    r"وضعیت فرزندم را نشان بده|پروفایل|پروفیل|نشون\s*میدی|نشان\s*می\s*دهی|"
    r"بچم(?:و| را)?\s*(?:نشون|نشان)|فرزندم(?:و| را)?\s*(?:نشون|نشان)",
    re.I,
)
CONCERN_RE = re.compile(
    r"\b(can'?t talk|cannot talk|doesn'?t talk|not talking|speech delay|late (?:to )?talk|"
    r"won'?t speak|no words|language delay|developmental (?:delay|concern)|"
    r"worried|problem|wrong|abnormal|delay|"
    r"fever|rash|vomit|cough|cry(?:ing)? a lot)\b|"
    r"حرف\s*نمی\s*زن|حرف نمیزن|صحبت نمی|گفتار|تاخیر|نگران|مشکل|چی\s*کار|چه\s*کار|"
    r"چرا\s*.*(?:حرف|صحبت)|نمیتواند\s*حرف|کی\s*حرف|چه\s*موقع\s*حرف|کی\s*صحبت",
    re.I,
)
MEDICAL_RE = re.compile(
    r"\b(iron|sleep|vitamin|breast|feed|feeding|nutrition|vaccine|fever|colic|"
    r"complementary|weaning|sids|milestone|development|speech|talk(?:ing)?|language|"
    r"آهن|خواب|شیر|رشد|تغذیه|واکسن|حرف|گفتار|صحبت)\b|"
    r"\btell me about\b|\bwhat about\b|\bexplain\b|\bwhy (?:is|does|can'?t)\b|"
    r"مشکل چی|چی کار کنم|چه باید|کی\s*حرف|چه\s*موقع",
    re.I,
)
GROWTH_COMPUTE_RE = re.compile(
    r"\b(overlay|plot|percentile|z-?score|compute|calculate|check growth|"
    r"on the chart|growth chart|show(?:\s+me)?(?:\s+the)?\s+chart|the chart|نمودار)\b|"
    r"\b(?:show|see|open|display).{0,40}\bchart\b|"
    r"\b(?:child(?:'s)?|baby(?:'s)?)\s+chart\b|"
    r"\b(weight|length|height|head circumference|وزن|قد|دور\s*سر)\b.+\b\d+(?:\.\d+)?|"
    r"\b\d+(?:\.\d+)?\s*(?:kg|cm|کیلو(?:گرم)?)\b|"
    r"\b\d+(?:\.\d+)?\s*(?:w|weeks?|هفته)\b.+\b\d+(?:\.\d+)?|"
    r"\b(?:age|pma|pna|سن)\s*[:=]?\s*\d+|"
    r"\bvalue\s*[:=]|"
    r"چارت|نمودار|چارتش",
    re.I,
)
SCREEN_RE = re.compile(r"\b(asq|m-?chat|autism screen|غربال|سنین و مراحل)\b", re.I)


def classify_intent(user_message: str, prior_slots: dict | None = None) -> set[str]:
    """Classify the *current* user turn only (never history)."""
    msg = (user_message or "").strip()
    intents: set[str] = set()
    if not msg:
        return {"help"}

    concern = bool(CONCERN_RE.search(msg)) or bool(TALK_WORRY_RE.search(msg))
    measure_q = bool(MEASURE_EXPLAIN_RE.search(msg))
    show_chart = bool(SHOW_CHART_RE.search(msg))
    bare_show = bool(BARE_SHOW_RE.search(msg))
    affirm = bool(AFFIRM_RE.search(msg))
    analyze = bool(ANALYZE_GROWTH_RE.search(msg))

    # Pure greeting / capability question only
    if (
        HELP_RE.search(msg)
        and not concern
        and not GROWTH_COMPUTE_RE.search(msg)
        and not measure_q
        and not show_chart
        and not analyze
    ):
        intents.add("help")
        return intents

    if analyze:
        intents.add("growth_analysis")
        return intents

    if affirm and not concern and not show_chart and not GROWTH_COMPUTE_RE.search(msg):
        intents.add("reassure")
        return intents

    if measure_q or show_chart:
        intents.add("growth")

    if bare_show and prior_slots and (
        prior_slots.get("want_overlay")
        or prior_slots.get("value") is not None
        or prior_slots.get("child_id")
    ):
        intents.add("growth")

    if (concern or MEDICAL_RE.search(msg) or TALK_WORRY_RE.search(msg)) and not show_chart:
        if not re.search(
            r"\b(?:show|open|view)\s+my\s+child\b|child profile|پرونده|اطلاعات فرزند",
            msg,
            re.I,
        ):
            intents.add("medical")
        if re.search(r"talk|speech|language|حرف|گفتار|صحبت|chald", msg, re.I):
            intents.add("screening")

    if (
        HISTORY_RE.search(msg)
        and not concern
        and not GROWTH_COMPUTE_RE.search(msg)
        and not measure_q
        and not show_chart
        and not bare_show
    ):
        intents.add("history")

    if GROWTH_COMPUTE_RE.search(msg):
        intents.add("growth")
    if SCREEN_RE.search(msg):
        intents.add("screening")

    slots = extract_growth_slots(msg)
    # Continue a chart only when THIS turn adds real growth facts (not just boy/girl
    # from a sentence like «پسرم کی حرف میزنه»).
    growth_progress = any(k in slots for k in ("measure", "weeks", "age_months", "value"))
    if slots.get("want_overlay"):
        intents.add("growth")
        intents.discard("slot_update")
    elif (
        prior_slots
        and prior_slots.get("want_overlay")
        and growth_progress
        and not concern
        and "medical" not in intents
    ):
        intents.add("growth")
        intents.discard("slot_update")

    # Care / speech questions must never reuse leftover chart tools
    if ("medical" in intents or "screening" in intents or concern) and not (
        show_chart or bare_show or slots.get("want_overlay") or GROWTH_COMPUTE_RE.search(msg)
    ):
        intents.discard("growth")

    if "growth" in intents and (show_chart or bare_show or slots.get("want_overlay")):
        intents.discard("history")

    if "history" in intents and show_chart:
        intents.discard("history")

    if (
        slots
        and not intents.intersection(
            {"growth", "medical", "history", "screening", "help", "reassure", "growth_analysis"}
        )
        and not concern
        and not measure_q
        and len(msg.split()) <= 6
        and set(slots) - {"want_overlay", "chart_standard", "sex"}
    ):
        intents.add("slot_update")
    if not intents:
        intents.add("chat")
    return intents


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
            if "weeks" in slots:
                args["weeks"] = slots["weeks"]
            if "age_months" in slots:
                args["age_months"] = slots["age_months"]
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
            if "weeks" in slots:
                args["weeks"] = slots["weeks"]
            if "age_months" in slots:
                args["age_months"] = slots["age_months"]
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
        use_pleias: bool | None = None,
    ):
        self.db = db or ChildMemoryDB()
        self.chat_memory = chat_memory or ChatMemory()
        self.medical = MedicalRAG()
        self.child_rag = ChildRAG()
        # Default: load models when NESTLING_LOAD_MODELS=1
        load = os.environ.get("NESTLING_LOAD_MODELS", "0") == "1"
        self.use_pleias = use_pleias if use_pleias is not None else load
        self.tool_caller = XLAMToolCaller(enabled=False)
        if use_xlam if use_xlam is not None else load:
            self.tool_caller.load()
        self.medical.load()
        self.child_rag.load()

    def refresh_medical_index(self):
        return self.medical.build_from_chunks()

    def refresh_child_index(self, child_id: str):
        docs = self.db.timeline_documents(child_id)
        self.child_rag.reindex_child(docs)
        return len(docs)

    def ask_medical(self, query: str) -> dict:
        return self.medical.answer(query, use_pleias=self.use_pleias)

    def ask_child(self, child_id: str, query: str) -> dict:
        return self.child_rag.answer(query, child_id=child_id, use_pleias=self.use_pleias)

    def analyze_parent_photo(
        self,
        image_bytes: bytes,
        *,
        mime: str = "image/png",
        prompt: str = "",
        ui_lang: str | None = None,
    ) -> dict:
        """Vision + RAG path for parent-sent photos (rash/wound). Never diagnoses."""
        from assistant.llm.bonsai_client import bonsai_enabled, get_bonsai
        from assistant.parent_voice import medical_chat_answer
        from assistant.runtime_translate import ensure_pure_lang, translate_for_models

        reply_lang, en_prompt = translate_for_models(
            prompt or "Please look at this photo of my child's skin.",
            ui_lang=ui_lang,
        )
        rag_q = (
            f"{en_prompt} pediatric rash palm sole blister hand foot mouth wound redness fever"
        )
        rag = self.ask_medical(rag_q)
        context = rag.get("context") or rag.get("answer") or ""
        vision_text = ""
        mode = "rag_only"
        model = rag.get("model")
        if bonsai_enabled():
            try:
                client = get_bonsai()
                if client.ready:
                    vision_text = client.analyze_image(
                        image_bytes,
                        mime=mime,
                        prompt=en_prompt,
                        context=context,
                    )
                    mode = "bonsai-vision+rag"
                    model = "prism-ml/Bonsai-27B-gguf+mmproj"
            except Exception as exc:
                vision_text = ""
                mode = f"rag_fallback:{exc}"
        if not vision_text:
            vision_text = (
                "I received your photo. Without the vision model online I cannot see pixels yet, "
                "but here is calm guidance from our care notes based on common pediatric skin concerns "
                "(rashes on palms/soles can include viral illnesses such as hand-foot-and-mouth). "
                f"{rag.get('answer') or ''}"
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
            "tool_model": XLAM_MODEL_ID if self.tool_caller.enabled else "deterministic_router",
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
                store_weeks = float(overlay["age_months"]) * 4.345
            self.db.add_growth(
                child_id,
                weeks=store_weeks if store_weeks is not None else (weeks or 0),
                measure=overlay["measure"],
                value=value,
                z_score=overlay.get("z_score"),
                centile=overlay.get("centile"),
                track_status=overlay.get("track_status"),
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

    def chat(
        self,
        session_id: str,
        user_message: str,
        child_id: str | None = None,
        ui_lang: str | None = None,
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
        # Classify on both original + English so Persian concerns are not lost in MT
        intents = classify_intent(en_message, prior_slots=session_slots) | classify_intent(
            user_message, prior_slots=session_slots
        )
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
        if "growth" in intents:
            intents.discard("slot_update")
            intents.discard("chat")
            intents.discard("history")
        if "reassure" in intents:
            intents = {"reassure"}
        if "growth_analysis" in intents:
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

        tool_block = self.run_tools(
            en_message,
            slots=slots,
            user_message=en_message,
            intents=intents,
        )
        for tc in tool_block["tool_calls"]:
            res = tc.get("result") or {}
            if res.get("ok") and tc["name"] in {"growth_percentile", "overlay_growth_on_chart"}:
                # Remember last plotted result for follow-up analysis questions
                slots = self.chat_memory.merge_slots(
                    session_id,
                    {
                        "last_centile": res.get("centile"),
                        "last_z_score": res.get("z_score"),
                        "last_track_status": res.get("track_status"),
                        "last_measure": res.get("measure"),
                        "last_value": res.get("value"),
                        "last_age_months": res.get("age_months"),
                        "last_chart_standard": res.get("chart_standard"),
                        "want_overlay": True,
                    },
                )
                if child_id and res.get("value") is not None and new_measurement_turn:
                    store_weeks = res.get("weeks")
                    if store_weeks is None and res.get("age_months") is not None:
                        store_weeks = float(res["age_months"]) * 4.345
                    self.db.add_growth(
                        child_id,
                        weeks=store_weeks if store_weeks is not None else 0.0,
                        measure=res["measure"],
                        value=res["value"],
                        z_score=res.get("z_score"),
                        centile=res.get("centile"),
                        track_status=res.get("track_status"),
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
                "tool_calling": XLAM_MODEL_ID,
                "tool_calling_loaded": bool(self.tool_caller.enabled),
                "rag": "PleIAs/Pleias-RAG-1B",
                "rag_loaded": bool(self.use_pleias),
            },
        }

        if "growth_analysis" in intents:
            snap = latest_growth_snapshot(self.db, child_id, slots)
            if snap and (snap.get("centile") is not None or snap.get("track_status")):
                out["growth_analysis"] = snap
            else:
                out["growth_analysis"] = {"missing": True}

        if "medical" in intents:
            out["medical_rag"] = self.ask_medical(en_message)
        # Prefer deterministic child summary tool; skip noisy child-RAG dump when summary exists
        if child_id and "history" in intents:
            has_summary = any(
                tc.get("name") == "get_child_summary" and (tc.get("result") or {}).get("ok")
                for tc in tool_block.get("tool_calls") or []
            )
            if not has_summary:
                out["child_rag"] = self.ask_child(child_id, en_message)

        missing = []
        if "growth" in intents:
            for key in ("sex", "measure"):
                if key not in slots:
                    missing.append(key)
            if "weeks" not in slots and "age_months" not in slots:
                missing.append("age (weeks or months)")
            if "value" not in slots:
                missing.append("value")
        out["missing_slots"] = missing
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
        out["history"] = self.chat_memory.get_history(session_id)
        return out

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

        if "reassure" in intents:
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
                summary = res.get("summary") or ""
                parts.append(child_summary_chat(summary, fa=fa))
            elif res.get("ok") is False and res.get("detail"):
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
            if fa and ans:
                ans = translate_en_to_fa(ans)
            parts.append(medical_chat_answer(ans, fa=fa))

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
            uniq.append(open_chat_turn(fa=fa, has_growth=has_growth))
        return "\n\n".join(uniq)


    def handle(self, query: str, child_id: str | None = None) -> dict:
        sid = self.start_session(child_id=child_id)
        return self.chat(sid, query, child_id=child_id)

    def close(self) -> None:
        """Release SQLite handles (needed on Windows before deleting temp DBs)."""
        try:
            self.db.close()
        except Exception:
            pass
        try:
            self.chat_memory.close()
        except Exception:
            pass
