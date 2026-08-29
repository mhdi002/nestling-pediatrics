"""Growth slot extraction from parent messages."""
from __future__ import annotations

import re
from typing import Any

from assistant.refdata import weeks_per_month

def extract_growth_slots(text: str) -> dict:
    """Extract any growth slots present in a message (partial OK). Soft parent language."""
    text = re.split(
        r"\[SESSION_SLOTS\]|\[RECENT_CHAT\]|\[SESSION_SUMMARY\]|\[CURRENT_USER\]",
        text or "",
        maxsplit=1,
    )[0]
    slots: dict[str, Any] = {}

    # Gestational age at birth (must run before generic weeks parsing)
    ga = re.search(
        r"\b(?:born(?:\s+at)?|gestational(?:\s+age)?|g\.?a\.?|birth(?:\s+at)?)\s*"
        r"(?:was\s+)?(?:at\s+)?(\d+(?:\.\d+)?)\s*(?:w(?:eeks?)?|هفته)\b|"
        r"\b(\d+(?:\.\d+)?)\s*(?:w(?:eeks?)?|هفته)\s*(?:at\s+)?(?:birth|gestation)|"
        r"(?:سن\s*بارداری|هفته\s*ای)\s*(\d+(?:\.\d+)?)",
        text,
        re.I,
    )
    if ga:
        for g in ga.groups():
            if g is not None:
                slots["gestational_age_weeks"] = float(g)
                break

    # Sex — only on growth/slot-fill turns, not medical concerns like «پسرم حرف نمیزنه»
    sex_male = bool(re.search(r"\b(male|boy|پسر(?:م|ه)?)\b", text, re.I))
    sex_female = bool(re.search(r"\b(female|girl|دختر(?:م|ه)?)\b", text, re.I))
    growthish = bool(
        re.search(
            r"\b(weight|length|height|head|kg|cm|overlay|chart|plot|percentile|"
            r"weeks?|months?|preterm|term|وزن|قد|دور\s*سر|نمودار|نارس|طبیعی|کیلو)\b",
            text,
            re.I,
        )
    ) or len(re.findall(r"\w+", text)) <= 6
    if growthish:
        if sex_male and not sex_female:
            slots["sex"] = "male"
        elif sex_female and not sex_male:
            slots["sex"] = "female"

    # Prefer explicit body measures; avoid matching "hc" inside unrelated words
    if re.search(r"\b(head(?:\s*circumference)?|hc)\b|دور\s*سر", text, re.I):
        slots["measure"] = "head_circumference"
    elif re.search(r"\b(length|height|قد)\b", text, re.I):
        slots["measure"] = "length"
    elif re.search(r"\b(weight|وزن|کilo|کیلو)\b", text, re.I):
        slots["measure"] = "weight"

    # Age in years — parents routinely say "2 years old" / "۲ ساله" rather
    # than "24 months". Must run before the months and bare-number rules so
    # that "2 years" is not left unparsed (which previously made the agent
    # re-ask for an age the parent had already given).
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:y(?:ea)?rs?\b|yo\b|سال(?:ه|گی)?)",
        text,
        re.I,
    )
    if m:
        years = float(m.group(1))
        slots["age_months"] = years * 12.0
        slots["weeks"] = years * 12.0 * weeks_per_month()

    # Age in months (chronological) — term WHO path
    m = re.search(
        r"(\d+(?:\.\d+)?)\s*(?:months?|mos?|ماه(?:ه|گی)?)",
        text,
        re.I,
    )
    if m:
        slots["age_months"] = float(m.group(1))
        # Also stash approx weeks for tools that still want a weeks field
        slots["weeks"] = float(m.group(1)) * weeks_per_month()

    # Weeks (PMA or chronological) — skip the same span used as birth GA
    m = re.search(r"(\d+(?:\.\d+)?)\s*(?:w(?:eeks?)?|هفته)", text, re.I)
    if m and "age_months" not in slots:
        wval = float(m.group(1))
        # If this number was only the birth GA phrase, don't treat it as measurement age
        if slots.get("gestational_age_weeks") == wval and re.search(
            r"\b(born|gestational|g\.?a\.?|birth|سن\s*بارداری)\b", text, re.I
        ):
            # Look for a *different* weeks mention (measurement age)
            others = re.findall(r"(\d+(?:\.\d+)?)\s*(?:w(?:eeks?)?|هفته)", text, re.I)
            other_vals = [float(x) for x in others if abs(float(x) - wval) > 1e-9]
            if other_vals:
                slots["weeks"] = other_vals[0]
            # else: leave weeks unset so prior session weeks survive merge
        else:
            slots["weeks"] = wval

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
                slots["weeks"] = age_n * weeks_per_month()

    # Value with unit
    m = re.search(r"(\d+(?:\.\d+)?)\s*(kg|کیلو(?:گرم)?|cm|سانتی\s*متر)", text, re.I)
    if not m:
        m = re.search(r"value\s*[:=]\s*(\d+(?:\.\d+)?)", text, re.I)
    if not m:
        # Soft: "وزن ۳.۲" or "weight 3.2" — but not "weight 40 weeks"
        m = re.search(
            r"(?:weight|وزن)\s*[:=]?\s*(\d+(?:\.\d+)?)(?!\s*(?:w(?:eeks?)?|هفته|months?|mos?|ماه))",
            text,
            re.I,
        )
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


