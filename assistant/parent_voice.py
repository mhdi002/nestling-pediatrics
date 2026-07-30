"""Conversational parent-facing phrasing — grounded, not a database dump."""

from __future__ import annotations

import re
from typing import Any


def _first_sentences(text: str, max_sentences: int = 3, max_chars: int = 520) -> str:
    text = re.sub(r"\s+", " ", (text or "").strip())
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?۔؟])\s+", text)
    out: list[str] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        out.append(p)
        if len(out) >= max_sentences or sum(len(x) for x in out) >= max_chars:
            break
    return " ".join(out)


def medical_chat_answer(raw: str, *, fa: bool = False, from_llm: bool = False) -> str:
    """Turn RAG extract / model text into a short parent chat reply."""
    text = (raw or "").strip()
    if not text:
        if fa:
            return "سوال‌تان را کمی دقیق‌تر بگویید تا از راهنمای مراقبتی‌مان کمک کنم."
        return "Tell me a bit more and I’ll answer from our care guides."

    # Strip dump headers / citation bullets
    text = re.sub(r"(?i)^based on retrieved sources:\s*", "", text)
    text = re.sub(r"(?i)^بر اساس منابع بازیابی شده:\s*", "", text)
    text = re.sub(r"(?i)^from our care notes[^:]*:\s*", "", text)
    text = re.sub(r"(?m)^-\s*\([^)]+\)\s*", "", text)
    text = re.sub(r"(?m)^-\s*[^:]+:\s*", "", text)
    text = re.sub(r"(?i)\n?for diagnosis or treatment decisions.*?$", "", text)
    text = re.sub(r"(?i)\n?برای تصمیم.?گیری.*?$", "", text)
    text = re.sub(r"(?i)\[LLM unavailable:.*?\]", "", text)
    text = re.sub(r"^#+\s*", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    if from_llm:
        # Keep generative answers intact (light trim only)
        body = text[:1200].strip() if len(text) > 1200 else text
        if fa:
            return f"{body}\n\nبرای تشخیص یا درمان حتماً با متخصص کودکان مشورت کنید."
        return f"{body}\n\nFor diagnosis or treatment, please check with your pediatrician."

    body = _first_sentences(text, max_sentences=3, max_chars=480)
    if not body:
        body = text[:420]

    if fa:
        return (
            f"{body}\n\n"
            "اگر بخواهید، می‌توانم ساده‌تر توضیح بدهم. "
            "برای تشخیص یا درمان حتماً با متخصص کودکان مشورت کنید."
        )
    return (
        f"{body}\n\n"
        "Want me to simplify that, or talk about something else next? "
        "For diagnosis or treatment, please check with your pediatrician."
    )


def growth_plot_chat(res: dict[str, Any], *, fa: bool = False) -> str:
    """Narrate a plotted growth point like a chat, not a tool log."""
    measure = res.get("measure") or "measurement"
    value = res.get("value")
    centile = res.get("centile")
    age_m = res.get("age_months")
    weeks = res.get("weeks")
    chart = "WHO" if res.get("chart_standard") == "who_term" else "INTERGROWTH"
    mat = (res.get("maturity_label_fa") if fa else res.get("maturity_label_en")) or res.get("maturity") or ""
    track = (res.get("track_status") or "").lower()

    age_bit = ""
    if age_m is not None:
        try:
            age_bit = f"about {float(age_m):.1f} months" if not fa else f"حدود {float(age_m):.1f} ماهگی"
        except (TypeError, ValueError):
            age_bit = str(age_m)
    elif weeks is not None:
        age_bit = f"{weeks} weeks" if not fa else f"{weeks} هفته"

    c_bit = ""
    try:
        if centile is not None:
            c_bit = f"{float(centile):.0f}"
    except (TypeError, ValueError):
        c_bit = str(centile) if centile is not None else ""

    if fa:
        lead = f"نمودار را برایتان کشیدم ({chart}"
        if mat:
            lead += f"، {mat}"
        lead += ")."
        mid = f" {measure} حدود {value}"
        if age_bit:
            mid += f" در {age_bit}"
        if c_bit:
            mid += f" — تقریباً صدک {c_bit}"
        mid += "."
        if "within_10_90" in track:
            tail = " این در بازه معمول همسالان است. اگر بخواهید می‌توانم همین را ساده تحلیل کنم."
        elif "below_3rd" in track or "above_97" in track:
            tail = " این نقطه خارج بازه معمول است — پیشنهاد می‌کنم با پزشک در میان بگذارید. می‌خواهید توضیح بدهم؟"
        else:
            tail = " نمودار را پایین ببینید. می‌خواهید تحلیلش کنم؟"
        return lead + mid + tail

    lead = f"I plotted this on the {chart} chart"
    if mat:
        lead += f" ({mat})"
    lead += "."
    mid = f" {measure} {value}"
    if age_bit:
        mid += f" at {age_bit}"
    if c_bit:
        mid += f" is around the {c_bit}th centile"
    mid += "."
    if "within_10_90" in track:
        tail = " That’s in the usual peer range. Want me to walk through what that means?"
    elif "below_3rd" in track or "above_97" in track:
        tail = " That’s outside the usual range — worth reviewing with your pediatrician. Want a plain-language read?"
    else:
        tail = " See the chart below. I can analyze it with you if you’d like."
    return lead + mid + tail


def child_summary_chat(summary_text: str, *, fa: bool = False) -> str:
    """Soften get_child_summary tool text into a short chat intro + bullets."""
    lines = [ln.strip() for ln in (summary_text or "").splitlines() if ln.strip()]
    if not lines:
        if fa:
            return "هنوز چیز زیادی از پرونده این گفتگو ندارم. می‌توانید وزن یا نگرانی‌تان را بگویید."
        return "I don’t have much on file for this child yet — share a weight or a worry and we’ll start."
    head = lines[0]
    rest = lines[1:6]
    if fa:
        opener = f"بفرمایید، خلاصه پرونده این است:\n{head}"
    else:
        opener = f"Here’s what I have on file:\n{head}"
    if rest:
        opener += "\n" + "\n".join(rest)
    if fa:
        opener += "\n\nچی کمکتان می‌کند — نمودار، تحلیل رشد، یا سوال مراقبتی؟"
    else:
        opener += "\n\nWhat would help next — the chart, a growth read, or a care question?"
    return opener


def open_chat_turn(*, fa: bool = False, has_growth: bool = False) -> str:
    """Fallback when intent is vague — still conversational, not a dead end."""
    if has_growth:
        if fa:
            return (
                "گوش می‌دهم. می‌توانیم درباره همان اندازه‌گیری حرف بزنیم، "
                "یا هر نگرانی مراقبتی مثل زخم، خواب، آهن یا تغذیه را بگویید."
            )
        return (
            "I’m listening. We can revisit the measurement we just looked at, "
            "or ask about any care concern — a scar or wound, sleep, iron, feeding, talking."
        )
    if fa:
        return (
            "گوش می‌دهم. می‌توانیم درباره رشد، تغذیه، خواب یا نگرانی‌های تکاملی حرف بزنیم — "
            "هرطور راحتید بگویید چه چیزی ذهنتان را مشغول کرده."
        )
    return (
        "I’m listening. We can talk about growth, feeding, sleep, or developmental worries — "
        "just tell me what’s on your mind in your own words."
    )
