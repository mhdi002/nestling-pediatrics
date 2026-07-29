#!/usr/bin/env python3
"""Deterministic clinical tools — equations, scoring, overlays. No LLM math."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Literal

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import intergrowth_preterm_equations as ig
from assistant.tools import who_term_equations as who
from assistant.config import (
    ASQ_DEFAULT_CUTOFF,
    ASQ_SCORE_NOT_YET,
    ASQ_SCORE_SOMETIMES,
    ASQ_SCORE_YES,
    EN_DIR,
    EXTRACTED,
    OVERLAY_DIR,
)

AnswerASQ = Literal["yes", "sometimes", "not_yet", "بله", "گاهی", "هنوز نه", "No", "Yes", "Sometimes", "Not yet"]

# Clinical input bounds (never invent; reject out-of-range)
WEEKS_MIN = ig.AGE_WEEKS_MIN  # 27
WEEKS_MAX = ig.AGE_WEEKS_MAX  # 64
VALUE_RANGES = {
    "weight": (0.2, 20.0),  # kg
    "length": (20.0, 90.0),  # cm
    "head_circumference": (15.0, 55.0),  # cm
}

# Optional ChildMemoryDB injected by ParentAssistant for get_child_summary
_CHILD_DB = None


def set_child_db(db) -> None:
    """Register ChildMemoryDB instance used by get_child_summary tool."""
    global _CHILD_DB
    _CHILD_DB = db


def tool_error(tool: str, message: str, **extra: Any) -> dict[str, Any]:
    return {"ok": False, "error": message, "detail": message, "tool": tool, **extra}


def tool_ok(tool: str, payload: dict[str, Any]) -> dict[str, Any]:
    out = {"ok": True, "tool": tool, **payload}
    return out


def _validate_sex(sex: Any) -> str | dict:
    try:
        return ig.normalize_sex(sex)
    except Exception as exc:
        return tool_error("validation", f"Invalid sex: {sex!r}. Use male/female (or boy/girl).", detail=str(exc))


def _validate_measure(measure: Any) -> str | dict:
    try:
        return ig.normalize_measure(measure)
    except Exception as exc:
        return tool_error(
            "validation",
            f"Invalid measure: {measure!r}. Use weight, length, or head_circumference.",
            detail=str(exc),
        )


def _validate_weeks(weeks: Any) -> float | dict:
    try:
        w = float(weeks)
    except (TypeError, ValueError):
        return tool_error("validation", f"Invalid weeks: {weeks!r}. Expected a number.")
    if not (WEEKS_MIN <= w <= WEEKS_MAX):
        return tool_error(
            "validation",
            f"weeks must be between {WEEKS_MIN} and {WEEKS_MAX} (postmenstrual age). Got {w}.",
            weeks=w,
        )
    return w


def _validate_value(measure: str, value: Any) -> float | dict:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return tool_error("validation", f"Invalid value: {value!r}. Expected a number.")
    lo, hi = VALUE_RANGES[measure]
    unit = "kg" if measure == "weight" else "cm"
    if not (lo <= v <= hi):
        return tool_error(
            "validation",
            f"{measure} value must be between {lo} and {hi} {unit}. Got {v}.",
            value=v,
            measure=measure,
        )
    return v


def _norm_asq_answer(ans: str) -> str:
    a = (ans or "").strip().lower()
    mapping = {
        "yes": "yes",
        "y": "yes",
        "بله": "yes",
        "بلی": "yes",
        "sometimes": "sometimes",
        "گاهی": "sometimes",
        "گاهي": "sometimes",
        "not_yet": "not_yet",
        "not yet": "not_yet",
        "هنوز نه": "not_yet",
        "هنوزنه": "not_yet",
        "no": "not_yet",
        "خیر": "not_yet",
        "خير": "not_yet",
    }
    if a not in mapping:
        raise ValueError(f"Invalid ASQ answer: {ans!r}")
    return mapping[a]


def score_asq_domain(answers: list[str]) -> dict[str, Any]:
    """Score one ASQ domain. Official points: Yes=10, Sometimes=5, Not yet=0."""
    if not isinstance(answers, list) or not answers:
        return tool_error("score_asq_domain", "answers must be a non-empty list of Yes/Sometimes/Not yet.")
    points = []
    try:
        for ans in answers:
            key = _norm_asq_answer(ans)
            points.append(
                {"yes": ASQ_SCORE_YES, "sometimes": ASQ_SCORE_SOMETIMES, "not_yet": ASQ_SCORE_NOT_YET}[key]
            )
    except ValueError as exc:
        return tool_error("score_asq_domain", str(exc))
    total = sum(points)
    cutoff = ASQ_DEFAULT_CUTOFF
    return {
        "ok": True,
        "item_scores": points,
        "total": total,
        "max": 10 * len(answers),
        "cutoff": cutoff,
        "below_cutoff": total < cutoff,
        "interpretation": "below_cutoff_refer" if total < cutoff else "above_cutoff_monitor",
    }


def score_asq_questionnaire(domain_answers: dict[str, list[str]]) -> dict[str, Any]:
    """
    domain_answers: {domain_id: [answers...]} for communication, gross_motor, etc.
    Overall section is yes/no concern items — stored separately, not point-scored here.
    """
    if not isinstance(domain_answers, dict) or not domain_answers:
        return tool_error("score_asq_questionnaire", "domain_answers must be a non-empty object.")
    domains = {}
    referrals = []
    for dom, answers in domain_answers.items():
        if dom == "overall":
            continue
        result = score_asq_domain(answers)
        if result.get("ok") is False:
            return {**result, "tool": "score_asq_questionnaire", "domain": dom}
        domains[dom] = result
        if result["below_cutoff"]:
            referrals.append(dom)
    return {
        "ok": True,
        "tool": "score_asq_questionnaire",
        "domains": domains,
        "referral_domains": referrals,
        "needs_referral": bool(referrals),
        "summary": (
            f"ASQ scored {len(domains)} domains; "
            f"{'referral suggested for: ' + ', '.join(referrals) if referrals else 'no domain below cutoff'}"
        ),
    }


# M-CHAT-R: items where NO is the risk answer (failed) vs YES is risk.
# Standard M-CHAT-R reverse-scored items: 2,5,12 (Yes = fail). Others: No = fail.
MCHAT_REVERSE = {2, 5, 12}


def score_mchat(answers: dict[int, str]) -> dict[str, Any]:
    """
    answers: {question_id: 'yes'|'no'|'آری'|'خیر'}
    Returns fail count and risk tier per common M-CHAT-R rules:
      0–2 low risk, 3–7 medium, 8–20 high (then follow-up interview for medium).
    """
    if not isinstance(answers, dict) or not answers:
        return tool_error("score_mchat", "answers must be a non-empty map of question_id → yes/no.")
    fails = []
    try:
        for qid, ans in answers.items():
            a = (ans or "").strip().lower()
            yes = a in {"yes", "y", "آری", "اري", "بله"}
            no = a in {"no", "n", "خیر", "خير", "نه"}
            if not yes and not no:
                return tool_error("score_mchat", f"Invalid M-CHAT answer for Q{qid}: {ans!r}")
            q = int(qid)
            if q in MCHAT_REVERSE:
                failed = yes
            else:
                failed = no
            if failed:
                fails.append(q)
    except (TypeError, ValueError) as exc:
        return tool_error("score_mchat", f"Invalid answers payload: {exc}")
    n = len(fails)
    if n <= 2:
        risk = "low"
    elif n <= 7:
        risk = "medium"
    else:
        risk = "high"
    return {
        "ok": True,
        "tool": "score_mchat",
        "failed_items": sorted(fails),
        "total_failed": n,
        "risk": risk,
        "summary": f"M-CHAT-R: {n} failed items → {risk} risk",
        "note": "Medium risk typically requires M-CHAT-R/F follow-up interview; high risk → refer.",
    }


def _term_age_months_from_weeks(
    weeks: float, gestational_age_weeks: float | None
) -> float:
    """
    Map a parent 'weeks' value onto WHO chronological age (months).

    Parents often say '40 weeks' meaning near birth / PMA, not 40 weeks of life.
    With birth GA known: chronological weeks = max(0, weeks − GA).
    Without GA: values in the 37–45w band are treated as near-term PMA (≈ birth).
    Smaller week counts are treated as weeks since birth.
    """
    w = float(weeks)
    if gestational_age_weeks is not None:
        ga = float(gestational_age_weeks)
        if w + 1e-9 >= ga:
            return max(0.0, w - ga) / 4.345
        return w / 4.345
    if 37.0 <= w <= 45.0:
        return max(0.0, w - 40.0) / 4.345
    return w / 4.345


def resolve_chart_route(
    gestational_age_weeks: float | None = None,
    weeks: float | None = None,
    age_months: float | None = None,
    chart_standard: str | None = None,
) -> dict[str, Any]:
    """
    Pick preterm (نارس / INTERGROWTH PMA) vs term (طبیعی / WHO months).
    Parent does not need to know the category — GA on the child profile decides.
    """
    if chart_standard in {"intergrowth_preterm", "who_term"}:
        std = chart_standard
        maturity = "preterm" if std == "intergrowth_preterm" else "term"
    else:
        maturity = who.classify_maturity(gestational_age_weeks)
        if maturity == "term":
            std = "who_term"
        else:
            # preterm or unknown with PMA weeks → INTERGROWTH
            std = "intergrowth_preterm"

    if std == "who_term":
        if age_months is None and weeks is not None:
            age_months = _term_age_months_from_weeks(float(weeks), gestational_age_weeks)
        if age_months is None and weeks is None and gestational_age_weeks is not None:
            return tool_error(
                "resolve_chart_route",
                "Term child needs age in months or weeks since birth.",
            )
        if age_months is None:
            return tool_error("resolve_chart_route", "Missing age for WHO term chart.")
        if not (0.0 <= float(age_months) <= 24.0):
            return tool_error("resolve_chart_route", "WHO term charts support 0–24 months.")
        return {
            "ok": True,
            "maturity": "term",
            "maturity_label_en": "term",
            "maturity_label_fa": "طبیعی",
            "chart_standard": "who_term",
            "age_months": float(age_months),
            "weeks": None,
        }

    # INTERGROWTH preterm — postmenstrual age
    pma = weeks
    if pma is None and age_months is not None and gestational_age_weeks is not None:
        pma = float(gestational_age_weeks) + float(age_months) * 4.345
    if pma is None:
        return tool_error(
            "resolve_chart_route",
            "Preterm growth needs postmenstrual weeks (or age months + birth GA).",
        )
    weeks_n = _validate_weeks(pma)
    if isinstance(weeks_n, dict):
        return weeks_n
    return {
        "ok": True,
        "maturity": "preterm" if maturity != "unknown" else "preterm_assumed",
        "maturity_label_en": "preterm",
        "maturity_label_fa": "نارس",
        "chart_standard": "intergrowth_preterm",
        "weeks": weeks_n,
        "age_months": None,
    }


def _track_status(c: float) -> str:
    if c < 3:
        return "below_3rd_investigate"
    if c > 97:
        return "above_97th_investigate"
    if c < 10 or c > 90:
        return "outer_centile_monitor"
    return "within_10_90"


def growth_percentile(
    sex: str,
    measure: str,
    weeks: float | None = None,
    value: float | None = None,
    percentile: float | None = None,
    gestational_age_weeks: float | None = None,
    age_months: float | None = None,
    chart_standard: str | None = None,
) -> dict[str, Any]:
    """Evaluate growth equations (INTERGROWTH preterm or WHO term)."""
    sex_n = _validate_sex(sex)
    if isinstance(sex_n, dict):
        return {**sex_n, "tool": "growth_percentile"}
    meas_n = _validate_measure(measure)
    if isinstance(meas_n, dict):
        return {**meas_n, "tool": "growth_percentile"}

    route = resolve_chart_route(
        gestational_age_weeks=gestational_age_weeks,
        weeks=weeks,
        age_months=age_months,
        chart_standard=chart_standard,
    )
    if route.get("ok") is False:
        return {**route, "tool": "growth_percentile"}

    value_n: float | None = None
    if value is not None:
        value_n = _validate_value(meas_n, value)
        if isinstance(value_n, dict):
            return {**value_n, "tool": "growth_percentile"}

    if percentile is not None:
        try:
            p = float(percentile)
        except (TypeError, ValueError):
            return tool_error("growth_percentile", f"Invalid percentile: {percentile!r}")
        if not (0.0 < p < 100.0):
            return tool_error("growth_percentile", "percentile must be between 0 and 100 exclusive.")
    else:
        p = None

    try:
        if route["chart_standard"] == "who_term":
            age_m = route["age_months"]
            chart = {pct: who.percentile(sex_n, meas_n, age_m, pct) for pct in who.CHART_PERCENTILES}
            ref = "WHO Child Growth Standards 2006 (term infants)"
            age_label = f"{age_m:.1f} months"
        else:
            weeks_n = route["weeks"]
            chart = {pct: ig.percentile(sex_n, meas_n, weeks_n, pct) for pct in ig.CHART_PERCENTILES}
            ref = "Villar et al. Lancet Glob Health 2015; INTERGROWTH-21st (preterm)"
            age_label = f"{weeks_n}w PMA"
    except Exception as exc:
        return tool_error("growth_percentile", f"Equation evaluation failed: {exc}")

    out: dict[str, Any] = {
        "ok": True,
        "tool": "growth_percentile",
        "sex": sex_n,
        "measure": meas_n,
        "weeks": route.get("weeks"),
        "age_months": route.get("age_months"),
        "maturity": route.get("maturity"),
        "maturity_label_en": route.get("maturity_label_en"),
        "maturity_label_fa": route.get("maturity_label_fa"),
        "chart_standard": route["chart_standard"],
        "chart_percentiles": chart,
        "units": {"weight": "kg", "length": "cm", "head_circumference": "cm"}[meas_n],
        "reference": ref,
    }
    if value_n is not None:
        if route["chart_standard"] == "who_term":
            z = who.z_score(sex_n, meas_n, route["age_months"], value_n)
            c = who.centile_from_measurement(sex_n, meas_n, route["age_months"], value_n)
        else:
            z = ig.z_score(sex_n, meas_n, route["weeks"], value_n)
            c = ig.centile_from_measurement(sex_n, meas_n, route["weeks"], value_n)
        out["value"] = value_n
        out["z_score"] = z
        out["centile"] = c
        track = _track_status(c)
        out["track_status"] = track
        # English-only parent summary (FA replies are translated at the chat boundary)
        mat = route.get("maturity_label_en") or route.get("maturity") or ""
        out["summary"] = (
            f"{out['measure']}={value_n} at {age_label} ({out['sex']}, {mat}): "
            f"centile≈{c:.1f}, z≈{z:.2f}, status={track}"
        )
        out["summary_fa"] = (
            f"{out['measure']}={value_n} در {age_label} ({out['sex']}، {route.get('maturity_label_fa')}): "
            f"صدک≈{c:.1f}، z≈{z:.2f}، وضعیت={track}"
        )
    if p is not None:
        if route["chart_standard"] == "who_term":
            out["requested_percentile_value"] = who.percentile(sex_n, meas_n, route["age_months"], p)
        else:
            out["requested_percentile_value"] = ig.percentile(sex_n, meas_n, route["weeks"], p)
    return out


def overlay_growth_on_chart(
    sex: str,
    measure: str,
    weeks: float | None = None,
    value: float | None = None,
    child_id: str | None = None,
    history: list[dict] | None = None,
    gestational_age_weeks: float | None = None,
    age_months: float | None = None,
    chart_standard: str | None = None,
) -> dict[str, Any]:
    """
    Plot percentile curves (INTERGROWTH preterm or WHO term) and overlay child point(s).
    """
    if value is None:
        return tool_error("overlay_growth_on_chart", "value is required for overlay.")
    assessment = growth_percentile(
        sex,
        measure,
        weeks=weeks,
        value=value,
        gestational_age_weeks=gestational_age_weeks,
        age_months=age_months,
        chart_standard=chart_standard,
    )
    if assessment.get("ok") is False:
        return {**assessment, "overlay_path": None, "tool": "overlay_growth_on_chart"}

    OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        return {
            **assessment,
            "tool": "overlay_growth_on_chart",
            "overlay_path": None,
            "plot_error": f"matplotlib unavailable: {exc}",
            "note": "Numeric assessment computed; chart image not generated.",
        }

    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        styles = {
            97: ("#c0392b", "-"),
            90: ("#2c3e50", "--"),
            50: ("#27ae60", "-"),
            10: ("#2c3e50", "--"),
            3: ("#c0392b", "-"),
        }
        std = assessment["chart_standard"]
        if std == "who_term":
            xs = [i / 2 for i in range(0, 24 * 2 + 1)]
            for p, (color, ls) in styles.items():
                ys = [who.percentile(sex, measure, m, p) for m in xs]
                ax.plot(xs, ys, color=color, linestyle=ls, linewidth=1.5, label=f"P{p}")
            x_child = assessment["age_months"]
            ax.plot([x_child], [value], "o", color="#2980b9", markersize=9, label="Child")
            ax.set_xlabel("Age (months)")
            title = f"WHO {assessment['measure']} ({assessment['sex']}, term) — child overlay"
            tag = f"{x_child:.1f}m"
        else:
            xs = [w / 2 for w in range(27 * 2, 64 * 2 + 1)]
            for p, (color, ls) in styles.items():
                ys = [ig.percentile(sex, measure, w, p) for w in xs]
                ax.plot(xs, ys, color=color, linestyle=ls, linewidth=1.5, label=f"P{p}")
            pts = list(history or []) + [{"weeks": assessment["weeks"], "value": value}]
            ax.plot(
                [p["weeks"] for p in pts],
                [p["value"] for p in pts],
                "o-",
                color="#2980b9",
                markersize=8,
                label="Child",
            )
            ax.set_xlabel("Postmenstrual age (weeks)")
            title = (
                f"INTERGROWTH-21st {assessment['measure']} "
                f"({assessment['sex']}, preterm) — child overlay"
            )
            tag = f"{assessment['weeks']}w"

        unit = assessment["units"]
        ax.set_ylabel(f"{assessment['measure']} ({unit})")
        ax.set_title(title)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        name = f"overlay_{child_id or 'child'}_{assessment['measure']}_{tag}.png".replace(" ", "")
        path = OVERLAY_DIR / name
        fig.savefig(path, dpi=140)
        plt.close(fig)
    except Exception as exc:
        try:
            plt.close("all")
        except Exception:
            pass
        return {
            **assessment,
            "tool": "overlay_growth_on_chart",
            "overlay_path": None,
            "plot_error": f"plot failed: {exc}",
            "note": "Numeric assessment computed; chart image not generated.",
        }

    return {
        **assessment,
        "tool": "overlay_growth_on_chart",
        "overlay_path": str(path),
        "overlay_filename": path.name,
        "overlay": str(path),
    }


def _resolve_asq_path(age_months: int) -> Path | None:
    for base in (EN_DIR / "asq", EXTRACTED / "asq"):
        path = base / f"{int(age_months)}m.json"
        if path.exists():
            return path
    return None


def list_asq_questions(age_months: int) -> dict[str, Any]:
    """Load ASQ question text for a given age (months) from data/en or extracted/."""
    try:
        age = int(age_months)
    except (TypeError, ValueError):
        return tool_error("list_asq_questions", f"Invalid age_months: {age_months!r}")
    if age < 1 or age > 72:
        return tool_error("list_asq_questions", f"age_months out of range: {age}")

    path = _resolve_asq_path(age)
    if path is None:
        return tool_error(
            "list_asq_questions",
            f"No ASQ questionnaire found for {age} months.",
            age_months=age,
        )
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return tool_error("list_asq_questions", f"Failed to read ASQ file: {exc}")

    domains_out = []
    for dom in data.get("domains", []):
        questions = []
        for q in dom.get("questions", []):
            questions.append(
                {
                    "id": q.get("id"),
                    "text_en": q.get("text_en") or q.get("text") or "",
                    "text_fa": q.get("text_fa"),
                    "options_en": q.get("options_en") or dom.get("answer_options_en"),
                }
            )
        domains_out.append(
            {
                "id": dom.get("id"),
                "title_en": dom.get("title_en") or dom.get("title"),
                "questions": questions,
            }
        )
    return {
        "ok": True,
        "tool": "list_asq_questions",
        "age_months": age,
        "source": str(path),
        "title_en": data.get("title_en"),
        "domains": domains_out,
        "domain_count": len(domains_out),
        "question_count": sum(len(d["questions"]) for d in domains_out),
    }


def _resolve_mchat_path() -> Path | None:
    candidates = [
        EN_DIR / "screens" / "mchat-r.json",
        EXTRACTED / "screens" / "mchat-r.json",
        EN_DIR / "mchat-r.json",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None


def list_mchat_questions() -> dict[str, Any]:
    """Load M-CHAT-R question list from data/en or extracted/."""
    path = _resolve_mchat_path()
    if path is None:
        return tool_error("list_mchat_questions", "M-CHAT-R questionnaire file not found.")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return tool_error("list_mchat_questions", f"Failed to read M-CHAT file: {exc}")

    questions = []
    for q in data.get("questions", []):
        questions.append(
            {
                "id": q.get("id"),
                "text_en": q.get("text_en") or q.get("text") or "",
                "text_fa": q.get("text_fa"),
                "options_en": q.get("options_en") or ["Yes", "No"],
                "reverse_scored": int(q.get("id") or 0) in MCHAT_REVERSE,
            }
        )
    return {
        "ok": True,
        "tool": "list_mchat_questions",
        "source": str(path),
        "title_en": data.get("title_en"),
        "instructions_en": data.get("instructions_en"),
        "questions": questions,
        "question_count": len(questions),
        "reverse_scored_ids": sorted(MCHAT_REVERSE),
    }


def get_child_summary(child_id: str, db=None) -> dict[str, Any]:
    """Summarize child profile, recent growth, screenings, and chart overlays."""
    if not child_id or not str(child_id).strip():
        return tool_error("get_child_summary", "child_id is required.")
    cid = str(child_id).strip()

    db = db or _CHILD_DB
    if db is None:
        from assistant.memory.child_db import ChildMemoryDB

        db = ChildMemoryDB()

    child = db.get_child(cid)
    if not child:
        return tool_error("get_child_summary", f"Unknown child_id: {cid}")

    growth = db.growth_history(cid)
    screens = db.screenings(cid)
    latest_by_measure: dict[str, dict] = {}
    for g in growth:
        latest_by_measure[g["measure"]] = g

    ga = child.get("gestational_age_weeks")
    maturity = "preterm" if ga is not None and float(ga) < 37 else ("term" if ga is not None else "unknown")
    overlays = []
    try:
        for p in sorted(OVERLAY_DIR.glob(f"overlay_{cid}_*.png"), key=lambda x: x.stat().st_mtime, reverse=True)[:8]:
            overlays.append({"filename": p.name, "url": f"/api/overlays/{p.name}"})
    except Exception:
        pass

    lines = [
        f"{child.get('name')} ({child.get('sex')})",
        f"GA at birth: {ga} weeks → {maturity}",
        f"Growth points: {len(growth)}; screenings: {len(screens)}",
    ]
    for measure, g in latest_by_measure.items():
        weeks = g.get("weeks")
        age_bits = []
        if weeks is not None:
            try:
                w = float(weeks)
                age_bits.append(f"{w:.1f} weeks since birth" if maturity == "term" else f"{w:.1f}w PMA")
                if maturity == "term":
                    age_bits.append(f"≈{w / 4.345:.1f} months")
            except (TypeError, ValueError):
                age_bits.append(f"{weeks}")
        age_txt = ", ".join(age_bits) if age_bits else "age unknown"
        cent = g.get("centile")
        cent_txt = f"{float(cent):.1f}" if cent is not None else "—"
        lines.append(
            f"- latest {measure}: {g.get('value')} at {age_txt} "
            f"(centile≈{cent_txt}, {g.get('track_status')})"
        )
    for s in screens[-3:]:
        lines.append(f"- screening {s.get('instrument')}: {(s.get('result') or {}).get('summary')}")
    if overlays:
        lines.append(f"Saved charts: {', '.join(o['filename'] for o in overlays[:3])}")

    return {
        "ok": True,
        "tool": "get_child_summary",
        "child_id": cid,
        "profile": {
            "name": child.get("name"),
            "sex": child.get("sex"),
            "date_of_birth": child.get("date_of_birth"),
            "gestational_age_weeks": child.get("gestational_age_weeks"),
            "notes": child.get("notes"),
            "maturity": maturity,
        },
        "growth_count": len(growth),
        "latest_growth": latest_by_measure,
        "growth_history": growth[-10:],
        "screening_count": len(screens),
        "recent_screenings": [
            {
                "instrument": s.get("instrument"),
                "age_months": s.get("age_months"),
                "summary": (s.get("result") or {}).get("summary"),
                "recorded_at": s.get("recorded_at"),
            }
            for s in screens[-5:]
        ],
        "overlays": overlays,
        "summary": "\n".join(lines),
    }


TOOL_SPECS = [
    {
        "name": "growth_percentile",
        "description": (
            "Compute INTERGROWTH-21st preterm growth percentiles/z-scores using official equations. "
            "Use for weight/length/head circumference at postmenstrual weeks 27-64. Never invent numbers."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sex": {
                    "type": "string",
                    "description": "male or female (also boy/girl)",
                    "enum": ["male", "female", "boy", "girl"],
                },
                "measure": {
                    "type": "string",
                    "description": "weight | length | head_circumference",
                    "enum": ["weight", "length", "head_circumference"],
                },
                "weeks": {
                    "type": "number",
                    "description": f"Postmenstrual age in weeks ({WEEKS_MIN}-{WEEKS_MAX})",
                    "minimum": WEEKS_MIN,
                    "maximum": WEEKS_MAX,
                },
                "value": {
                    "type": "number",
                    "description": "Measured value (kg for weight, cm for length/HC)",
                },
                "percentile": {
                    "type": "number",
                    "description": "Optional requested percentile (0-100 exclusive) to evaluate",
                },
            },
            "required": ["sex", "measure", "weeks"],
        },
    },
    {
        "name": "overlay_growth_on_chart",
        "description": (
            "Place the child's measurement on the INTERGROWTH percentile chart image and return track assessment."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "sex": {"type": "string", "description": "male or female"},
                "measure": {
                    "type": "string",
                    "description": "weight | length | head_circumference",
                    "enum": ["weight", "length", "head_circumference"],
                },
                "weeks": {
                    "type": "number",
                    "description": f"Postmenstrual age ({WEEKS_MIN}-{WEEKS_MAX})",
                    "minimum": WEEKS_MIN,
                    "maximum": WEEKS_MAX,
                },
                "value": {"type": "number", "description": "Measured value (kg or cm)"},
                "child_id": {"type": "string", "description": "Optional child id for filename/history"},
                "history": {
                    "type": "array",
                    "description": "Optional prior points [{weeks, value}, ...]",
                    "items": {
                        "type": "object",
                        "properties": {
                            "weeks": {"type": "number"},
                            "value": {"type": "number"},
                        },
                    },
                },
            },
            "required": ["sex", "measure", "weeks", "value"],
        },
    },
    {
        "name": "score_asq_questionnaire",
        "description": "Score ASQ domain answers (Yes/Sometimes/Not yet) and flag referral domains.",
        "parameters": {
            "type": "object",
            "properties": {
                "domain_answers": {
                    "type": "object",
                    "description": "Map of domain_id to list of answers (yes/sometimes/not_yet)",
                    "additionalProperties": {
                        "type": "array",
                        "items": {"type": "string"},
                    },
                }
            },
            "required": ["domain_answers"],
        },
    },
    {
        "name": "score_mchat",
        "description": "Score M-CHAT-R Yes/No answers and return risk tier (low/medium/high).",
        "parameters": {
            "type": "object",
            "properties": {
                "answers": {
                    "type": "object",
                    "description": "Map of question_id (1-20, int as string) to yes/no",
                    "additionalProperties": {"type": "string"},
                }
            },
            "required": ["answers"],
        },
    },
    {
        "name": "list_asq_questions",
        "description": "List ASQ questionnaire questions for a given age in months (from data/en or extracted).",
        "parameters": {
            "type": "object",
            "properties": {
                "age_months": {
                    "type": "integer",
                    "description": "Child age in months for the ASQ form (e.g. 4, 12, 24)",
                }
            },
            "required": ["age_months"],
        },
    },
    {
        "name": "list_mchat_questions",
        "description": "List all M-CHAT-R screening questions from data/en or extracted.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
    {
        "name": "get_child_summary",
        "description": "Return a structured summary of a child's profile, growth, and screenings from memory.",
        "parameters": {
            "type": "object",
            "properties": {
                "child_id": {"type": "string", "description": "Child UUID / id"},
            },
            "required": ["child_id"],
        },
    },
]


def dispatch_tool(name: str, arguments: dict, db=None) -> dict:
    """Execute a tool by name. Returns error dicts instead of crashing for API use."""
    if db is not None:
        set_child_db(db)
    args = dict(arguments or {})
    try:
        if name == "growth_percentile":
            return growth_percentile(**{k: args[k] for k in args if k in {
                "sex", "measure", "weeks", "value", "percentile",
                "gestational_age_weeks", "age_months", "chart_standard",
            }})
        if name == "overlay_growth_on_chart":
            return overlay_growth_on_chart(**{k: args[k] for k in args if k in {
                "sex", "measure", "weeks", "value", "child_id", "history",
                "gestational_age_weeks", "age_months", "chart_standard",
            }})
        if name == "score_asq_questionnaire":
            return score_asq_questionnaire(args.get("domain_answers"))
        if name == "score_mchat":
            raw = args.get("answers", {})
            if not isinstance(raw, dict):
                return tool_error("score_mchat", "answers must be an object.")
            answers = {int(k): v for k, v in raw.items()}
            return score_mchat(answers)
        if name == "list_asq_questions":
            return list_asq_questions(args.get("age_months"))
        if name == "list_mchat_questions":
            return list_mchat_questions()
        if name == "get_child_summary":
            return get_child_summary(args.get("child_id", ""), db=db)
        return tool_error(name or "unknown", "unknown_tool", detail=f"Unknown tool: {name}")
    except TypeError as exc:
        return tool_error(name, f"Invalid arguments: {exc}", arguments=args)
    except Exception as exc:
        return tool_error(name, f"Tool execution failed: {exc}", arguments=args)
