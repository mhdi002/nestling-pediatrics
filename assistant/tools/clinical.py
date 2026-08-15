#!/usr/bin/env python3
"""Deterministic clinical tools — equations, scoring, overlays. No LLM math."""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Literal

os.environ.setdefault("MPLBACKEND", "Agg")

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import intergrowth_preterm_equations as ig
from assistant.tools import who_term_equations as who
from assistant.config import EN_DIR, EXTRACTED, OVERLAY_DIR
from assistant.refdata import asq_scoring, clinical_bounds, mchat_config, weeks_per_month
from assistant.settings import get_settings

log = logging.getLogger(__name__)

AnswerASQ = Literal["yes", "sometimes", "not_yet", "بله", "گاهی", "هنوز نه", "No", "Yes", "Sometimes", "Not yet"]


def _bounds() -> dict[str, Any]:
    return clinical_bounds()


def _wpm() -> float:
    return weeks_per_month()


# Clinical input bounds (from config/clinical_bounds.json)
_b0 = _bounds()
WEEKS_MIN = float(_b0.get("intergrowth_weeks_min", ig.AGE_WEEKS_MIN))
WEEKS_MAX = float(_b0.get("intergrowth_weeks_max", ig.AGE_WEEKS_MAX))
VALUE_RANGES = {
    k: (float(v[0]), float(v[1])) for k, v in _b0["value_ranges"].items()
}
WEEKS_PER_MONTH = _wpm()

# Chart presentation only — clinical percentile values come from the equations.
CHART_PERCENTILE_STYLES = {
    97: ("#c0392b", "-"),
    90: ("#2c3e50", "--"),
    50: ("#27ae60", "-"),
    10: ("#2c3e50", "--"),
    3: ("#c0392b", "-"),
}
CHILD_POINT_COLOR = "#2980b9"
CHART_GRID_ALPHA = 0.3

# A `weeks` value within this many weeks of age_months * weeks_per_month was
# derived from the chronological age, so it is postnatal life-weeks, not a
# postmenstrual age. Treating it as PMA (and subtracting GA) once reported a
# 13.5-month-old as ~7 months.
POSTNATAL_WEEKS_TOLERANCE = 0.75

# How much history the child summary carries back to the agent.
GROWTH_HISTORY_SUMMARY_POINTS = 10
SCREENING_SUMMARY_LINES = 3
RECENT_SCREENINGS = 5
SUMMARY_CHART_NAMES = 3

# Optional ChildMemoryDB injected by ParentAssistant for get_child_summary
_CHILD_DB = None


def _axis_points(lo: float, hi: float, step: float) -> list[float]:
    """Inclusive [lo, hi] sample points at `step` spacing for curve plotting."""
    if step <= 0 or hi < lo:
        return [float(lo)]
    n = int(round((hi - lo) / step))
    pts = [lo + i * step for i in range(n + 1)]
    if pts[-1] < hi - 1e-9:
        pts.append(hi)
    return pts


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
    if w != w:
        return tool_error("validation", f"Invalid weeks: {weeks!r}. Expected a finite number.")
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
    if v != v or v in (float("inf"), float("-inf")):
        return tool_error("validation", f"Invalid value: {value!r}. Expected a finite number.")
    bounds = VALUE_RANGES.get(measure)
    if bounds is None:
        return tool_error("validation", f"No configured value range for measure: {measure!r}.")
    lo, hi = bounds
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


def _asq_cutoff(domain: str | None = None, age_months: int | None = None) -> tuple[float, str]:
    cfg = asq_scoring()
    source = str(cfg.get("cutoff_source", "unverified_default"))
    domain_cutoffs = cfg.get("domain_cutoffs") or {}
    if age_months is not None and domain:
        key = f"{int(age_months)}m"
        by_age = domain_cutoffs.get(key) or domain_cutoffs.get(str(age_months))
        if isinstance(by_age, dict) and domain in by_age:
            return float(by_age[domain]), source
    fallback = _bounds().get("asq_default_cutoff", 30)
    return float(cfg.get("default_cutoff", fallback)), source


def score_asq_domain(
    answers: list[str],
    *,
    domain: str | None = None,
    age_months: int | None = None,
) -> dict[str, Any]:
    """Score one ASQ domain. Points: Yes/Sometimes/Not yet from config/asq_scoring.json."""
    if not isinstance(answers, list) or not answers:
        return tool_error("score_asq_domain", "answers must be a non-empty list of Yes/Sometimes/Not yet.")
    cfg = asq_scoring()
    yes_pts = int(cfg["score_yes"])
    sometimes_pts = int(cfg["score_sometimes"])
    not_yet_pts = int(cfg["score_not_yet"])
    points = []
    try:
        for ans in answers:
            key = _norm_asq_answer(ans)
            points.append(
                {"yes": yes_pts, "sometimes": sometimes_pts, "not_yet": not_yet_pts}[key]
            )
    except ValueError as exc:
        return tool_error("score_asq_domain", str(exc))
    total = sum(points)
    cutoff, cutoff_source = _asq_cutoff(domain=domain, age_months=age_months)
    return {
        "ok": True,
        "item_scores": points,
        "total": total,
        "max": yes_pts * len(answers),
        "cutoff": cutoff,
        "cutoff_source": cutoff_source,
        "below_cutoff": total < cutoff,
        "interpretation": "below_cutoff_refer" if total < cutoff else "above_cutoff_monitor",
    }


def score_asq_questionnaire(
    domain_answers: dict[str, list[str]],
    *,
    age_months: int | None = None,
) -> dict[str, Any]:
    """
    domain_answers: {domain_id: [answers...]} for communication, gross_motor, etc.
    Overall section is yes/no concern items — stored separately, not point-scored here.
    """
    if not isinstance(domain_answers, dict) or not domain_answers:
        return tool_error("score_asq_questionnaire", "domain_answers must be a non-empty object.")
    domains = {}
    referrals = []
    cutoff_source = asq_scoring().get("cutoff_source", "unverified_default")
    for dom, answers in domain_answers.items():
        if dom == "overall":
            continue
        result = score_asq_domain(answers, domain=dom, age_months=age_months)
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
        "cutoff_source": cutoff_source,
        "cutoff_note": asq_scoring().get("cutoff_note"),
        "age_months": age_months,
        "summary": (
            f"ASQ scored {len(domains)} domains; "
            f"{'referral suggested for: ' + ', '.join(referrals) if referrals else 'no domain below cutoff'}"
            + (f" [cutoffs: {cutoff_source}]" if cutoff_source != "official_asq3" else "")
        ),
    }


# M-CHAT-R reverse-scored items (Yes = fail) — from config/mchat.json
MCHAT_REVERSE = set(int(x) for x in mchat_config()["reverse_items"])


def score_mchat(answers: dict[int, str]) -> dict[str, Any]:
    """
    answers: {question_id: 'yes'|'no'|'آری'|'خیر'}
    Returns fail count and risk tier from config/mchat.json.
    """
    if not isinstance(answers, dict) or not answers:
        return tool_error("score_mchat", "answers must be a non-empty map of question_id → yes/no.")
    cfg = mchat_config()
    reverse = set(int(x) for x in cfg["reverse_items"])
    tiers = cfg["risk_tiers"]
    fails = []
    try:
        for qid, ans in answers.items():
            a = (ans or "").strip().lower()
            yes = a in {"yes", "y", "آری", "اري", "بله"}
            no = a in {"no", "n", "خیر", "خير", "نه"}
            if not yes and not no:
                return tool_error("score_mchat", f"Invalid M-CHAT answer for Q{qid}: {ans!r}")
            q = int(qid)
            if q in reverse:
                failed = yes
            else:
                failed = no
            if failed:
                fails.append(q)
    except (TypeError, ValueError) as exc:
        return tool_error("score_mchat", f"Invalid answers payload: {exc}")
    n = len(fails)
    if n <= int(tiers["low_max"]):
        risk = "low"
    elif n <= int(tiers["medium_max"]):
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
        "note": cfg.get("note")
        or "Medium risk typically requires M-CHAT-R/F follow-up interview; high risk → refer.",
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
    wpm = _wpm()
    bounds = _bounds()
    band = bounds.get("term_pma_band", [37, 45])
    full_term = float(bounds.get("full_term_weeks", 40))
    if gestational_age_weeks is not None:
        ga = float(gestational_age_weeks)
        if w + 1e-9 >= ga:
            return max(0.0, w - ga) / wpm
        return w / wpm
    if float(band[0]) <= w <= float(band[1]):
        return max(0.0, w - full_term) / wpm
    return w / wpm


def corrected_age_months(
    chronological_age_months: float,
    gestational_age_weeks: float | None,
) -> dict[str, Any]:
    """
    ASQ / developmental screening: for preterm infants (GA < 37), use corrected age
    until ~24 months chronological. Returns both ages and which to use for form selection.
    """
    chrono = float(chronological_age_months)
    bounds = _bounds()
    thr = float(bounds.get("preterm_ga_threshold_weeks", 37))
    full_term = float(bounds.get("full_term_weeks", 40))
    if gestational_age_weeks is None:
        return {
            "chronological_age_months": chrono,
            "corrected_age_months": None,
            "age_for_questionnaire": chrono,
            "used_corrected": False,
            "note": "Gestational age unknown — using chronological age for ASQ form selection.",
        }
    ga = float(gestational_age_weeks)
    if ga >= thr:
        return {
            "chronological_age_months": chrono,
            "corrected_age_months": chrono,
            "age_for_questionnaire": chrono,
            "used_corrected": False,
            "gestational_age_weeks": ga,
        }
    weeks_early = max(0.0, full_term - ga)
    corrected = max(0.0, chrono - weeks_early / _wpm())
    return {
        "chronological_age_months": chrono,
        "corrected_age_months": corrected,
        "age_for_questionnaire": corrected,
        "used_corrected": True,
        "gestational_age_weeks": ga,
        "weeks_premature": weeks_early,
        "note": (
            f"Preterm (GA {ga}w): ASQ form selected by corrected age "
            f"{corrected:.1f} mo (chronological {chrono:.1f} mo)."
        ),
    }


def resolve_chart_route(
    gestational_age_weeks: float | None = None,
    weeks: float | None = None,
    age_months: float | None = None,
    chart_standard: str | None = None,
) -> dict[str, Any]:
    """
    Pick preterm (نارس / INTERGROWTH PMA) vs term (طبیعی / WHO months).

    Unknown GA: chronological age_months → assume WHO term (disclosed);
    weeks without GA → refuse (PMA vs life-weeks ambiguous).
    """
    assumed_term = False
    if chart_standard in {"intergrowth_preterm", "who_term"}:
        std = chart_standard
        maturity = "preterm" if std == "intergrowth_preterm" else "term"
    else:
        maturity = who.classify_maturity(gestational_age_weeks)
        if maturity == "unknown":
            if age_months is not None and weeks is None:
                std = "who_term"
                maturity = "term"
                assumed_term = True
            elif weeks is not None:
                return tool_error(
                    "resolve_chart_route",
                    "Gestational age at birth is required when age is given in weeks "
                    "(could be postmenstrual age for a preterm baby, or weeks of life). "
                    "Please provide gestational_age_weeks, say preterm/term, or set "
                    "chart_standard to who_term or intergrowth_preterm.",
                    needs_gestational_age=True,
                )
            else:
                return tool_error(
                    "resolve_chart_route",
                    "Gestational age at birth is required to choose WHO (term) vs INTERGROWTH "
                    "(preterm) charts. Please provide gestational_age_weeks, or set "
                    "chart_standard explicitly to who_term or intergrowth_preterm.",
                    needs_gestational_age=True,
                )
        elif maturity == "term":
            std = "who_term"
        else:
            std = "intergrowth_preterm"

    who_lo = float(_bounds().get("who_age_months_min", 0))
    who_hi = float(_bounds().get("who_age_months_max", 24))

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
        if not (who_lo <= float(age_months) <= who_hi):
            return tool_error(
                "resolve_chart_route",
                f"WHO term charts support {who_lo}–{who_hi} months; got {age_months}.",
            )
        out = {
            "ok": True,
            "maturity": "term",
            "maturity_label_en": "term",
            "maturity_label_fa": "طبیعی",
            "chart_standard": "who_term",
            "age_months": float(age_months),
            "weeks": None,
        }
        if assumed_term:
            out["assumed_term"] = True
            out["assumption_note"] = (
                "Gestational age unknown; used WHO term charts. "
                "If the baby was born before 37 weeks, provide gestational age for INTERGROWTH."
            )
        return out

    # INTERGROWTH preterm — postmenstrual age (PMA).
    # Slot extraction often sets weeks = age_months * wpm (postnatal weeks of life).
    # Those must NOT be treated as PMA; prefer chronological age_months when present.
    wpm = _wpm()
    ig_lo = float(_bounds().get("intergrowth_weeks_min", 27))
    ig_hi = float(_bounds().get("intergrowth_weeks_max", 64))
    chrono: float | None = float(age_months) if age_months is not None else None
    pma: float | None = None

    weeks_looks_postnatal = False
    if (
        age_months is not None
        and weeks is not None
        and abs(float(weeks) - float(age_months) * wpm) <= POSTNATAL_WEEKS_TOLERANCE
    ):
        weeks_looks_postnatal = True

    if chrono is not None and gestational_age_weeks is not None:
        pma = float(gestational_age_weeks) + chrono * wpm
    elif weeks is not None and not weeks_looks_postnatal:
        pma = float(weeks)
        if gestational_age_weeks is not None and chrono is None:
            chrono = max(0.0, pma - float(gestational_age_weeks)) / wpm
    elif weeks is not None and weeks_looks_postnatal and gestational_age_weeks is not None:
        pma = float(gestational_age_weeks) + float(age_months) * wpm
    elif chrono is not None and gestational_age_weeks is None and weeks is None:
        return tool_error(
            "resolve_chart_route",
            "Preterm growth needs birth gestational age (or postmenstrual weeks).",
            needs_gestational_age=True,
        )

    if pma is None:
        return tool_error(
            "resolve_chart_route",
            "Preterm growth needs postmenstrual weeks (or age months + birth GA).",
        )

    # Past INTERGROWTH window → WHO chronological age (still disclose preterm maturity).
    if pma > ig_hi + 1e-9 and chrono is not None:
        if not (who_lo <= float(chrono) <= who_hi):
            return tool_error(
                "resolve_chart_route",
                f"Age {chrono:.1f} months is outside WHO {who_lo}–{who_hi} months "
                f"(and PMA {pma:.1f}w exceeds INTERGROWTH {ig_hi}w).",
            )
        return {
            "ok": True,
            "maturity": "preterm",
            "maturity_label_en": "preterm",
            "maturity_label_fa": "نارس",
            "chart_standard": "who_term",
            "age_months": float(chrono),
            "weeks": None,
            "pma_weeks": float(pma),
            "note": (
                f"PMA {pma:.1f}w exceeds INTERGROWTH range ({ig_lo}–{ig_hi}w); "
                f"using WHO charts at chronological {chrono:.1f} months."
            ),
        }

    weeks_n = _validate_weeks(pma)
    if isinstance(weeks_n, dict):
        return weeks_n
    if chrono is None and gestational_age_weeks is not None:
        chrono = max(0.0, float(weeks_n) - float(gestational_age_weeks)) / wpm
    return {
        "ok": True,
        "maturity": "preterm",
        "maturity_label_en": "preterm",
        "maturity_label_fa": "نارس",
        "chart_standard": "intergrowth_preterm",
        "weeks": weeks_n,
        # Always expose chronological months for medical/feeding (never leave null).
        "age_months": float(chrono) if chrono is not None else None,
    }


def _track_status(c: float) -> str:
    ts = _bounds().get("track_status", {})
    if c < float(ts.get("investigate_below", 3)):
        return "below_3rd_investigate"
    if c > float(ts.get("investigate_above", 97)):
        return "above_97th_investigate"
    if c < float(ts.get("monitor_below", 10)) or c > float(ts.get("monitor_above", 90)):
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
            chrono = route.get("age_months")
            if chrono is not None:
                age_label = f"{weeks_n}w PMA (~{float(chrono):.1f} mo chronological)"
            else:
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
    if route.get("assumed_term"):
        out["assumed_term"] = True
        out["assumption_note"] = route.get("assumption_note")
    if route.get("note"):
        out["note"] = route["note"]
    if route.get("pma_weeks") is not None:
        out["pma_weeks"] = route["pma_weeks"]
    if route["chart_standard"] == "who_term":
        try:
            pmeta = who.precision_meta(sex_n, meas_n, route["age_months"])
            out["precision_note"] = pmeta.get("precision_note")
            out["lms_anchors"] = pmeta.get("anchors")
            out["lms_anchor_sources"] = pmeta.get("anchor_sources")
        except Exception as exc:
            log.debug("WHO precision metadata unavailable: %s", exc)
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


def growth_percentile_curves(
    sex: str,
    measure: str,
    chart_standard: str | None = None,
    gestational_age_weeks: float | None = None,
    age_max: float | None = None,
) -> dict[str, Any]:
    """
    Percentile curve points (P3/10/50/90/97) for client SVG charts.

    WHO term → ages in months; INTERGROWTH preterm → postmenstrual ages in weeks.
    """
    sex_n = _validate_sex(sex)
    if isinstance(sex_n, dict):
        return sex_n
    meas_n = _validate_measure(measure)
    if isinstance(meas_n, dict):
        return meas_n

    if chart_standard in {"intergrowth_preterm", "who_term"}:
        std = chart_standard
    else:
        maturity = who.classify_maturity(gestational_age_weeks)
        std = "intergrowth_preterm" if maturity == "preterm" else "who_term"

    pcts = list(ig.CHART_PERCENTILES)
    settings = get_settings()
    step = (
        settings.chart_curve_step_months
        if std == "who_term"
        else settings.chart_curve_step_weeks
    )
    if step <= 0:
        return tool_error("growth_curves", "chart curve step must be positive.")

    if std == "who_term":
        who_lo, who_hi = who.age_bounds()
        hi = float(age_max) if age_max is not None else who_hi
        hi = max(who_lo, min(hi, who_hi))
        ages = [i * step for i in range(0, int(hi / step) + 1) if i * step <= hi + 1e-9]
        if ages and ages[-1] < hi:
            ages.append(hi)
        try:
            curves = {
                str(int(p)): [float(who.percentile(sex_n, meas_n, a, p)) for a in ages]
                for p in pcts
            }
        except Exception as exc:
            return tool_error("growth_curves", f"WHO curve evaluation failed: {exc}")
        return tool_ok(
            "growth_curves",
            {
                "sex": sex_n,
                "measure": meas_n,
                "chart_standard": std,
                "age_unit": "months",
                "ages": ages,
                "percentiles": pcts,
                "curves": curves,
                "units": "kg" if meas_n == "weight" else "cm",
                "gestational_age_weeks": gestational_age_weeks,
                "age_max": hi,
                "reference": "WHO Child Growth Standards (0–24 months)",
            },
        )

    # INTERGROWTH preterm — PMA weeks
    lo, hi_default = WEEKS_MIN, WEEKS_MAX
    hi = float(age_max) if age_max is not None else hi_default
    hi = max(lo, min(hi, hi_default))
    ages = [i * step for i in range(int(lo / step), int(hi / step) + 1) if i * step <= hi + 1e-9]
    if ages and ages[-1] < hi:
        ages.append(hi)
    try:
        curves = {
            str(int(p)): [float(ig.percentile(sex_n, meas_n, a, p)) for a in ages]
            for p in pcts
        }
    except Exception as exc:
        return tool_error("growth_curves", f"INTERGROWTH curve evaluation failed: {exc}")
    return tool_ok(
        "growth_curves",
        {
            "sex": sex_n,
            "measure": meas_n,
            "chart_standard": std,
            "age_unit": "weeks",
            "ages": ages,
            "percentiles": pcts,
            "curves": curves,
            "units": "kg" if meas_n == "weight" else "cm",
            "gestational_age_weeks": gestational_age_weeks,
            "age_max": hi,
            "reference": (
                "Villar et al., Lancet Glob Health 2015;3:e681-91 (INTERGROWTH-21st)"
            ),
        },
    )


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
        # Figure/FigureCanvasAgg instead of pyplot: pyplot keeps global figure
        # state that is not safe across the threadpool serving sync endpoints.
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except Exception as exc:
        return {
            **assessment,
            "tool": "overlay_growth_on_chart",
            "overlay_path": None,
            "plot_error": f"matplotlib unavailable: {exc}",
            "note": "Numeric assessment computed; chart image not generated.",
        }

    settings = get_settings()
    try:
        fig = Figure(figsize=(settings.chart_figsize_w, settings.chart_figsize_h))
        FigureCanvasAgg(fig)
        ax = fig.subplots()
        styles = CHART_PERCENTILE_STYLES
        std = assessment["chart_standard"]
        # Use the normalized sex/measure from the assessment: the caller may have
        # passed parent language ("boy"), which the equation tables do not accept.
        sex_n = assessment["sex"]
        measure_n = assessment["measure"]
        if std == "who_term":
            who_lo, who_hi = who.age_bounds()
            xs = _axis_points(who_lo, who_hi, settings.chart_curve_step_months)
            for p, (color, ls) in styles.items():
                ys = [who.percentile(sex_n, measure_n, m, p) for m in xs]
                ax.plot(xs, ys, color=color, linestyle=ls, linewidth=1.5, label=f"P{p}")
            x_child = assessment["age_months"]
            ax.plot([x_child], [value], "o", color=CHILD_POINT_COLOR, markersize=9, label="Child")
            ax.set_xlabel("Age (months)")
            title = f"WHO {measure_n} ({sex_n}, term) — child overlay"
            tag = f"{x_child:.1f}m"
        else:
            xs = _axis_points(WEEKS_MIN, WEEKS_MAX, settings.chart_curve_step_weeks)
            for p, (color, ls) in styles.items():
                ys = [ig.percentile(sex_n, measure_n, w, p) for w in xs]
                ax.plot(xs, ys, color=color, linestyle=ls, linewidth=1.5, label=f"P{p}")
            pts = list(history or []) + [{"weeks": assessment["weeks"], "value": value}]
            ax.plot(
                [p["weeks"] for p in pts],
                [p["value"] for p in pts],
                "o-",
                color=CHILD_POINT_COLOR,
                markersize=8,
                label="Child",
            )
            ax.set_xlabel("Postmenstrual age (weeks)")
            title = (
                f"INTERGROWTH-21st {measure_n} "
                f"({sex_n}, preterm) — child overlay"
            )
            tag = f"{assessment['weeks']}w"

        unit = assessment["units"]
        ax.set_ylabel(f"{measure_n} ({unit})")
        # Include value so parents can tell replots apart; we still replace older files.
        title = f"{title} · {value:g} {unit}"
        ax.set_title(title)
        ax.legend(loc="upper left")
        ax.grid(True, alpha=CHART_GRID_ALPHA)
        fig.tight_layout()
        measure_key = str(measure_n).replace(" ", "")
        value_tag = f"{value:g}".replace(".", "p")
        name = (
            f"overlay_{child_id or 'child'}_{measure_key}_{tag}_{value_tag}.png"
        ).replace(" ", "")
        path = OVERLAY_DIR / name
        # Keep one live chart per child+measure — remove stale conflicting overlays.
        if child_id:
            prefix = f"overlay_{child_id}_{measure_key}_"
            for old in OVERLAY_DIR.glob(f"{prefix}*.png"):
                if old.name == path.name:
                    continue
                try:
                    old.unlink()
                except OSError as exc:
                    log.debug("Could not remove stale overlay %s: %s", old, exc)
        fig.savefig(path, dpi=settings.chart_dpi)
    except Exception as exc:
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


def _nearest_asq_age(target_months: float) -> int | None:
    """Pick closest available ASQ interval to target age in months."""
    available: list[int] = []
    for base in (EN_DIR / "asq", EXTRACTED / "asq"):
        if not base.is_dir():
            continue
        for p in base.glob("*m.json"):
            try:
                available.append(int(p.name.replace("m.json", "")))
            except ValueError:
                continue
    if not available:
        return None
    available = sorted(set(available))
    return min(available, key=lambda a: abs(a - float(target_months)))


def list_asq_questions(
    age_months: int | float,
    *,
    gestational_age_weeks: float | None = None,
    use_corrected_age: bool = True,
) -> dict[str, Any]:
    """Load ASQ question text. For preterm infants, select form by corrected age."""
    try:
        chrono = float(age_months)
    except (TypeError, ValueError):
        return tool_error("list_asq_questions", f"Invalid age_months: {age_months!r}")
    asq_max = float(_bounds().get("asq_age_months_max", 72))
    if not (0 <= chrono <= asq_max):
        return tool_error(
            "list_asq_questions",
            f"age_months out of range: {chrono} (supported 0–{asq_max:g}).",
        )

    age_info: dict[str, Any] = {
        "chronological_age_months": chrono,
        "corrected_age_months": None,
        "age_for_questionnaire": chrono,
        "used_corrected": False,
    }
    if use_corrected_age and gestational_age_weeks is not None:
        age_info = corrected_age_months(chrono, gestational_age_weeks)

    form_age = _nearest_asq_age(float(age_info["age_for_questionnaire"]))
    if form_age is None:
        return tool_error(
            "list_asq_questions",
            f"No ASQ questionnaire found near {age_info['age_for_questionnaire']} months.",
            **age_info,
        )

    path = _resolve_asq_path(form_age)
    if path is None:
        return tool_error(
            "list_asq_questions",
            f"No ASQ questionnaire found for {form_age} months.",
            age_months=form_age,
            **age_info,
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
        "age_months": form_age,
        "requested_age_months": chrono,
        **age_info,
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
    latest_growth_by_measure: dict[str, dict] = {}
    for g in growth:
        latest_growth_by_measure[g["measure"]] = g

    ga = child.get("gestational_age_weeks")
    maturity = who.classify_maturity(ga)
    overlays = []
    try:
        # Latest overlay per measure only (avoid confusing duplicate/conflicting charts).
        latest_overlay_by_measure: dict[str, dict] = {}
        for p in sorted(
            OVERLAY_DIR.glob(f"overlay_{cid}_*.png"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        ):
            # overlay_{cid}_{measure}_{tag}_{value}.png — measure is token after cid.
            rest = p.name[len(f"overlay_{cid}_") :]
            measure = rest.split("_", 1)[0] if rest else "chart"
            if measure in latest_overlay_by_measure:
                continue
            latest_overlay_by_measure[measure] = {
                "filename": p.name,
                "url": f"/api/overlays/{p.name}",
                "measure": measure,
            }
        overlays = list(latest_overlay_by_measure.values())[
            : get_settings().nestling_tool_overlay_limit
        ]
    except OSError as exc:
        log.warning("Could not list overlays for child %s: %s", cid, exc)

    lines = [
        f"{child.get('name')} ({child.get('sex')})",
        f"GA at birth: {ga} weeks → {maturity}",
        f"Growth points: {len(growth)}; screenings: {len(screens)}",
    ]
    for measure, g in latest_growth_by_measure.items():
        weeks = g.get("weeks")
        age_bits = []
        if weeks is not None:
            try:
                w = float(weeks)
                age_bits.append(f"{w:.1f} weeks since birth" if maturity == "term" else f"{w:.1f}w PMA")
                if maturity == "term":
                    age_bits.append(f"≈{w / _wpm():.1f} months")
            except (TypeError, ValueError):
                age_bits.append(f"{weeks}")
        age_txt = ", ".join(age_bits) if age_bits else "age unknown"
        cent = g.get("centile")
        cent_txt = f"{float(cent):.1f}" if cent is not None else "—"
        lines.append(
            f"- latest {measure}: {g.get('value')} at {age_txt} "
            f"(centile≈{cent_txt}, {g.get('track_status')})"
        )
    for s in screens[-SCREENING_SUMMARY_LINES:]:
        lines.append(f"- screening {s.get('instrument')}: {(s.get('result') or {}).get('summary')}")
    if overlays:
        names = ", ".join(o["filename"] for o in overlays[:SUMMARY_CHART_NAMES])
        lines.append(f"Saved charts: {names}")

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
        "latest_growth": latest_growth_by_measure,
        "growth_history": growth[-GROWTH_HISTORY_SUMMARY_POINTS:],
        "screening_count": len(screens),
        "recent_screenings": [
            {
                "instrument": s.get("instrument"),
                "age_months": s.get("age_months"),
                "summary": (s.get("result") or {}).get("summary"),
                "recorded_at": s.get("recorded_at"),
            }
            for s in screens[-RECENT_SCREENINGS:]
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
        if name == "list_asq_questions":
            return list_asq_questions(
                args.get("age_months"),
                gestational_age_weeks=args.get("gestational_age_weeks"),
                use_corrected_age=args.get("use_corrected_age", True),
            )
        if name == "score_asq_questionnaire":
            return score_asq_questionnaire(
                args.get("domain_answers"),
                age_months=args.get("age_months"),
            )
        if name == "score_mchat":
            raw = args.get("answers", {})
            if not isinstance(raw, dict):
                return tool_error("score_mchat", "answers must be an object.")
            try:
                answers = {int(k): v for k, v in raw.items()}
            except (TypeError, ValueError):
                return tool_error(
                    "score_mchat", "answers keys must be question numbers (1-20)."
                )
            return score_mchat(answers)
        if name == "list_mchat_questions":
            return list_mchat_questions()
        if name == "get_child_summary":
            return get_child_summary(args.get("child_id", ""), db=db)
        return tool_error(name or "unknown", "unknown_tool", detail=f"Unknown tool: {name}")
    except TypeError as exc:
        return tool_error(name, f"Invalid arguments: {exc}", arguments=args)
    except Exception as exc:
        return tool_error(name, f"Tool execution failed: {exc}", arguments=args)
