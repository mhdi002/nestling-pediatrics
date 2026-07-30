"""
WHO Child Growth Standards (0–24 months) — term infants.

LMS parameters loaded from config/who_lms.json (verbatim from prior inline tables).
Z = ((value/M)^L - 1) / (L*S)  when L≠0;  else ln(value/M)/S
"""

from __future__ import annotations

import math
from typing import Literal

from assistant.refdata import who_lms as _who_lms_cfg

Sex = Literal["male", "female"]
Measure = Literal["weight", "length", "head_circumference"]


def _tables(sex: Sex, measure: Measure) -> dict[float, tuple[float, float, float]]:
    raw = _who_lms_cfg()["tables"][measure][sex]
    out: dict[float, tuple[float, float, float]] = {}
    for age_s, row in raw.items():
        out[float(age_s)] = (float(row["L"]), float(row["M"]), float(row["S"]))
    return out


def _anchor_sources(sex: Sex, measure: Measure) -> dict[float, str]:
    raw = _who_lms_cfg()["tables"][measure][sex]
    return {float(age_s): str(row.get("source", "unknown")) for age_s, row in raw.items()}


def age_bounds() -> tuple[float, float]:
    cfg = _who_lms_cfg()
    return float(cfg["age_months_min"]), float(cfg["age_months_max"])


class AgeOutOfRangeError(ValueError):
    """WHO LMS tables only cover a fixed age window."""


def _interp_lms(
    table: dict[float, tuple[float, float, float]],
    months: float,
    *,
    strict: bool = True,
) -> tuple[float, float, float, dict]:
    """
    Linear interpolate L, M, S.
    When strict=True (default), ages outside [min,max] raise AgeOutOfRangeError
    instead of silently clamping.
    """
    lo, hi = age_bounds()
    months_f = float(months)
    meta: dict = {"interpolated": False, "clamped": False, "anchors": []}
    if months_f < lo - 1e-9 or months_f > hi + 1e-9:
        if strict:
            raise AgeOutOfRangeError(
                f"WHO term charts support ages {lo}–{hi} months; got {months_f}."
            )
        months_f = max(lo, min(hi, months_f))
        meta["clamped"] = True

    keys = sorted(table.keys())
    if months_f <= keys[0]:
        meta["anchors"] = [keys[0]]
        L, M, S = table[keys[0]]
        return L, M, S, meta
    if months_f >= keys[-1]:
        meta["anchors"] = [keys[-1]]
        L, M, S = table[keys[-1]]
        return L, M, S, meta
    for i in range(len(keys) - 1):
        a, b = keys[i], keys[i + 1]
        if a <= months_f <= b:
            t = (months_f - a) / (b - a) if b != a else 0.0
            La, Ma, Sa = table[a]
            Lb, Mb, Sb = table[b]
            meta["interpolated"] = abs(t) > 1e-12 and abs(t - 1.0) > 1e-12
            meta["anchors"] = [a, b]
            return (La + t * (Lb - La), Ma + t * (Mb - Ma), Sa + t * (Sb - Sa), meta)
    L, M, S = table[keys[-1]]
    meta["anchors"] = [keys[-1]]
    return L, M, S, meta


def precision_meta(sex: Sex, measure: Measure, age_months: float) -> dict:
    """Provenance / precision note for API responses."""
    _, _, _, meta = _interp_lms(_tables(sex, measure), age_months, strict=True)
    sources = _anchor_sources(sex, measure)
    anchor_sources = [sources.get(a, "unknown") for a in meta["anchors"]]
    note_parts = []
    if meta["interpolated"]:
        note_parts.append(
            f"LMS interpolated between months {meta['anchors'][0]} and {meta['anchors'][1]}"
        )
    if any(s == "rounded_approximation" for s in anchor_sources):
        note_parts.append(
            "one or more LMS anchors are rounded approximations (not full WHO monthly tables)"
        )
    return {
        "interpolated": meta["interpolated"],
        "anchors": meta["anchors"],
        "anchor_sources": anchor_sources,
        "precision_note": "; ".join(note_parts) if note_parts else None,
    }


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def _norm_ppf(p: float) -> float:
    # Acklam approximation
    if p <= 0 or p >= 1:
        raise ValueError("p out of range")
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879557428e01,
        1.615858368580409e02,
        -1.556989798598866e02,
        6.680131188771972e01,
        -1.328068155288572e01,
    ]
    c = [
        -7.784894002430293e-03,
        -3.223964580411365e-01,
        -2.400758277161838e00,
        -2.549732539343734e00,
        4.374664141464968e00,
        2.938163982698783e00,
    ]
    d = [
        7.784695709041462e-03,
        3.224671290700398e-01,
        2.445134137142996e00,
        3.754408661907416e00,
    ]
    plow = 0.02425
    phigh = 1 - plow
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5]) / (
            (((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q
    ) / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)


def z_score(sex: Sex, measure: Measure, age_months: float, value: float) -> float:
    L, M, S, _ = _interp_lms(_tables(sex, measure), age_months)
    y = float(value)
    if y <= 0:
        raise ValueError("value must be positive")
    if abs(L) < 1e-9:
        return math.log(y / M) / S
    return ((y / M) ** L - 1.0) / (L * S)


def centile_from_measurement(sex: Sex, measure: Measure, age_months: float, value: float) -> float:
    return 100.0 * _norm_cdf(z_score(sex, measure, age_months, value))


def percentile(sex: Sex, measure: Measure, age_months: float, p: float) -> float:
    L, M, S, _ = _interp_lms(_tables(sex, measure), age_months)
    z = _norm_ppf(float(p) / 100.0)
    if abs(L) < 1e-9:
        return M * math.exp(S * z)
    return M * ((1.0 + L * S * z) ** (1.0 / L))


CHART_PERCENTILES = tuple(_who_lms_cfg().get("chart_percentiles", (3, 10, 50, 90, 97)))


def classify_maturity(gestational_age_weeks: float | None) -> str:
    """Return 'preterm' if GA < 37, 'term' if GA ≥ 37, 'unknown' if GA missing."""
    if gestational_age_weeks is None:
        return "unknown"
    from assistant.refdata import clinical_bounds

    thr = float(clinical_bounds().get("preterm_ga_threshold_weeks", 37))
    return "preterm" if float(gestational_age_weeks) < thr else "term"
