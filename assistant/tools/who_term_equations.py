"""
WHO Child Growth Standards (0–24 months) — term / طبیعی infants.

LMS parameters from WHO Child Growth Standards (weight-for-age, length-for-age,
head circumference-for-age). Used when gestational age at birth ≥ 37 weeks.

Z = ((value/M)^L - 1) / (L*S)  when L≠0;  else ln(value/M)/S
Centile via normal CDF.
"""

from __future__ import annotations

import math
from typing import Literal

Sex = Literal["male", "female"]
Measure = Literal["weight", "length", "head_circumference"]

# Monthly LMS: age_months -> (L, M, S). Compact published WHO anchors.
# Sources: WHO Child Growth Standards (2006).

_WHO_WFA_M: dict[float, tuple[float, float, float]] = {
    0: (-0.0631, 3.3464, 0.14602),
    1: (0.0348, 4.4709, 0.13395),
    2: (0.1296, 5.5675, 0.12385),
    3: (0.1967, 6.3762, 0.11727),
    4: (0.2374, 7.0023, 0.11316),
    5: (0.2601, 7.5105, 0.11080),
    6: (0.2701, 7.9340, 0.10958),
    9: (0.2497, 8.9328, 0.10919),
    12: (0.2036, 9.6479, 0.10859),
    18: (0.0804, 10.9046, 0.10837),
    24: (-0.0391, 12.1515, 0.10959),
}

_WHO_WFA_F: dict[float, tuple[float, float, float]] = {
    0: (-0.0631, 3.2322, 0.14171),
    1: (0.0018, 4.1871, 0.13724),
    2: (0.0544, 5.1282, 0.13002),
    3: (0.0944, 5.8458, 0.12619),
    4: (0.1199, 6.4237, 0.12402),
    5: (0.1339, 6.8985, 0.12274),
    6: (0.1395, 7.2970, 0.12204),
    9: (0.1258, 8.2254, 0.12181),
    12: (0.0902, 8.9481, 0.12215),
    18: (-0.0116, 10.1220, 0.12337),
    24: (-0.1257, 11.4800, 0.12579),
}

_WHO_LFA_M: dict[float, tuple[float, float, float]] = {
    0: (1.0, 49.8842, 0.03795),
    1: (1.0, 54.7244, 0.03557),
    2: (1.0, 58.4249, 0.03424),
    3: (1.0, 61.4292, 0.03328),
    4: (1.0, 63.8860, 0.03291),
    5: (1.0, 65.9026, 0.03291),
    6: (1.0, 67.6236, 0.03305),
    9: (1.0, 72.0, 0.0335),
    12: (1.0, 75.7488, 0.03448),
    18: (1.0, 82.0, 0.0355),
    24: (1.0, 87.0, 0.0360),
}

_WHO_LFA_F: dict[float, tuple[float, float, float]] = {
    0: (1.0, 49.1477, 0.03790),
    1: (1.0, 53.6872, 0.03640),
    2: (1.0, 57.0673, 0.03568),
    3: (1.0, 59.8029, 0.03520),
    4: (1.0, 62.0899, 0.03486),
    5: (1.0, 64.0301, 0.03463),
    6: (1.0, 65.7311, 0.03448),
    9: (1.0, 70.0, 0.0348),
    12: (1.0, 74.0, 0.0352),
    18: (1.0, 80.5, 0.0360),
    24: (1.0, 85.5, 0.0365),
}

_WHO_HC_M: dict[float, tuple[float, float, float]] = {
    0: (1.0, 34.4618, 0.03686),
    1: (1.0, 37.2759, 0.03135),
    2: (1.0, 39.1285, 0.02976),
    3: (1.0, 40.5135, 0.02904),
    6: (1.0, 43.3, 0.0285),
    12: (1.0, 46.1, 0.0280),
    24: (1.0, 48.4, 0.0275),
}

_WHO_HC_F: dict[float, tuple[float, float, float]] = {
    0: (1.0, 33.8787, 0.03496),
    1: (1.0, 36.5462, 0.03054),
    2: (1.0, 38.2521, 0.02933),
    3: (1.0, 39.5324, 0.02882),
    6: (1.0, 42.2, 0.0282),
    12: (1.0, 44.9, 0.0278),
    24: (1.0, 47.2, 0.0272),
}


def _tables(sex: Sex, measure: Measure) -> dict[float, tuple[float, float, float]]:
    male = sex == "male"
    if measure == "weight":
        return _WHO_WFA_M if male else _WHO_WFA_F
    if measure == "length":
        return _WHO_LFA_M if male else _WHO_LFA_F
    return _WHO_HC_M if male else _WHO_HC_F


def _interp_lms(table: dict[float, tuple[float, float, float]], months: float) -> tuple[float, float, float]:
    months = max(0.0, min(24.0, float(months)))
    keys = sorted(table.keys())
    if months <= keys[0]:
        return table[keys[0]]
    if months >= keys[-1]:
        return table[keys[-1]]
    for i in range(len(keys) - 1):
        a, b = keys[i], keys[i + 1]
        if a <= months <= b:
            t = (months - a) / (b - a) if b != a else 0.0
            La, Ma, Sa = table[a]
            Lb, Mb, Sb = table[b]
            return (La + t * (Lb - La), Ma + t * (Mb - Ma), Sa + t * (Sb - Sa))
    return table[keys[-1]]


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
    L, M, S = _interp_lms(_tables(sex, measure), age_months)
    y = float(value)
    if y <= 0:
        raise ValueError("value must be positive")
    if abs(L) < 1e-9:
        return math.log(y / M) / S
    return ((y / M) ** L - 1.0) / (L * S)


def centile_from_measurement(sex: Sex, measure: Measure, age_months: float, value: float) -> float:
    return 100.0 * _norm_cdf(z_score(sex, measure, age_months, value))


def percentile(sex: Sex, measure: Measure, age_months: float, p: float) -> float:
    L, M, S = _interp_lms(_tables(sex, measure), age_months)
    z = _norm_ppf(float(p) / 100.0)
    if abs(L) < 1e-9:
        return M * math.exp(S * z)
    return M * ((1.0 + L * S * z) ** (1.0 / L))


CHART_PERCENTILES = (3, 10, 50, 90, 97)


def classify_maturity(gestational_age_weeks: float | None) -> str:
    """Return 'preterm' (نارس) if GA < 37, else 'term' (طبیعی). Unknown → preterm-safe INTERGROWTH if PMA given."""
    if gestational_age_weeks is None:
        return "unknown"
    return "preterm" if float(gestational_age_weeks) < 37.0 else "term"
