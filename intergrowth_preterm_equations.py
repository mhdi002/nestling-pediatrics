#!/usr/bin/env python3
"""
INTERGROWTH-21st International Postnatal Growth Standards for Preterm Infants.

Equations for median (mean on transformed scale) and SD follow Villar et al.,
Lancet Glob Health 2015;3:e681–91 (appendix), as implemented in the reference
ki-tools/growthstandards R package (igprepost_*).

Age: postmenstrual age in weeks (valid approximately 27–64 weeks; model
allows 24–64+6/7 weeks).

Measures:
  - weight (kg): log-normal model
  - length (cm): log-normal model
  - head_circumference (cm): normal (identity) model

Percentiles on charts: 3rd, 10th, 50th, 90th, 97th.
"""

from __future__ import annotations

from math import exp, log, sqrt
from typing import Iterable, Literal

try:
    from statistics import NormalDist
except ImportError:  # pragma: no cover
    NormalDist = None  # type: ignore

Sex = Literal["male", "female", "boy", "girl", "Male", "Female", "M", "F"]
Measure = Literal["weight", "length", "head_circumference", "wtkg", "lencm", "hcircm"]

_Z_CACHE = {
    3: -1.880793608151251,
    10: -1.2815515655446004,
    50: 0.0,
    90: 1.2815515655446004,
    97: 1.880793608151251,
}

CHART_PERCENTILES = (3, 10, 50, 90, 97)
AGE_WEEKS_MIN = 27.0
AGE_WEEKS_MAX = 64.0


def _norm_ppf(p: float) -> float:
    """Inverse CDF of standard normal for probability p in (0, 1)."""
    if NormalDist is not None:
        return NormalDist().inv_cdf(p)
    # Abramowitz & Stegun approximation fallback
    if p <= 0.0 or p >= 1.0:
        raise ValueError("p must be in (0, 1)")
    # rational approximation for erf inverse via Beasley-Springer-Moro-ish
    a = [
        -3.969683028665376e01,
        2.209460984245205e02,
        -2.759285104469687e02,
        1.383577518672690e02,
        -3.066479806614716e01,
        2.506628277459239e00,
    ]
    b = [
        -5.447609879822406e01,
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
        q = sqrt(-2 * log(p))
        return (
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    if p > phigh:
        q = sqrt(-2 * log(1 - p))
        return -(
            (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q + c[5])
            / ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1)
        )
    q = p - 0.5
    r = q * q
    return (
        (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5])
        * q
        / (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r + 1)
    )


def _norm_cdf(z: float) -> float:
    if NormalDist is not None:
        return NormalDist().cdf(z)
    # erf-based approximation
    t = 1.0 / (1.0 + 0.2316419 * abs(z))
    d = 0.3989423 * exp(-z * z / 2.0)
    p = d * t * (
        0.3193815
        + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274)))
    )
    return 1.0 - p if z > 0 else p


def normalize_sex(sex: Sex) -> str:
    s = str(sex).strip().lower()
    if s in {"m", "male", "boy", "پسر", "boys"}:
        return "male"
    if s in {"f", "female", "girl", "دختر", "girls"}:
        return "female"
    raise ValueError(f"Unknown sex: {sex!r}")


def normalize_measure(measure: Measure) -> str:
    m = str(measure).strip().lower()
    aliases = {
        "weight": "weight",
        "wtkg": "weight",
        "wt": "weight",
        "وزن": "weight",
        "length": "length",
        "lencm": "length",
        "len": "length",
        "قد": "length",
        "height": "length",
        "head_circumference": "head_circumference",
        "hcircm": "head_circumference",
        "hc": "head_circumference",
        "دور سر": "head_circumference",
        "head": "head_circumference",
    }
    if m not in aliases:
        raise ValueError(f"Unknown measure: {measure!r}")
    return aliases[m]


def _male_indicator(sex: str) -> float:
    return 1.0 if sex == "male" else 0.0


def mu_sd(sex: Sex, measure: Measure, weeks: float) -> tuple[float, float, str]:
    """
    Return (mu, sd, transform) on the model scale.

    transform is 'log' for weight/length and 'identity' for head circumference.
    For log models, mu is E[log(y)]; for identity, mu is E[y].
    """
    sex_n = normalize_sex(sex)
    meas = normalize_measure(measure)
    x = float(weeks)
    if x <= 0:
        raise ValueError("weeks must be positive")
    male = _male_indicator(sex_n)

    if meas == "weight":
        # log(weight kg)
        mu = 2.591277 - 0.01155 * (x**0.5) - 2201.705 * (x**-2) + 0.0911639 * male
        sd = 0.1470258 + 505.92394 * (x**-2) - 140.0576 * (x**-2) * log(x)
        return mu, sd, "log"
    if meas == "length":
        # log(length cm)
        mu = 4.136244 - 547.0018 * (x**-2) + 0.0026066 * x + 0.0314961 * male
        sd = 0.050489 + 310.44761 * (x**-2) - 90.0742 * (x**-2) * log(x)
        return mu, sd, "log"
    # head circumference cm (identity)
    mu = 55.53617 - 852.0059 * (x**-1) + 0.7957903 * male
    sd = 3.0582292 + 3910.05 * (x**-2) - 180.5625 * (x**-1)
    return mu, sd, "identity"


def mean(sex: Sex, measure: Measure, weeks: float) -> float:
    """Median / 50th percentile in original units (kg or cm)."""
    return percentile(sex, measure, weeks, 50)


def sd_original_approx(sex: Sex, measure: Measure, weeks: float) -> float:
    """
    Approximate SD on the original scale using (p84 - p50).
    For log models this is not the model SD; use mu_sd() for model-scale SD.
    """
    return percentile(sex, measure, weeks, 84.1345) - percentile(sex, measure, weeks, 50)


def z_for_percentile(p: float) -> float:
    p = float(p)
    if p in _Z_CACHE:
        return _Z_CACHE[p]
    if not 0.0 < p < 100.0:
        raise ValueError("percentile must be between 0 and 100 exclusive")
    return _norm_ppf(p / 100.0)


def percentile(sex: Sex, measure: Measure, weeks: float, p: float) -> float:
    """Return the value at percentile p (0–100) for given sex/measure/age."""
    mu, sd, transform = mu_sd(sex, measure, weeks)
    z = z_for_percentile(p)
    on_scale = mu + z * sd
    if transform == "log":
        return exp(on_scale)
    return on_scale


def z_score(sex: Sex, measure: Measure, weeks: float, value: float) -> float:
    """Z-score of a measurement."""
    mu, sd, transform = mu_sd(sex, measure, weeks)
    y = float(value)
    if transform == "log":
        if y <= 0:
            raise ValueError("value must be positive for log-scale measures")
        return (log(y) - mu) / sd
    return (y - mu) / sd


def centile_from_measurement(sex: Sex, measure: Measure, weeks: float, value: float) -> float:
    """Centile (0–100) corresponding to a measurement."""
    return 100.0 * _norm_cdf(z_score(sex, measure, weeks, value))


def chart_curves(
    sex: Sex,
    measure: Measure,
    weeks: Iterable[float] | None = None,
    percentiles: Iterable[float] = CHART_PERCENTILES,
) -> dict:
    """
    Evaluate chart percentile curves.

    Returns dict with keys: weeks, percentiles, values[p] -> list of y.
    """
    if weeks is None:
        weeks = [w / 2 for w in range(int(AGE_WEEKS_MIN * 2), int(AGE_WEEKS_MAX * 2) + 1)]
    weeks_list = [float(w) for w in weeks]
    pcts = [float(p) for p in percentiles]
    values = {p: [percentile(sex, measure, w, p) for w in weeks_list] for p in pcts}
    return {
        "sex": normalize_sex(sex),
        "measure": normalize_measure(measure),
        "weeks": weeks_list,
        "percentiles": pcts,
        "values": values,
        "units": {"weight": "kg", "length": "cm", "head_circumference": "cm"}[
            normalize_measure(measure)
        ],
        "reference": "Villar et al., Lancet Glob Health 2015;3:e681-91 (INTERGROWTH-21st)",
    }


def equations_summary() -> str:
    return """
INTERGROWTH-21st preterm postnatal standards (x = postmenstrual age in weeks)

WEIGHT (kg), log-normal:
  mu = 2.591277 - 0.01155*sqrt(x) - 2201.705/x^2 + 0.0911639*[male]
  sd = 0.1470258 + 505.92394/x^2 - 140.0576/x^2 * ln(x)
  weight = exp(mu + z*sd)

LENGTH (cm), log-normal:
  mu = 4.136244 - 547.0018/x^2 + 0.0026066*x + 0.0314961*[male]
  sd = 0.050489 + 310.44761/x^2 - 90.0742/x^2 * ln(x)
  length = exp(mu + z*sd)

HEAD CIRCUMFERENCE (cm), normal:
  mu = 55.53617 - 852.0059/x + 0.7957903*[male]
  sd = 3.0582292 + 3910.05/x^2 - 180.5625/x
  HC = mu + z*sd

Chart percentiles use z for 3, 10, 50, 90, 97.
""".strip()


if __name__ == "__main__":
    print(equations_summary())
    print()
    for sex in ("male", "female"):
        for meas in ("weight", "length", "head_circumference"):
            for w in (27, 40, 64):
                vals = {p: round(percentile(sex, meas, w, p), 3) for p in CHART_PERCENTILES}
                print(f"{sex:6} {meas:20} @{w}w: {vals}")
