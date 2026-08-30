#!/usr/bin/env python3
"""Verify INTERGROWTH preterm equations against published reference values."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import intergrowth_preterm_equations as ig

# External validation lives in PUBLISHED_EXAMPLES below (value->centile anchors
# taken from the ki-tools/growthstandards docs). The self-consistency checks
# (roundtrip / monotonic / age-increasing / chart-sanity) run in main().
#
# NOTE: an earlier CHECKS table listed 50th-centile "checkpoints" with expected
# values left as None and was never iterated in main() -- it validated nothing
# while looking like it did, so it has been removed. Add real, sourced expected
# values here (not None placeholders) if median checkpoints are wanted.

# Self-consistency and published example from growthstandards docs:
# igprepost_value2centile(27*7, 0.99, wtkg, Male) ≈ 96.89
PUBLISHED_EXAMPLES = [
    {
        "name": "docs example male 27w 0.99kg ~97th",
        "sex": "male",
        "measure": "weight",
        "weeks": 27,
        "value": 0.99,
        "expected_centile": 96.89486,
        "tol": 0.05,
    },
    {
        "name": "docs example female 27w 0.91kg ~97th",
        "sex": "female",
        "measure": "weight",
        "weeks": 27,
        "value": 0.91,
        "expected_centile": 97.12034,
        "tol": 0.05,
    },
    {
        "name": "docs example female 64w median length ~64.68",
        "sex": "female",
        "measure": "length",
        "weeks": 64,
        "value": 64.68,
        "expected_z": 0.0,
        "tol_z": 0.05,
    },
]


def roundtrip_ok(sex, measure, weeks, p, tol_rel=1e-9) -> tuple[bool, str]:
    y = ig.percentile(sex, measure, weeks, p)
    c = ig.centile_from_measurement(sex, measure, weeks, y)
    z = ig.z_score(sex, measure, weeks, y)
    z_exp = ig.z_for_percentile(p)
    ok_c = abs(c - p) < 1e-6
    ok_z = abs(z - z_exp) < 1e-9
    msg = f"roundtrip {sex} {measure} @{weeks}w p{p}: y={y:.6f} centile={c:.6f} z={z:.6f}"
    return ok_c and ok_z, msg


def monotonic_ok(sex, measure, weeks) -> tuple[bool, str]:
    vals = [ig.percentile(sex, measure, weeks, p) for p in ig.CHART_PERCENTILES]
    ok = all(vals[i] < vals[i + 1] for i in range(len(vals) - 1))
    return ok, f"monotonic {sex} {measure} @{weeks}w: {vals}"


def age_increasing_ok(sex, measure, p=50) -> tuple[bool, str]:
    weeks = list(range(27, 65))
    vals = [ig.percentile(sex, measure, w, p) for w in weeks]
    # allow tiny numeric noise; generally strictly increasing for these standards
    ok = all(vals[i] <= vals[i + 1] + 1e-9 for i in range(len(vals) - 1))
    return ok, f"age↑ {sex} {measure} p{p}: {vals[0]:.3f} → {vals[-1]:.3f}"


def chart_sanity_ok() -> list[tuple[bool, str]]:
    """Loose visual ranges from INTERGROWTH weight boys chart."""
    results = []
    # At 27 weeks boys weight (model): 3rd ~0.55, 50th ~0.75, 97th ~0.99
    # (docs: 0.99 kg male @27w ≈ 97th centile)
    for p, lo, hi in [(3, 0.35, 0.65), (50, 0.65, 0.95), (97, 0.90, 1.15)]:
        y = ig.percentile("male", "weight", 27, p)
        ok = lo <= y <= hi
        results.append((ok, f"chart boys wt 27w p{p}: {y:.3f} in [{lo},{hi}]"))
    # At 64 weeks boys weight: 3rd ~6.3, 50th ~8.0, 97th ~10.4
    for p, lo, hi in [(3, 5.5, 7.0), (50, 7.2, 8.8), (97, 9.5, 11.2)]:
        y = ig.percentile("male", "weight", 64, p)
        ok = lo <= y <= hi
        results.append((ok, f"chart boys wt 64w p{p}: {y:.3f} in [{lo},{hi}]"))
    return results


def main() -> int:
    failures = []
    passes = []

    print(ig.equations_summary())
    print("\n=== Verification ===\n")

    for sex in ("male", "female"):
        for measure in ("weight", "length", "head_circumference"):
            for weeks in (27, 40, 50, 64):
                for p in ig.CHART_PERCENTILES:
                    ok, msg = roundtrip_ok(sex, measure, weeks, p)
                    (passes if ok else failures).append(msg)
                ok, msg = monotonic_ok(sex, measure, weeks)
                (passes if ok else failures).append(msg)
            ok, msg = age_increasing_ok(sex, measure)
            (passes if ok else failures).append(msg)

    for ex in PUBLISHED_EXAMPLES:
        if "expected_centile" in ex:
            c = ig.centile_from_measurement(ex["sex"], ex["measure"], ex["weeks"], ex["value"])
            ok = abs(c - ex["expected_centile"]) <= ex["tol"]
            msg = f"{ex['name']}: centile={c:.5f} expected≈{ex['expected_centile']} (±{ex['tol']})"
            (passes if ok else failures).append(msg)
        if "expected_z" in ex:
            z = ig.z_score(ex["sex"], ex["measure"], ex["weeks"], ex["value"])
            ok = abs(z - ex["expected_z"]) <= ex["tol_z"]
            msg = f"{ex['name']}: z={z:.5f} expected≈{ex['expected_z']} (±{ex['tol_z']})"
            (passes if ok else failures).append(msg)

    for ok, msg in chart_sanity_ok():
        (passes if ok else failures).append(msg)

    # Inverse: 50th percentile measurement should have z≈0
    for sex in ("male", "female"):
        y = ig.percentile(sex, "weight", 40, 50)
        z = ig.z_score(sex, "weight", 40, y)
        ok = abs(z) < 1e-9
        msg = f"median z≈0 {sex} weight@40w: z={z}"
        (passes if ok else failures).append(msg)

    # Optional plot
    out_dir = Path(__file__).resolve().parent / "extracted" / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        import matplotlib.pyplot as plt

        weeks = [w / 2 for w in range(27 * 2, 64 * 2 + 1)]
        fig, ax = plt.subplots(figsize=(10, 6))
        for p, color, style in [
            (97, "red", "-"),
            (90, "black", "--"),
            (50, "green", "-"),
            (10, "black", "--"),
            (3, "red", "-"),
        ]:
            ys = [ig.percentile("male", "weight", w, p) for w in weeks]
            ax.plot(weeks, ys, color=color, linestyle=style, label=f"p{p}")
        ax.set_xlabel("Postmenstrual age (weeks)")
        ax.set_ylabel("Weight (kg)")
        ax.set_title("INTERGROWTH-21st Weight (Boys) — model curves")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        plot_path = out_dir / "model_weight_boys.png"
        fig.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"Wrote plot: {plot_path}")
    except Exception as exc:
        print(f"(plot skipped: {exc})")

    print(f"Passed: {len(passes)}")
    print(f"Failed: {len(failures)}")
    for msg in failures:
        print("FAIL:", msg)
    if failures:
        return 1
    print("All verification checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
