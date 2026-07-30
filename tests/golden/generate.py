#!/usr/bin/env python3
"""Generate golden JSON snapshots for clinical tools. Run from repo root:

    python -m tests.golden.generate
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from assistant.tools.clinical import (  # noqa: E402
    growth_percentile,
    resolve_chart_route,
    score_asq_domain,
    score_asq_questionnaire,
    score_mchat,
)
from tests.golden.cases import (  # noqa: E402
    ASQ_DOMAIN_CASES,
    ASQ_Q_CASES,
    GROWTH_CASES,
    MCHAT_CASES,
    ROUTE_CASES,
)

OUT_DIR = Path(__file__).resolve().parent / "snapshots"


def _normalize(obj, ndigits: int = 6):
    if isinstance(obj, float):
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {str(k): _normalize(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_normalize(v, ndigits) for v in obj]
    return obj


def _strip_volatile(d: dict) -> dict:
    """Drop fields that may change without affecting clinical math."""
    skip = {
        "summary",
        "summary_fa",
        "note",
        "precision_note",
        "lms_anchors",
        "lms_anchor_sources",
        "cutoff_note",
    }
    return {k: v for k, v in d.items() if k not in skip}


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    catalog: dict[str, dict] = {}

    for case_id, kwargs in GROWTH_CASES:
        result = _strip_volatile(growth_percentile(**kwargs))
        catalog[case_id] = {"kind": "growth", "kwargs": kwargs, "result": _normalize(result)}

    for case_id, answers in ASQ_DOMAIN_CASES:
        result = _strip_volatile(score_asq_domain(answers))
        catalog[case_id] = {"kind": "asq_domain", "kwargs": {"answers": answers}, "result": _normalize(result)}

    for case_id, domain_answers in ASQ_Q_CASES:
        result = _strip_volatile(score_asq_questionnaire(domain_answers))
        catalog[case_id] = {
            "kind": "asq_q",
            "kwargs": {"domain_answers": domain_answers},
            "result": _normalize(result),
        }

    for case_id, answers in MCHAT_CASES:
        result = _strip_volatile(score_mchat(answers))
        catalog[case_id] = {
            "kind": "mchat",
            "kwargs": {"answers": {str(k): v for k, v in answers.items()}},
            "result": _normalize(result),
        }

    for case_id, kwargs in ROUTE_CASES:
        result = _strip_volatile(resolve_chart_route(**kwargs))
        catalog[case_id] = {"kind": "route", "kwargs": kwargs, "result": _normalize(result)}

    out_path = OUT_DIR / "clinical_golden.json"
    out_path.write_text(json.dumps(catalog, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {len(catalog)} cases → {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
