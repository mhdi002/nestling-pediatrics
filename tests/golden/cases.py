"""Shared case list for golden clinical snapshots (generator + tests)."""

from __future__ import annotations

from typing import Any

# Each case: (case_id, kind, kwargs)
# kind ∈ growth | asq_domain | asq_q | mchat | route

GROWTH_CASES: list[tuple[str, dict[str, Any]]] = [
    # INTERGROWTH published checkpoints (explicit chart_standard — no silent GA default)
    ("ig_male_27w_weight_0.99", {"sex": "male", "measure": "weight", "weeks": 27, "value": 0.99, "chart_standard": "intergrowth_preterm"}),
    ("ig_female_27w_weight_0.91", {"sex": "female", "measure": "weight", "weeks": 27, "value": 0.91, "chart_standard": "intergrowth_preterm"}),
    ("ig_female_64w_length_64.68", {"sex": "female", "measure": "length", "weeks": 64, "value": 64.68, "chart_standard": "intergrowth_preterm"}),
    ("ig_male_40w_weight_3.2", {"sex": "male", "measure": "weight", "weeks": 40, "value": 3.2, "chart_standard": "intergrowth_preterm"}),
    ("ig_male_32w_hc_29", {"sex": "male", "measure": "head_circumference", "weeks": 32, "value": 29.0, "chart_standard": "intergrowth_preterm"}),
    ("ig_female_50w_weight_5.5", {"sex": "female", "measure": "weight", "weeks": 50, "value": 5.5, "chart_standard": "intergrowth_preterm"}),
    # WHO term
    ("who_male_0m_weight_3.3", {"sex": "male", "measure": "weight", "age_months": 0, "value": 3.3, "gestational_age_weeks": 39}),
    ("who_male_0.23m_weight_3.2", {"sex": "male", "measure": "weight", "weeks": 40, "value": 3.2, "gestational_age_weeks": 39}),
    ("who_female_6m_weight_7.3", {"sex": "female", "measure": "weight", "age_months": 6, "value": 7.3, "gestational_age_weeks": 40}),
    ("who_male_12m_length_76", {"sex": "male", "measure": "length", "age_months": 12, "value": 76.0, "gestational_age_weeks": 38}),
    ("who_female_12m_hc_45", {"sex": "female", "measure": "head_circumference", "age_months": 12, "value": 45.0, "gestational_age_weeks": 40}),
    ("who_male_24m_weight_12.2", {"sex": "male", "measure": "weight", "age_months": 24, "value": 12.2, "gestational_age_weeks": 40}),
    # Chart percentiles only (no value)
    ("ig_male_40w_chart_only", {"sex": "male", "measure": "weight", "weeks": 40, "chart_standard": "intergrowth_preterm"}),
    ("who_female_3m_chart_only", {"sex": "female", "measure": "length", "age_months": 3, "gestational_age_weeks": 39}),
]

ASQ_DOMAIN_CASES: list[tuple[str, list[str]]] = [
    ("asq_all_yes_6", ["yes"] * 6),
    ("asq_all_not_yet_6", ["not_yet"] * 6),
    ("asq_mixed_6", ["yes", "sometimes", "not_yet", "yes", "sometimes", "yes"]),
    ("asq_borderline_30", ["yes", "yes", "yes", "not_yet", "not_yet", "not_yet"]),  # 30
    ("asq_below_25", ["yes", "yes", "sometimes", "not_yet", "not_yet", "not_yet"]),  # 25
]

ASQ_Q_CASES: list[tuple[str, dict[str, list[str]]]] = [
    (
        "asq_q_two_domains",
        {
            "communication": ["yes"] * 6,
            "gross_motor": ["not_yet"] * 6,
            "overall": ["yes", "no"],  # ignored
        },
    ),
]

MCHAT_CASES: list[tuple[str, dict[int, str]]] = [
    ("mchat_all_pass", {i: ("no" if i in {2, 5, 12} else "yes") for i in range(1, 21)}),
    ("mchat_low_1fail", {**{i: ("no" if i in {2, 5, 12} else "yes") for i in range(1, 21)}, 1: "no"}),
    ("mchat_medium_5fail", {**{i: ("no" if i in {2, 5, 12} else "yes") for i in range(1, 21)}, **{j: "no" for j in (1, 3, 4, 6, 7)}}),
    ("mchat_high_reverse_fail", {i: "yes" for i in range(1, 21)}),  # reverse items fail
]

ROUTE_CASES: list[tuple[str, dict[str, Any]]] = [
    ("route_term_ga39_weeks40", {"gestational_age_weeks": 39, "weeks": 40}),
    ("route_preterm_ga32_weeks40", {"gestational_age_weeks": 32, "weeks": 40}),
    ("route_term_age_months_6", {"gestational_age_weeks": 40, "age_months": 6}),
    ("route_explicit_who", {"chart_standard": "who_term", "age_months": 3}),
    ("route_explicit_ig", {"chart_standard": "intergrowth_preterm", "weeks": 36}),
]
