"""
The child summary a parent sees must be in their language and free of internals.

A live Persian session rendered the summary as an English block that also
listed a raw overlay filename:

    GA at birth: 32.0 weeks -> preterm
    Growth points: 1; screenings: 0
    - latest weight: 3.2 at 40.7w PMA (centile~43.0, within_10_90)
    Saved charts: overlay_765032b9-...-9fca460de790_weight_32.57w_3p2.png

`get_child_summary`'s `summary` string is an agent-facing digest; the
parent-facing text is now rendered from the structured fields instead.
"""

from __future__ import annotations

from assistant.parent_voice import child_summary_chat

RESULT = {
    "profile": {
        "name": "monika",
        "sex": "female",
        "gestational_age_weeks": 32.0,
        "maturity": "preterm",
    },
    "latest_growth": {
        "weight": {"value": 3.2, "centile": 42.78, "track_status": "within_10_90"}
    },
    "recent_screenings": [],
    "overlays": [{"filename": "overlay_765032b9-7c9a-4afb-8a93-9fca460de790_weight.png"}],
    "summary": "GA at birth: 32.0 weeks\nSaved charts: overlay_765032b9-....png",
}

INTERNAL_MARKERS = (
    "overlay_",
    ".png",
    "within_10_90",
    "GA at birth",
    "Growth points",
    "PMA",
    "centile≈",
)


def test_persian_summary_has_no_english_labels_or_filenames():
    text = child_summary_chat(RESULT, fa=True)
    for marker in INTERNAL_MARKERS:
        assert marker not in text, f"{marker!r} leaked into the Persian reply:\n{text}"
    assert "monika" in text
    # Persian content, not a translated-looking English skeleton.
    assert "وزن" in text and "صدک" in text


def test_english_summary_is_clean_prose():
    text = child_summary_chat(RESULT, fa=False)
    for marker in ("overlay_", ".png", "within_10_90", "centile≈"):
        assert marker not in text, f"{marker!r} leaked:\n{text}"
    assert "monika" in text
    assert "preterm" in text


def test_missing_measurements_reads_as_an_invitation():
    empty = {"profile": {"name": "Sara"}, "latest_growth": {}, "recent_screenings": []}
    for fa in (True, False):
        text = child_summary_chat(empty, fa=fa)
        assert "Sara" in text
        assert "None" not in text and "{}" not in text


def test_plain_string_input_still_supported():
    """Older callers pass the digest string; it must not crash."""
    text = child_summary_chat("some digest line", fa=False)
    assert "some digest line" in text


def test_opinion_requests_route_to_analysis():
    """'Any thoughts?' after a chart must not fall through to the generic menu."""
    from assistant.agent.rules import clear_rules_cache, match_intents_from_rules

    clear_rules_cache()
    for phrase in ("نظری نداری ؟", "نظرت چیه؟", "any thoughts?", "what do you think"):
        assert "growth_analysis" in match_intents_from_rules(phrase), phrase


def test_measure_labels_come_from_config_not_python():
    """
    Adding a measure to config/intent_rules.yaml must be enough to have it
    named correctly for parents. A second list inside Python would silently
    show the internal key until someone remembered to update it too.
    """
    import assistant.parent_voice as pv
    from assistant.agent.rules import clear_rules_cache, load_intent_rules

    clear_rules_cache()
    configured = (load_intent_rules().get("slots") or {}).get("measure") or {}
    assert configured, "expected slots.measure in config/intent_rules.yaml"

    for canonical, words in configured.items():
        label = pv._measure_label(canonical, fa=True)
        # The label must be one of the words the config supplies, never a
        # value invented in code.
        assert label in {str(w) for w in words} or label == canonical.replace("_", " "), (
            canonical,
            label,
        )

    # An unconfigured measure degrades to a readable key rather than raising.
    assert pv._measure_label("bmi", fa=True) == "bmi"
