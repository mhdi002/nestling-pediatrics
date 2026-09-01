"""The `urgent` intent: does it catch emergencies without catching everything?

The escalation this drives was switched off once already because its trigger
(a BM25 score comparison) called "what foods are good for her?" an emergency.
So the properties asserted here are about the whole class of parent turns, not
about a handful of sentences:

  * ANY generated emergency report escalates -- in English or Persian, with
    the LLM sidecar unavailable, which is the state these tests run in.
  * NO generated ordinary question escalates. This is the one that matters.
    A false positive here is the failure that killed the feature.
  * The red-flag classes are the corpus's. Every `corpus_terms` entry in
    config/urgency_signs.yaml has to be a word WHO's own danger-sign items in
    this corpus actually use, so a class cannot be invented in config.

The turns come from tests/urgency_scenarios.py, which composes them from
vocabularies the detector was not written against.
"""

from __future__ import annotations

import pytest

from assistant.agent import urgency
from assistant.agent.router import route_message
from assistant.settings import get_settings, reset_settings
from tests import urgency_scenarios as scenarios

EMERGENCIES = scenarios.emergencies(0)
ORDINARY = scenarios.ordinary(0)


# --- the corpus, not a word list ---------------------------------------


def _corpus_items():
    items = urgency.danger_sign_items()
    if not items:
        pytest.skip("knowledge chunks not built")
    return items


def test_the_corpus_actually_enumerates_danger_signs():
    """The provenance selectors in config still match sections in the corpus."""
    items = _corpus_items()
    assert len(items) > 20, f"only {len(items)} danger-sign items extracted"


def test_every_red_flag_class_is_keyed_on_a_word_the_corpus_uses():
    """A class cannot be invented in config for a topic WHO does not flag.

    This is what keeps config/urgency_signs.yaml a description of the corpus
    rather than a list of frightening words somebody liked.
    """
    corpus = urgency.danger_sign_terms()
    _corpus_items()
    signs = urgency.urgency_signs()["signs"]
    assert signs
    for name, spec in signs.items():
        terms = spec.get("corpus_terms") or []
        assert terms, f"sign {name!r} declares no corpus_terms"
        missing = [t for t in terms if t.lower() not in corpus]
        assert not missing, f"sign {name!r} claims words the corpus never uses: {missing}"


def test_severity_markers_are_the_corpus_own_qualifiers():
    """"Difficulty breathing", "Severe chest indrawing", "Heavy bleeding"."""
    _corpus_items()
    severity = (urgency.urgency_signs()["impairment_markers"]["severity"].get("en")) or []
    assert severity
    missing = [w for w in severity if not urgency.corpus_attests(w, child_only=False)]
    assert not missing, f"severity markers the danger-sign items never use: {missing}"


# --- the properties ----------------------------------------------------


@pytest.mark.parametrize("message", EMERGENCIES)
def test_every_generated_emergency_is_recognised(message):
    assert urgency.is_urgent(message), f"missed emergency: {message!r}"


@pytest.mark.parametrize("message", ORDINARY)
def test_no_generated_ordinary_question_escalates(message):
    signs = urgency.urgent_signs(message)
    assert not signs, f"ordinary question escalated as {signs}: {message!r}"


def test_measured_rates_on_the_generated_set():
    """Detection and false-positive rates, reported by the failure message."""
    missed = [m for m in EMERGENCIES if not urgency.is_urgent(m)]
    false = [m for m in ORDINARY if urgency.is_urgent(m)]
    detection = 1.0 - len(missed) / len(EMERGENCIES)
    false_rate = len(false) / len(ORDINARY)
    assert false_rate == 0.0, (
        f"false positives {len(false)}/{len(ORDINARY)}: {false[:5]}"
    )
    assert detection == 1.0, f"missed {len(missed)}/{len(EMERGENCIES)}: {missed[:5]}"


def test_naming_a_danger_sign_inside_a_question_is_not_reporting_one():
    """The structural half of the signal, stated as a property.

    Every emergency report, turned into a question about the same sign, must
    stop escalating. This is the failure mode that disabled the feature: the
    words are identical, only the grammar differs.
    """
    for report in EMERGENCIES[:60]:
        asked = f"what should I do if {report}"
        assert not urgency.urgent_signs(asked), f"question escalated: {asked!r}"


def test_a_general_question_about_babies_is_never_a_report():
    """No determiner, no specific child, no emergency."""
    for phrase in ("convulsions", "not breathing", "blue lips", "unconscious"):
        assert not urgency.urgent_signs(f"when do babies have {phrase}")
        assert not urgency.urgent_signs(f"how common is {phrase} in newborns")


def test_persian_is_recognised_without_translation():
    """The FA→EN hop is an outside HTTP call; urgency must not depend on it."""
    persian = [m for m in EMERGENCIES if not m.isascii()]
    assert len(persian) > 20, "generated set lost its Persian half"
    for message in persian:
        assert urgency.is_urgent(message), f"missed Persian emergency: {message!r}"


# --- the intent ---------------------------------------------------------


def test_router_marks_an_emergency_urgent_with_no_model_available():
    """The sidecar is down in this test run; the intent still appears."""
    for message in EMERGENCIES[:40]:
        decision = route_message(message)
        assert urgency.INTENT in decision.intents, f"router missed: {message!r}"


def test_router_leaves_ordinary_turns_alone():
    for message in ORDINARY:
        decision = route_message(message)
        assert urgency.INTENT not in decision.intents, f"router escalated: {message!r}"


def test_escalation_is_on():
    reset_settings()
    assert get_settings().nestling_urgent_escalation_enabled is True


def test_the_escalation_switch_only_gates_the_reply(monkeypatch):
    """Turning it off must not turn off recognising the emergency."""
    from assistant.rag import stores

    monkeypatch.setenv("NESTLING_URGENT_ESCALATION_ENABLED", "0")
    reset_settings()
    try:
        assert not stores._urgent_question("my baby is not breathing")
        assert urgency.is_urgent("my baby is not breathing")
    finally:
        monkeypatch.delenv("NESTLING_URGENT_ESCALATION_ENABLED", raising=False)
        reset_settings()
