"""Unit tests for assistant.agent.router.route_message / IntentDecision."""

from __future__ import annotations

from assistant.agent.router import IntentDecision, route_message

_VALID_SOURCES = {"rules", "regex", "hybrid", "llm"}


def _assert_decision_shape(d: IntentDecision) -> None:
    assert isinstance(d, IntentDecision)
    assert isinstance(d.intents, list)
    assert all(isinstance(i, str) for i in d.intents)
    assert isinstance(d.slots, dict)
    assert isinstance(d.confidence, float)
    assert 0.0 <= d.confidence <= 1.0
    assert d.source in _VALID_SOURCES
    assert isinstance(d.rationale, str)


def test_route_help_greeting():
    d = route_message("hi, how can you help me?")
    _assert_decision_shape(d)
    assert "help" in d.intents
    assert "medical" not in d.intents


def test_route_growth_overlay():
    d = route_message("boy weight 40 weeks 3.2 kg overlay")
    _assert_decision_shape(d)
    assert "growth" in d.intents


def test_route_medical_iron():
    d = route_message("tell me about iron")
    _assert_decision_shape(d)
    assert "medical" in d.intents


def test_route_analyze_on_track():
    d = route_message("is he on track?")
    _assert_decision_shape(d)
    assert "growth_analysis" in d.intents


def test_route_fa_speech_concern():
    d = route_message("پسرم حرف نمیزنه نگرانم")
    _assert_decision_shape(d)
    assert "medical" in d.intents or "screening" in d.intents
    assert "help" not in d.intents


def test_route_en_speech_not_help():
    d = route_message("my child cant talk")
    _assert_decision_shape(d)
    assert "medical" in d.intents
    assert "help" not in d.intents


def test_route_fa_feeding_not_growth():
    d = route_message("بچم غذا باید چی بخوره ؟")
    _assert_decision_shape(d)
    assert "medical" in d.intents
    assert "growth" not in d.intents
    assert "growth_analysis" not in d.intents


def test_route_en_feeding_not_growth():
    d = route_message("what should my baby eat")
    _assert_decision_shape(d)
    assert "medical" in d.intents
    assert "growth" not in d.intents


def test_route_scar_hand_is_medical_not_chat():
    d = route_message("she has a small scar in her hand what should i do ?")
    _assert_decision_shape(d)
    assert "medical" in d.intents
    assert "chat" not in d.intents
    assert "growth_analysis" not in d.intents


def test_route_speech_followup_dada_is_medical():
    d = route_message(
        "yes she says dada is that good so ?",
        prior_slots={
            "last_topic": "talk",
            "last_intents": ["medical", "screening"],
            "last_medical_query": "she cant talk well is that okey?",
        },
    )
    _assert_decision_shape(d)
    assert "medical" in d.intents
    assert "chat" not in d.intents


def test_route_topic_switch_iron_and_scar():
    prior = {
        "last_topic": "talk",
        "last_intents": ["medical", "screening"],
        "last_medical_query": "she cant talk well",
    }
    iron = route_message("how about her iron?", prior_slots=prior)
    assert "medical" in iron.intents
    scar = route_message("she has a small scar", prior_slots=prior)
    assert "medical" in scar.intents
    skin_follow = route_message(
        "is that okay?",
        prior_slots={
            "last_topic": "scar",
            "last_intents": ["medical"],
            "last_medical_query": "she has a small scar",
        },
    )
    assert "medical" in skin_follow.intents
    assert "chat" not in skin_follow.intents


def test_route_dynamic_fever_and_teething():
    fever_prior = {
        "last_topic": "fever",
        "last_intents": ["medical"],
        "last_medical_query": "my baby has a fever",
    }
    soft = route_message("is that okay?", prior_slots=fever_prior)
    assert "medical" in soft.intents
    teeth = route_message("she is teething a lot", prior_slots=fever_prior)
    assert "medical" in teeth.intents
    soft2 = route_message(
        "yes how long?",
        prior_slots={
            "last_topic": "teething",
            "last_intents": ["medical"],
            "last_medical_query": "she is teething a lot",
        },
    )
    assert "medical" in soft2.intents
    assert "chat" not in soft2.intents


def test_route_wound_injury_medical():
    for msg in (
        "my baby has a wound on her arm",
        "small cut on finger what should i do",
        "bruise on his hand",
        "زخم کوچیک روی دستش چیکار کنم",
    ):
        d = route_message(msg)
        assert "medical" in d.intents, msg


def test_intent_decision_model_defaults():
    d = IntentDecision()
    _assert_decision_shape(d)
    assert d.intents == []
    assert d.slots == {}
    assert d.source == "hybrid"
