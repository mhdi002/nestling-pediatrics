"""
An unrecognised symptom must reach retrieval, not the "I'm listening" menu.

A live session asked "جیش مونیکا زرده" and then "ادرارش زرده" (her urine is
yellow) and got the generic listening menu both times — despite the corpus
holding 8 chunks on jaundice and 22 mentioning urine. The medical intent is an
enumerated keyword list (iron|sleep|fever|rash|...), and no finite list can
cover every way a parent describes a symptom. The same failure occurred in
English ("yellow skin newborn jaundice"), so it was never a language problem.

Unmatched but substantive messages now go to retrieval, which is equipped to
judge relevance, rather than dead-ending.
"""

from __future__ import annotations

import pytest

from assistant.agent.intents import classify_intent
from assistant.settings import get_settings


@pytest.fixture(autouse=True)
def _fresh_settings():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.mark.parametrize(
    "message",
    [
        "جیش مونیکا زرده",          # her pee is yellow
        "ادرارش زرده",              # her urine is yellow (2 words - Persian is compact)
        "زردی پوست نوزاد",          # newborn skin yellowing
        "her urine is yellow",
        "yellow skin newborn jaundice",
        "my baby pee is dark yellow",
    ],
)
def test_unmatched_symptoms_reach_retrieval(message):
    assert "medical" in classify_intent(message), classify_intent(message)


@pytest.mark.parametrize("message", ["سلام", "بله", "ok", "آره"])
def test_short_utterances_stay_conversational(message):
    """Greetings and one-word replies must not trigger a retrieval round-trip."""
    assert "medical" not in classify_intent(message), classify_intent(message)


def test_persian_is_not_penalised_for_being_compact():
    """
    A word-count-only threshold ignores Persian: "ادرارش زرده" is a complete
    sentence in two words. The character threshold is what saves it.
    """
    assert "medical" in classify_intent("ادرارش زرده")
    assert len("ادرارش زرده".split()) == 2


def test_soft_followup_does_not_hijack_an_open_thread():
    """A clarification belongs to the thread already open."""
    prior = {"last_topic": "growth_analysis", "last_intents": ["growth_analysis"]}
    assert "medical" not in classify_intent("is that good", prior_slots=prior)
    assert "medical" not in classify_intent("so?", prior_slots=prior)


def test_new_symptom_mid_thread_still_reaches_retrieval():
    """
    The guard must not swallow a genuinely new concern raised during another
    conversation — the failure mode of gating on a care-keyword list.
    """
    prior = {"last_topic": "growth_analysis", "last_intents": ["growth_analysis"]}
    assert "medical" in classify_intent("ادرارش زرده", prior_slots=prior)
    assert "medical" in classify_intent("her urine is yellow", prior_slots=prior)


def test_thresholds_are_configurable(monkeypatch):
    """No magic numbers: the cutoffs come from settings."""
    monkeypatch.setenv("NESTLING_RETRIEVAL_FALLBACK_MIN_WORDS", "99")
    monkeypatch.setenv("NESTLING_RETRIEVAL_FALLBACK_MIN_CHARS", "999")
    get_settings.cache_clear()
    assert "medical" not in classify_intent("her urine is yellow")
