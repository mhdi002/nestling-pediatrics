"""A recall question must reach memory, not the record dump.

"history" means show me the file. HISTORY_RE also matched a bare "remind me",
so "remind me where her ulcer was" was answered with the child's profile and
never reached the memory-grounded path -- even though episodic recall ranks
the right turn first for exactly that question.
"""

from __future__ import annotations

import pytest

from assistant.agent.intents import _asks_about_the_record, classify_intent

# Naming something specific: these want an answer, not a profile.
RECALL = [
    "remind me where her ulcer was",
    "remind me what she is allergic to",
    "do you remember which clinic we went to",
    "remember what I said about his rash",
    "remind me about her bronchiolitis",
    "remind me which medicine he is taking",
]

# Referring to the record itself: these want the file.
RECORD = [
    "show me my child's profile",
    "what do you know about my baby",
    "child profile",
    "show my child data",
    "history",
    "remind me",
    "show my baby record",
]


@pytest.mark.parametrize("question", RECALL)
def test_a_recall_question_does_not_become_a_record_dump(question):
    assert "history" not in classify_intent(question), question


@pytest.mark.parametrize("question", RECORD)
def test_a_request_for_the_record_still_gets_the_record(question):
    assert "history" in classify_intent(question), question


@pytest.mark.parametrize("question", RECALL)
def test_recall_questions_reach_a_path_that_carries_memory(question):
    """Routing them away from history is only useful if they land somewhere."""
    assert classify_intent(question) & {"medical", "chat"}, question


def test_phrases_history_never_claimed_are_left_alone():
    """Only messages HISTORY_RE matches are re-routed by this change.

    "show me everything" and "what did I tell you" were already medical
    before it, because HISTORY_RE requires "child"/"baby" after "show me".
    Recorded so a future widening of HISTORY_RE does not change them silently.
    """
    from assistant.agent.intents import HISTORY_RE

    for phrase in ("show me everything", "what did I tell you"):
        assert not HISTORY_RE.search(phrase), phrase
        assert "history" not in classify_intent(phrase), phrase


def test_the_split_is_about_what_is_named_not_which_verb_was_used():
    assert _asks_about_the_record("remind me") is True
    assert _asks_about_the_record("remind me where her ulcer was") is False
    # Record vocabulary wins even when something else is named.
    assert _asks_about_the_record("show me the growth chart data") is True


def test_a_persian_recall_question_is_not_treated_as_a_record_request():
    """Stripping the English framing must not empty a Persian message."""
    assert _asks_about_the_record("یادت هست کهیر داشت") is False
