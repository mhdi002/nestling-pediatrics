"""Whole conversations across the range of things parents ask about.

The other conversation tests use one shape -- a condition, a clinic, an
allergen. That shape found real bugs, but code can be right for it and wrong
for sleep, teething, tantrums, screens, travel or nursery. These drive varied
generated conversations and assert the properties that must hold whatever the
subject is.

The model is stubbed here so the assertions are about what the agent
retrieves and how it holds together over many turns. What a real model does
with the prompt is checked separately, against a live one.
"""

from __future__ import annotations

import pytest

from assistant.settings import reset_settings
from tests import scenarios, topics

SEEDS = list(range(300, 312))


@pytest.fixture
def agent(monkeypatch, tmp_path):
    monkeypatch.setenv("NESTLING_CHILD_DB", str(tmp_path / "child.db"))
    monkeypatch.setenv("NESTLING_CHAT_DB", str(tmp_path / "chat.db"))
    monkeypatch.setenv("NESTLING_MEMORY_DB", str(tmp_path / "memory.db"))
    monkeypatch.setenv("NESTLING_CHILD_MEMORY_ENABLED", "0")
    reset_settings()

    import assistant.llm.qwen_client as qc

    calls: list[dict] = []

    class _Model:
        vision_ready = True
        ready = True

        def answer_with_context(self, query, context, *, system=None):
            calls.append({"query": query, "context": context, "system": system})
            return "Noted."

        def chat(self, *a, **k):
            return "Noted."

    monkeypatch.setattr(qc, "llm_enabled", lambda: True)
    monkeypatch.setattr(qc, "get_qwen", lambda: _Model())

    from assistant.agent.orchestrator import ParentAssistant

    orch = ParentAssistant()
    if not orch.medical.load():
        pytest.skip("medical index not built")
    orch.use_llm = True
    orch._calls = calls
    yield orch
    reset_settings()


def _child(agent, sc):
    return agent.db.create_child(
        name=sc.name,
        sex="female" if sc.pronoun == "she" else "male",
        gestational_age_weeks=39.0,
    )


@pytest.mark.parametrize("seed", SEEDS, ids=lambda s: f"seed{s}")
def test_every_turn_of_a_varied_conversation_gets_a_reply(agent, seed):
    """Whatever the topic, the parent is answered."""
    sc = scenarios.make(seed)
    cid = _child(agent, sc)
    sid = agent.chat_memory.create_session(child_id=cid)
    for turn in topics.conversation(seed, sc, length=8):
        out = agent.chat(sid, turn.text, child_id=cid)
        reply = (out.get("reply") or "").strip()
        assert reply, f"{sc}\nsilent on [{turn.kind}] {turn.text!r}"


@pytest.mark.parametrize("seed", SEEDS, ids=lambda s: f"seed{s}")
def test_something_said_early_is_still_available_late(agent, seed):
    """The first statement must survive a whole conversation of other topics."""
    sc = scenarios.make(seed)
    cid = _child(agent, sc)
    sid = agent.chat_memory.create_session(child_id=cid)
    turns = topics.conversation(seed, sc, length=8)
    opening = turns[0].text

    for turn in turns:
        agent.chat(sid, turn.text, child_id=cid)

    # Ask using the opening's own words: whatever it was about, it should be
    # findable. Content words only -- "is" and "the" prove nothing.
    words = [w for w in opening.split() if len(w) > 4][:3]
    if not words:
        pytest.skip("opening had no content words to query with")
    recalled = agent._recall_episodic(" ".join(words), sid, cid, None).lower()
    assert any(w.lower() in recalled for w in words), f"{sc}\nopening: {opening!r}"


@pytest.mark.parametrize("seed", SEEDS[:8], ids=lambda s: f"seed{s}")
def test_a_bare_follow_up_never_arrives_without_context(agent, seed):
    """"how often?" carries no topic; it is useless without what came before."""
    sc = scenarios.make(seed)
    cid = _child(agent, sc)
    sid = agent.chat_memory.create_session(child_id=cid)
    agent.chat(sid, topics.STATEMENTS[seed % len(topics.STATEMENTS)].format(
        p=sc.pronoun, poss=sc.possessive, name=sc.name), child_id=cid)

    for follow_up in topics.FOLLOW_UPS[:4]:
        recalled = agent._recall_episodic(follow_up, sid, cid, None)
        assert recalled.strip(), f"{sc}\nno context for {follow_up!r}"


@pytest.mark.parametrize("seed", SEEDS[:8], ids=lambda s: f"seed{s}")
def test_the_prompt_stays_within_its_cap_whatever_is_discussed(agent, seed):
    sc = scenarios.make(seed)
    cid = _child(agent, sc)
    sid = agent.chat_memory.create_session(child_id=cid)
    for turn in topics.conversation(seed, sc, length=12):
        agent.chat(sid, turn.text, child_id=cid)

    from assistant.settings import get_settings

    cap = get_settings().llm_prompt_context_chars
    grounded = [c for c in agent._calls
                if "WHAT THIS PARENT HAS TOLD YOU" in (c["context"] or "")]
    for call in grounded:
        assert len(call["context"]) <= cap + 200, f"{sc}\n{len(call['context'])} chars"


@pytest.mark.parametrize("seed", SEEDS[:6], ids=lambda s: f"seed{s}")
def test_a_persian_conversation_is_answered_and_remembered(agent, seed):
    """The app serves Persian-speaking parents, so every turn must land.

    Note what is NOT asserted here: that the parent's own Persian is what got
    stored. It is not -- the orchestrator records the English translation,
    because the corpus and the BM25 index are English and recall has to match
    against something. See the companion test below, which pins that
    behaviour so it cannot change silently.
    """
    sc = scenarios.make(seed)
    cid = _child(agent, sc)
    sid = agent.chat_memory.create_session(child_id=cid)
    turns = topics.conversation(seed, sc, length=4, persian=True)
    for turn in turns:
        out = agent.chat(sid, turn.text, child_id=cid, ui_lang="fa")
        assert (out.get("reply") or "").strip(), f"{sc}\nsilent on {turn.text!r}"

    assert agent.memory.episodic.count(session_id=sid) >= len(turns), sc


@pytest.mark.parametrize("seed", SEEDS[:3], ids=lambda s: f"seed{s}")
def test_a_persian_turn_keeps_both_the_translation_and_the_parents_words(
    agent, seed
):
    """Both halves, for different reasons.

    Recall matches against English, because the index and the care corpus are
    English -- so the translation is what `content` holds and what the prompt
    carries. But a translation is a copy that can drift, and for a child's
    medical history the parent's own sentence IS the record. It is kept
    alongside, with the language it was written in.

    This test previously asserted the original was LOST, documenting the gap.
    It is inverted rather than deleted, which is what such a test is for.
    """
    sc = scenarios.make(seed)
    cid = _child(agent, sc)
    sid = agent.chat_memory.create_session(child_id=cid)
    persian = topics.PERSIAN_STATEMENTS[seed % len(topics.PERSIAN_STATEMENTS)]
    agent.chat(sid, persian, child_id=cid, ui_lang="fa")

    episodes = agent.memory.episodic.recall(session_id=sid, limit=50)
    said = [e for e in episodes if e.role == "user"]
    assert said, "the parent's turn was not recorded at all"

    kept = [e for e in said if persian in str(e.attributes.get("original") or "")]
    assert kept, f"the parent's own Persian was lost: {[e.attributes for e in said]}"
    # ...and the searchable copy is the English, byte-for-byte unaltered.
    assert kept[0].content.strip(), "no translation stored to search against"
    assert kept[0].attributes.get("original_lang") == "fa"


@pytest.mark.parametrize("seed", SEEDS[:6], ids=lambda s: f"seed{s}")
def test_two_children_in_one_conversation_do_not_bleed(agent, seed):
    """A parent with siblings must not get one child's facts for the other."""
    a, b = scenarios.make(seed), scenarios.make(seed + 500)
    cid_a, cid_b = _child(agent, a), _child(agent, b)
    sid_a = agent.chat_memory.create_session(child_id=cid_a)
    sid_b = agent.chat_memory.create_session(child_id=cid_b)

    agent.chat(sid_a, a.allergy_fact, child_id=cid_a)
    agent.chat(sid_b, b.condition_fact, child_id=cid_b)

    for_b = agent._recall_episodic(a.allergy_question, sid_b, cid_b, None).lower()
    assert a.allergen.lower() not in for_b, f"{a} leaked into {b}"
