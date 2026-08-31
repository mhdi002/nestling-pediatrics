"""Memory invariants across generated scenarios, not hand-picked examples.

Each test states a property that must hold for ANY child, condition, clinic,
allergen and medication -- drawn from vocabularies the implementation has not
been tuned against. A failure prints the seed that produced it.
"""

from __future__ import annotations

import pytest

from assistant.memory.backends.native import NativeMemoryBackend
from assistant.memory.system import MemorySystem
from assistant.settings import reset_settings
from tests import scenarios

OWNER = "owner-1"
COUNT = 25


@pytest.fixture
def mem(tmp_path, monkeypatch):
    monkeypatch.setenv("NESTLING_MEMORY_DB", str(tmp_path / "memory.db"))
    reset_settings()
    system = MemorySystem(NativeMemoryBackend(tmp_path / "memory.db"))
    yield system
    system.backend.close()
    reset_settings()


ALL = scenarios.many(COUNT)


@pytest.mark.parametrize("sc", ALL, ids=lambda s: f"seed{s.seed}")
def test_a_stated_fact_is_recalled_by_a_question_about_it(mem, sc):
    """Whatever the parent said, asking about it must find it."""
    child = f"child-{sc.seed}"
    for fact in sc.facts:
        mem.semantic.remember(fact, subject=child, owner_user_id=OWNER, use_llm=False)

    for question, expected in (
        (sc.where_question, sc.body_part),
        (sc.allergy_question, sc.allergen),
        (sc.medication_question, sc.medication),
    ):
        hits = mem.semantic.recall(
            subject=child, owner_user_id=OWNER, query=question, limit=4
        )
        joined = " ".join(h.text for h in hits).lower()
        assert expected.lower() in joined, f"{sc}\nQ: {question}\nGot: {joined!r}"


@pytest.mark.parametrize("sc", ALL, ids=lambda s: f"seed{s.seed}")
def test_relevance_beats_recency_whatever_the_topic(mem, sc):
    """The remembered detail must survive a wall of later small talk."""
    child = f"child-{sc.seed}"
    mem.observe(session_id="s1", role="user", content=sc.condition_fact,
                subject=child, owner_user_id=OWNER)
    for line in sc.chatter:
        mem.observe(session_id="s1", role="user", content=line,
                    subject=child, owner_user_id=OWNER)

    hits = mem.episodic.recall(
        session_id="s1", owner_user_id=OWNER, query=sc.where_question, limit=5
    )
    joined = " ".join(h.content for h in hits).lower()
    assert sc.condition.lower() in joined, f"{sc}\nGot: {joined!r}"


@pytest.mark.parametrize("sc", ALL, ids=lambda s: f"seed{s.seed}")
def test_a_later_session_still_remembers(mem, sc):
    """A new session starts empty; the child's thread did not."""
    child = f"child-{sc.seed}"
    mem.observe(session_id="first", role="user", content=sc.allergy_fact,
                subject=child, owner_user_id=OWNER)

    hits = mem.episodic.recall_across_sessions(
        subject=child, owner_user_id=OWNER, query=sc.allergy_question
    )
    joined = " ".join(h.content for h in hits).lower()
    assert sc.allergen.lower() in joined, f"{sc}\nGot: {joined!r}"


@pytest.mark.parametrize("sc", ALL, ids=lambda s: f"seed{s.seed}")
def test_no_account_ever_sees_another_s_child(mem, sc):
    """Scoping cannot depend on which words were used."""
    child = f"child-{sc.seed}"
    for fact in sc.facts:
        mem.semantic.remember(fact, subject=child, owner_user_id=OWNER, use_llm=False)
    assert mem.semantic.recall(subject=child, owner_user_id="intruder") == [], sc
    assert mem.episodic.recall(subject=child, owner_user_id="intruder") == [], sc


@pytest.mark.parametrize("sc", ALL, ids=lambda s: f"seed{s.seed}")
def test_the_context_budget_holds_whatever_is_remembered(mem, sc):
    """A long history must never push the prompt past its cap."""
    child = f"child-{sc.seed}"
    for fact in sc.facts:
        mem.semantic.remember(fact, subject=child, owner_user_id=OWNER, use_llm=False)
    for line in sc.chatter * 6:
        mem.observe(session_id="s1", role="user", content=line,
                    subject=child, owner_user_id=OWNER)

    ctx = mem.context_for(
        question=sc.where_question, session_id="s1", subject=child,
        owner_user_id=OWNER, intents={"medical"}, working="w" * 4000,
        total_chars=1500,
    )
    assert sum(ctx.used.values()) <= 1500, f"{sc}\n{ctx.used}"


@pytest.mark.parametrize("sc", ALL, ids=lambda s: f"seed{s.seed}")
def test_consolidation_captures_something_the_parent_stated(mem, sc, monkeypatch):
    """Without the model, statements must still become durable facts."""
    import assistant.llm.qwen_client as qc

    monkeypatch.setattr(qc, "llm_enabled", lambda: False)
    child = f"child-{sc.seed}"
    for fact in sc.facts:
        mem.observe(session_id="s1", role="user", content=fact,
                    subject=child, owner_user_id=OWNER)
    for line in sc.chatter:
        mem.observe(session_id="s1", role="user", content=line,
                    subject=child, owner_user_id=OWNER)

    stored = mem.maybe_consolidate(
        session_id="s1", subject=child, owner_user_id=OWNER, force=True
    )
    joined = " ".join(stored).lower()
    assert stored, f"{sc}\nnothing consolidated"
    # A question is not a fact, whatever it is about.
    assert not any(f.strip().endswith("?") for f in stored), f"{sc}\n{stored}"


# ---------------------------------------------------------------------------
# The profile graph, across generated scenarios
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sc", ALL, ids=lambda s: f"seed{s.seed}")
def test_the_graph_joins_a_condition_to_the_clinic(mem, sc):
    """Two facts, neither containing both words, must still connect."""
    child = f"child-{sc.seed}"
    mem.semantic.remember(sc.condition_fact, subject=child,
                          owner_user_id=OWNER, use_llm=False)
    mem.semantic.remember(sc.clinic_fact, subject=child,
                          owner_user_id=OWNER, use_llm=False)

    related = mem.semantic.related(
        subject=child,
        question=f"which clinic did we go to about {sc.possessive} {sc.condition}?",
        owner_user_id=OWNER,
    )
    assert sc.clinic.lower() in related.lower(), f"{sc}\nGot: {related!r}"


@pytest.mark.parametrize("sc", ALL, ids=lambda s: f"seed{s.seed}")
def test_the_graph_records_the_allergen(mem, sc):
    child = f"child-{sc.seed}"
    mem.semantic.remember(sc.allergy_fact, subject=child,
                          owner_user_id=OWNER, use_llm=False)
    labels = {
        n["label"].lower()
        for n in mem.semantic.graph.nodes(subject=child, owner_user_id=OWNER)
    }
    assert sc.allergen.lower() in labels, f"{sc}\nGot: {labels!r}"
