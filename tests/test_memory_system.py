"""The four-layer memory system: procedural, semantic, episodic, consolidation."""

from __future__ import annotations

import pytest

from assistant.memory.assembly import (
    EPISODIC_HEADING,
    PROCEDURAL_HEADING,
    SEMANTIC_HEADING,
    WORKING_HEADING,
    assemble,
    budget,
)
from assistant.memory.backends.native import NativeMemoryBackend
from assistant.memory.system import MemorySystem
from assistant.memory.types import EPISODIC, PROCEDURAL, SEMANTIC, Episode, MemoryRecord
from assistant.settings import reset_settings


@pytest.fixture
def mem(tmp_path, monkeypatch):
    monkeypatch.setenv("NESTLING_MEMORY_DB", str(tmp_path / "memory.db"))
    reset_settings()
    system = MemorySystem(NativeMemoryBackend(tmp_path / "memory.db"))
    yield system
    system.close()
    reset_settings()


OWNER = "user-1"
CHILD = "child-1"


# ---------------------------------------------------------------------------
# Semantic
# ---------------------------------------------------------------------------


def test_a_fact_survives_and_is_recalled_by_relevance(mem):
    mem.semantic.remember("she has an ulcer in her stomach", subject=CHILD, owner_user_id=OWNER)
    mem.semantic.remember("we go to Mehr hospital", subject=CHILD, owner_user_id=OWNER)
    mem.semantic.remember("he sleeps through the night", subject=CHILD, owner_user_id=OWNER)

    hits = mem.semantic.recall(subject=CHILD, owner_user_id=OWNER, query="where was her ulcer")
    assert hits, "no facts recalled"
    assert "ulcer" in hits[0].text


def test_a_superseded_fact_is_no_longer_asserted_but_is_still_history(mem):
    """A rash clears; the record of it should not vanish, only stop being current."""
    old = mem.semantic.remember("she has a rash on her arm", subject=CHILD, owner_user_id=OWNER)
    mem.semantic.remember(
        "the rash has cleared", subject=CHILD, owner_user_id=OWNER, supersedes=old.id
    )
    current = mem.semantic.recall(subject=CHILD, owner_user_id=OWNER)
    assert all("has a rash" not in r.text for r in current)
    everything = mem.backend.recall_facts(CHILD, owner_user_id=OWNER, include_superseded=True)
    assert any("has a rash" in r.text for r in everything), "history was destroyed"


def test_one_account_cannot_read_another_s_child_facts(mem):
    mem.semantic.remember("confidential family history", subject=CHILD, owner_user_id=OWNER)
    assert mem.semantic.recall(subject=CHILD, owner_user_id="attacker") == []


# ---------------------------------------------------------------------------
# Episodic
# ---------------------------------------------------------------------------


def test_episodes_are_recalled_by_relevance_not_only_recency(mem):
    """Twelve recent turns about sleep must not bury the ulcer turn."""
    mem.observe(session_id="s1", role="user", content="she has an ulcer in her stomach",
                subject=CHILD, owner_user_id=OWNER)
    for i in range(12):
        mem.observe(session_id="s1", role="user", content=f"how much should she sleep {i}",
                    subject=CHILD, owner_user_id=OWNER)

    hits = mem.episodic.recall(session_id="s1", owner_user_id=OWNER, query="ulcer")
    assert hits, "relevance recall returned nothing"
    assert "ulcer" in hits[0].content


def test_relevant_turns_are_found_across_earlier_sessions(mem):
    """A new session starts empty; the child's thread did not."""
    mem.observe(session_id="old", role="user", content="he is allergic to peanuts",
                subject=CHILD, owner_user_id=OWNER)
    hits = mem.episodic.recall_across_sessions(
        subject=CHILD, owner_user_id=OWNER, query="what is he allergic to"
    )
    assert any("peanut" in h.content for h in hits)


def test_an_irrelevant_turn_is_not_padded_into_a_relevance_search(mem):
    mem.observe(session_id="s1", role="user", content="she has an ulcer",
                subject=CHILD, owner_user_id=OWNER)
    mem.observe(session_id="s1", role="user", content="the weather is nice",
                subject=CHILD, owner_user_id=OWNER)
    hits = mem.episodic.recall(session_id="s1", owner_user_id=OWNER, query="ulcer")
    assert all("weather" not in h.content for h in hits)


# ---------------------------------------------------------------------------
# Procedural
# ---------------------------------------------------------------------------


def test_house_rules_load_and_are_ordered_by_priority(mem):
    rules = mem.procedural.rules_for(intents={"medical"})
    assert rules, "no procedural rules loaded"
    priorities = [r.priority for r in rules]
    assert priorities == sorted(priorities, reverse=True)


def test_a_rule_scoped_to_an_intent_does_not_apply_elsewhere(mem):
    vision_ids = {r.id for r in mem.procedural.rules_for(intents={"vision"})}
    chat_ids = {r.id for r in mem.procedural.rules_for(intents={"chat"})}
    assert "describe_what_you_see" in vision_ids
    assert "describe_what_you_see" not in chat_ids


def test_a_parent_correction_overrides_the_house_rule_of_the_same_id(mem):
    mem.procedural.learn(
        "Always answer in very short sentences.",
        owner_user_id=OWNER,
        rule_id="use_memory_silently",
        priority=200,
    )
    rules = {r.id: r for r in mem.procedural.rules_for(intents=None, owner_user_id=OWNER)}
    assert rules["use_memory_silently"].text == "Always answer in very short sentences."
    assert rules["use_memory_silently"].source == "parent_correction"


def test_procedural_rendering_drops_the_least_important_rule_first(mem):
    full = mem.procedural.render(intents={"medical"})
    squeezed = mem.procedural.render(intents={"medical"}, budget_chars=400)
    assert len(squeezed) <= 400
    assert squeezed, "budget pressure silenced every rule"
    # The top rule survives whole; the tail is what gets dropped.
    assert squeezed.split("\n")[0] == full.split("\n")[0]
    assert len(squeezed.split("\n")) < len(full.split("\n"))


def test_a_budget_too_small_for_any_rule_still_says_something(mem):
    """No rules at all is worse than one truncated rule."""
    squeezed = mem.procedural.render(intents={"medical"}, budget_chars=60)
    assert squeezed.strip(), "returned nothing"
    assert len(squeezed) <= 60


# ---------------------------------------------------------------------------
# Consolidation
# ---------------------------------------------------------------------------


def test_consolidation_turns_conversation_into_durable_facts(mem, monkeypatch):
    """Without the model, statements are still captured."""
    import assistant.llm.qwen_client as qc

    monkeypatch.setattr(qc, "llm_enabled", lambda: False)
    for text in (
        "my daughter has an ulcer in her stomach",
        "what should I feed her?",
        "she is allergic to peanuts",
    ):
        mem.observe(session_id="s1", role="user", content=text,
                    subject=CHILD, owner_user_id=OWNER)

    stored = mem.maybe_consolidate(session_id="s1", subject=CHILD, owner_user_id=OWNER, force=True)
    joined = " ".join(stored).lower()
    assert "ulcer" in joined
    assert "peanut" in joined
    assert "what should i feed" not in joined, "a question was stored as a fact"


def test_consolidation_folds_each_stretch_exactly_once(mem, monkeypatch):
    import assistant.llm.qwen_client as qc

    monkeypatch.setattr(qc, "llm_enabled", lambda: False)
    mem.observe(session_id="s1", role="user", content="she has an ulcer in her stomach",
                subject=CHILD, owner_user_id=OWNER)
    first = mem.maybe_consolidate(session_id="s1", subject=CHILD, owner_user_id=OWNER, force=True)
    second = mem.maybe_consolidate(session_id="s1", subject=CHILD, owner_user_id=OWNER, force=True)
    assert first and not second, "the same stretch was folded twice"


def test_consolidation_does_not_delete_the_episodes_it_folded(mem, monkeypatch):
    import assistant.llm.qwen_client as qc

    monkeypatch.setattr(qc, "llm_enabled", lambda: False)
    mem.observe(session_id="s1", role="user", content="she has an ulcer in her stomach",
                subject=CHILD, owner_user_id=OWNER)
    mem.maybe_consolidate(session_id="s1", subject=CHILD, owner_user_id=OWNER, force=True)
    assert mem.episodic.count(session_id="s1", owner_user_id=OWNER) == 1


def test_consolidation_waits_until_enough_has_been_said(mem, monkeypatch):
    import assistant.llm.qwen_client as qc

    monkeypatch.setattr(qc, "llm_enabled", lambda: False)
    monkeypatch.setenv("NESTLING_MEMORY_CONSOLIDATE_EVERY", "8")
    reset_settings()
    mem.observe(session_id="s1", role="user", content="she has an ulcer in her stomach",
                subject=CHILD, owner_user_id=OWNER)
    assert mem.maybe_consolidate(session_id="s1", subject=CHILD, owner_user_id=OWNER) == []


def test_a_model_reply_wrapped_in_prose_still_yields_facts():
    from assistant.memory.consolidation import _parse_facts

    assert _parse_facts('Here you go: ["she has an ulcer"] hope that helps') == [
        "she has an ulcer"
    ]
    assert _parse_facts('[{"fact": "he is allergic to peanuts"}]') == [
        "he is allergic to peanuts"
    ]
    assert _parse_facts("no json here") is None


# ---------------------------------------------------------------------------
# Assembly / budget
# ---------------------------------------------------------------------------


def test_every_source_is_labelled_and_ordered(mem):
    ctx = assemble(procedural="- be brief", semantic="- has an ulcer",
                   episodic="USER: hello", working="feeding guidance")
    text = ctx.text
    for heading in (PROCEDURAL_HEADING, SEMANTIC_HEADING, EPISODIC_HEADING, WORKING_HEADING):
        assert heading in text
    # The child's own facts must precede general guidance.
    assert text.index(SEMANTIC_HEADING) < text.index(WORKING_HEADING)


def test_a_chatty_session_cannot_crowd_out_the_child_s_facts():
    """The one line recording an allergy must survive a wall of small talk."""
    ctx = assemble(
        semantic="- he is allergic to peanuts",
        episodic="\n".join(f"USER: chatter number {i}" for i in range(400)),
        total_chars=1000,
    )
    assert "allergic to peanuts" in ctx.text


def test_a_kind_with_content_always_gets_at_least_its_own_share():
    """Every kind oversized: none may be squeezed below its guaranteed share."""
    ctx = assemble(
        procedural="p" * 5000, semantic="s" * 5000,
        episodic="e" * 5000, working="w" * 5000, total_chars=2000,
    )
    for kind, used in ctx.used.items():
        assert used >= ctx.budget[kind] - 1, f"{kind} squeezed below its share"


def test_the_whole_context_respects_the_prompt_cap():
    ctx = assemble(
        procedural="p" * 5000, semantic="s" * 5000,
        episodic="e" * 5000, working="w" * 5000, total_chars=1200,
    )
    assert sum(ctx.used.values()) <= 1200


def test_an_empty_kind_hands_its_budget_to_the_others():
    with_episodes = assemble(semantic="s" * 5000, episodic="e" * 5000, total_chars=1000)
    without = assemble(semantic="s" * 5000, episodic="", total_chars=1000)
    assert without.used["semantic"] > with_episodes.used["semantic"]


def test_budget_shares_are_normalised_not_required_to_sum_to_one(monkeypatch):
    monkeypatch.setenv("NESTLING_MEMORY_SHARE_SEMANTIC", "6")
    monkeypatch.setenv("NESTLING_MEMORY_SHARE_PROCEDURAL", "2")
    monkeypatch.setenv("NESTLING_MEMORY_SHARE_EPISODIC", "1")
    monkeypatch.setenv("NESTLING_MEMORY_SHARE_WORKING", "1")
    reset_settings()
    b = budget(1000)
    assert b.total <= 1000
    assert b.semantic > b.procedural > b.episodic
    reset_settings()


# ---------------------------------------------------------------------------
# System wiring
# ---------------------------------------------------------------------------


def test_context_for_a_recall_question_carries_the_remembered_fact(mem):
    mem.semantic.remember("she has an ulcer in her stomach", subject=CHILD, owner_user_id=OWNER)
    mem.observe(session_id="s1", role="user", content="she has an ulcer in her stomach",
                subject=CHILD, owner_user_id=OWNER)
    ctx = mem.context_for(
        question="where was her ulcer?", session_id="s1", subject=CHILD,
        owner_user_id=OWNER, intents={"medical"}, working="vaccination site care",
    )
    assert "ulcer" in ctx.text
    assert SEMANTIC_HEADING in ctx.text
    assert ctx.text.index(SEMANTIC_HEADING) < ctx.text.index(WORKING_HEADING)


def test_an_unknown_backend_falls_back_to_native_instead_of_failing(monkeypatch, tmp_path):
    monkeypatch.setenv("NESTLING_MEMORY_DB", str(tmp_path / "m.db"))
    monkeypatch.setenv("NESTLING_MEMORY_BACKEND", "does-not-exist")
    reset_settings()
    from assistant.memory.system import build_backend

    assert build_backend().name == "native"
    reset_settings()


def test_forgetting_a_child_removes_facts_and_episodes(mem):
    mem.semantic.remember("she has an ulcer", subject=CHILD, owner_user_id=OWNER)
    mem.observe(session_id="s1", role="user", content="she has an ulcer",
                subject=CHILD, owner_user_id=OWNER)
    assert mem.forget(subject=CHILD, owner_user_id=OWNER) >= 2
    assert mem.semantic.recall(subject=CHILD, owner_user_id=OWNER) == []
