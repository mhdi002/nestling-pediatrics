"""The memory system as the agent actually uses it, through chat().

The layers passing their own tests proves nothing if the orchestrator never
calls them, so these drive real chat turns and assert on what reaches the
model.
"""

from __future__ import annotations

import pytest

from assistant.settings import reset_settings


@pytest.fixture
def agent(monkeypatch, tmp_path):
    """A ParentAssistant with a recording stub in place of the model."""
    monkeypatch.setenv("NESTLING_CHILD_DB", str(tmp_path / "child.db"))
    monkeypatch.setenv("NESTLING_CHAT_DB", str(tmp_path / "chat.db"))
    monkeypatch.setenv("NESTLING_MEMORY_DB", str(tmp_path / "memory.db"))
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



def _grounded(calls):
    """The call that built the parent's answer, not the intent router.

    The router also goes through answer_with_context, so picking the last
    call at random tested the wrong prompt.
    """
    for call in reversed(calls):
        if "WHAT THIS PARENT HAS TOLD YOU" in (call.get("context") or ""):
            return call
    return calls[-1] if calls else {}


def _child(orch, name="Kian"):
    return orch.db.create_child(name=name, sex="male", gestational_age_weeks=39.0)


def test_turns_are_written_to_episodic_memory(agent):
    cid = _child(agent)
    sid = agent.chat_memory.create_session(child_id=cid)
    agent.chat(sid, "she has an ulcer in her stomach", child_id=cid)

    assert agent.memory is not None
    # Both halves of the exchange, so a later question can find either.
    assert agent.memory.episodic.count(session_id=sid) >= 2


def test_a_later_session_recalls_what_an_earlier_one_was_told(agent, monkeypatch):
    """The failure this system exists to fix, driven through chat().

    The pre-existing child-note digest is switched off, so the recall can only
    have come from the new episodic/semantic layers -- otherwise this test
    would pass without them.
    """
    monkeypatch.setenv("NESTLING_CHILD_MEMORY_ENABLED", "0")
    reset_settings()
    cid = _child(agent, "Bita")
    told = agent.chat_memory.create_session(child_id=cid)
    agent.chat(told, "she has an ulcer in her stomach", child_id=cid)

    agent._calls.clear()
    asked = agent.chat_memory.create_session(child_id=cid)
    agent.chat(asked, "where was her ulcer?", child_id=cid)

    assert agent._calls, "the model was never asked"
    context = _grounded(agent._calls).get("context") or ""
    assert "ulcer" in context.lower(), "the earlier session did not reach the prompt"


def test_relevance_beats_recency_across_a_long_session(agent, monkeypatch):
    """A dozen turns about sleep must not bury the one line about an ulcer."""
    monkeypatch.setenv("NESTLING_CHILD_MEMORY_ENABLED", "0")
    reset_settings()
    cid = _child(agent, "Sara")
    sid = agent.chat_memory.create_session(child_id=cid)
    agent.chat(sid, "she has an ulcer in her stomach", child_id=cid)
    for i in range(12):
        agent.chat(sid, f"how much should she sleep at {i} months?", child_id=cid)

    agent._calls.clear()
    # Phrased so it routes to the medical path. "remind me ..." is currently
    # classified as a request for the child's record instead -- see
    # test_a_recall_phrasing_is_misrouted_to_the_record below.
    agent.chat(sid, "where was her ulcer?", child_id=cid)
    context = (_grounded(agent._calls).get("context") or "").lower()
    assert "ulcer" in context, "recency buried the relevant turn"
    assert "sleep" not in context.split("ulcer")[0][-200:], "buried under sleep turns"


def test_a_recall_phrasing_now_reaches_memory(agent):
    """This test previously documented a bug; the bug is fixed.

    HISTORY_RE matches a bare "remind me", so "remind me where her ulcer was"
    used to be classified as a request for the child's record and never
    reached the memory-grounded path. Routing now asks what the message names
    rather than which verb it used.
    """
    from assistant.agent.intents import classify_intent

    assert classify_intent("where was her ulcer?") == {"medical"}
    assert "history" not in classify_intent("remind me where her ulcer was")

    cid = _child(agent, "Nika")
    sid = agent.chat_memory.create_session(child_id=cid)
    agent.chat(sid, "she has an ulcer in her stomach", child_id=cid)
    recalled = agent._recall_episodic("remind me where her ulcer was", sid, cid, None)
    assert "ulcer" in recalled.lower()


def test_procedural_rules_reach_the_system_prompt(agent):
    cid = _child(agent)
    sid = agent.chat_memory.create_session(child_id=cid)
    agent.chat(sid, "what should I feed a two month old?", child_id=cid)

    system = _grounded(agent._calls).get("system") or ""
    assert "How to answer:" in system
    # The rule added because the assistant kept narrating its own notes.
    assert "Never mention notes" in system or "without talking about it" in system


def test_consolidation_runs_and_produces_durable_facts(agent, monkeypatch):
    """After enough turns, the conversation becomes facts about the child."""
    monkeypatch.setenv("NESTLING_MEMORY_CONSOLIDATE_EVERY", "2")
    reset_settings()
    cid = _child(agent)
    sid = agent.chat_memory.create_session(child_id=cid)
    agent.chat(sid, "my son has bad eczema on his left elbow", child_id=cid)
    agent.chat(sid, "he is also allergic to peanuts", child_id=cid)

    facts = " ".join(
        r.text for r in agent.memory.semantic.recall(subject=cid, limit=50)
    ).lower()
    assert "eczema" in facts or "peanut" in facts, f"nothing consolidated: {facts!r}"


def test_a_broken_memory_system_does_not_break_the_chat(agent, monkeypatch):
    """Memory degrades; the conversation continues."""

    class _Broken:
        def __getattr__(self, name):
            raise RuntimeError("memory is down")

    monkeypatch.setattr(type(agent), "memory", property(lambda self: _Broken()))
    cid = _child(agent)
    sid = agent.chat_memory.create_session(child_id=cid)
    out = agent.chat(sid, "what should I feed a two month old?", child_id=cid)
    assert (out.get("reply") or "").strip(), "a memory failure silenced the reply"


def test_memory_is_scoped_to_the_account(agent):
    """One account's remembered facts must not reach another's prompt."""
    cid = _child(agent)
    sid = agent.chat_memory.create_session(child_id=cid)
    agent.chat(sid, "she has an ulcer in her stomach", child_id=cid,
               owner_user_id="owner")

    assert not agent._recall_semantic("ulcer", cid, "attacker")


def test_the_profile_graph_joins_facts_from_different_turns(agent, monkeypatch):
    """"which hospital treated her ulcer?" needs two facts joined.

    Neither sentence contains both words, so only a graph walk connects the
    condition to the clinic.
    """
    monkeypatch.setenv("NESTLING_CHILD_MEMORY_ENABLED", "0")
    reset_settings()
    cid = _child(agent, "Roya")
    sid = agent.chat_memory.create_session(child_id=cid)

    # Store as durable facts the way consolidation would, without the model.
    agent.memory.semantic.remember("she has an ulcer in her stomach",
                                   subject=cid, use_llm=False)
    agent.memory.semantic.remember("we saw a doctor at Mehr hospital",
                                   subject=cid, use_llm=False)

    recalled = agent._recall_semantic("which hospital treated her ulcer?", cid, None)
    assert "Mehr hospital" in recalled, recalled


def test_the_child_s_history_stays_reachable_after_a_session_has_turns(agent, monkeypatch):
    """Found by real conversation: it forgot mid-chat.

    Cross-session recall used to run only when the current session produced
    nothing, so a new session could reach the child's history for its FIRST
    question and never again -- by the second, the session had turns of its
    own. In conversation that read as the assistant recalling an allergy and
    then denying it knew the clinic and the diagnosis from minutes earlier.
    """
    monkeypatch.setenv("NESTLING_CHILD_MEMORY_ENABLED", "0")
    reset_settings()
    cid = _child(agent, "Darya")

    told = agent.chat_memory.create_session(child_id=cid)
    agent.chat(told, "she has bronchiolitis on her chest", child_id=cid)
    agent.chat(told, "we took her to Razi clinic for it", child_id=cid)
    agent.chat(told, "she is allergic to sesame", child_id=cid)

    asked = agent.chat_memory.create_session(child_id=cid)
    # First question: the session is empty, so this always worked.
    first = agent._recall_episodic("what is she allergic to?", asked, cid, None)
    assert "sesame" in first.lower()

    agent.chat(asked, "what is she allergic to?", child_id=cid)

    # Second and third: the session now has turns. These used to come back
    # with nothing from the child's earlier sessions at all.
    clinic = agent._recall_episodic("which clinic did we take her to?", asked, cid, None)
    assert "razi" in clinic.lower(), clinic
    condition = agent._recall_episodic("what was her chest problem?", asked, cid, None)
    assert "bronchiolitis" in condition.lower(), condition
