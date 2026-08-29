"""A recall question must reach the model with the parent's notes in context.

Regression for "where was her ulcer?" answered as vaccination-site guidance:
the ulcer was in the prompt, but as part of the *question*, while the retrieved
WHO chunks held the context slot the model was told to answer from.
"""

from __future__ import annotations

from assistant.agent.grounding import (
    CARE_NOTES_HEADING,
    GROUNDED_SYSTEM,
    PARENT_NOTES_HEADING,
    compose_context,
    memory_context,
    parent_question,
)

MEMORY = (
    "[CHILD_MEMORY]\n"
    "Profile: monika, female, GA 32.0w, currently ~2.0 months old\n"
    "Known chronological age: 2.0 months\n"
    "Earlier: she has some ulcer in her stomach, seen at Mehr hospital"
)


def _query(question: str) -> str:
    return f"{MEMORY}\n[CURRENT_USER]\n{question}"


def test_memory_and_question_are_separated():
    q = _query("where was her ulcer?")
    assert "ulcer in her stomach" in memory_context(q)
    assert "where was" not in memory_context(q).lower()
    assert parent_question(q) == "where was her ulcer?"


def test_no_memory_block_yields_empty_context():
    assert memory_context("just a plain question") == ""


def test_both_sources_are_labelled_in_context():
    ctx = compose_context(memory_context(_query("x")), "Vaccination site care: keep clean.")
    assert PARENT_NOTES_HEADING in ctx
    assert CARE_NOTES_HEADING in ctx
    assert ctx.index(PARENT_NOTES_HEADING) < ctx.index(CARE_NOTES_HEADING)
    assert "ulcer in her stomach" in ctx
    # Injected "Known ..." metadata lines are the orchestrator's, not the parent's.
    assert "Known chronological age" not in ctx


def test_context_omits_a_missing_source():
    assert CARE_NOTES_HEADING not in compose_context("she has an ulcer", "")
    assert PARENT_NOTES_HEADING not in compose_context("", "Vaccination site care.")


def test_system_prompt_routes_child_questions_to_the_parents_notes():
    assert PARENT_NOTES_HEADING in GROUNDED_SYSTEM
    assert CARE_NOTES_HEADING in GROUNDED_SYSTEM
    # It must admit ignorance rather than answer from the wrong source.
    assert "not mentioned it yet" in GROUNDED_SYSTEM


def test_ask_medical_sends_both_sources_to_the_model(monkeypatch):
    """The model must receive the parent's notes as context, not as the question."""
    import assistant.llm.qwen_client as qc
    from assistant.agent.orchestrator import ParentAssistant

    seen = {}

    class _Client:
        def answer_with_context(self, query, context, *, system=None):
            seen["query"] = query
            seen["context"] = context
            seen["system"] = system
            return "Your notes say the ulcer is in her stomach."

    monkeypatch.setattr(qc, "llm_enabled", lambda: True)
    monkeypatch.setattr(qc, "get_qwen", lambda: _Client())

    orch = ParentAssistant()
    if not orch.medical.load():
        import pytest

        pytest.skip("medical index not built")
    orch.use_llm = True
    res = orch.ask_medical(_query("where was her ulcer?"))

    assert seen["query"] == "where was her ulcer?"
    assert "ulcer in her stomach" in seen["context"]
    assert PARENT_NOTES_HEADING in seen["context"]
    assert PARENT_NOTES_HEADING in seen["system"]
    assert res["answer_source"] == "memory+notes"
    assert "stomach" in res["answer"]
