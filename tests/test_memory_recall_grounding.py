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


# --------------------------------------------------------------------------
# End-to-end: the parent must actually receive the remembered answer
# --------------------------------------------------------------------------


class _FaithfulModel:
    """A model that answers only from what the context actually contains."""

    vision_ready = False
    ready = True

    def __init__(self, reply=None):
        self.calls = []
        self._reply = reply

    def answer_with_context(self, query, context, *, system=None):
        self.calls.append({"query": query, "context": context, "system": system})
        if self._reply is not None:
            return self._reply
        low = (context or "").lower()
        if "ulcer" in low and "ulcer" in (query or "").lower():
            return "You told me earlier that she has an ulcer in her stomach."
        return "I could not find that in what you have told me."


def _assistant(monkeypatch, tmp_path, model):
    import assistant.llm.qwen_client as qc
    from assistant.agent.orchestrator import ParentAssistant
    from assistant.settings import reset_settings

    monkeypatch.setenv("NESTLING_CHILD_DB", str(tmp_path / "child.db"))
    monkeypatch.setenv("NESTLING_CHAT_DB", str(tmp_path / "chat.db"))
    reset_settings()
    monkeypatch.setattr(qc, "llm_enabled", lambda: True)
    monkeypatch.setattr(qc, "get_qwen", lambda: model)
    orch = ParentAssistant()
    if not orch.medical.load():
        import pytest

        pytest.skip("medical index not built")
    orch.use_llm = True
    return orch


def test_a_later_session_can_answer_what_an_earlier_one_was_told(monkeypatch, tmp_path):
    """The original bug: 'where was her ulcer?' answered as vaccination guidance."""
    model = _FaithfulModel()
    orch = _assistant(monkeypatch, tmp_path, model)
    cid = orch.db.create_child(name="monika", sex="female", gestational_age_weeks=32.0)

    told = orch.chat_memory.create_session(child_id=cid)
    orch.chat(told, "she has some ulcer in her stomach", child_id=cid)

    model.calls.clear()
    asked = orch.chat_memory.create_session(child_id=cid)
    reply = orch.chat(asked, "where was her ulcer?", child_id=cid).get("reply") or ""

    assert model.calls, "the model was never asked"
    context = model.calls[-1]["context"]
    # The parent's own words, not the memory blob, must be the question.
    assert model.calls[-1]["query"] == "where was her ulcer?"
    # The remembered fact must reach the slot the model answers from, and must
    # come before the retrieved guidance.
    assert "ulcer" in context.lower()
    assert context.index(PARENT_NOTES_HEADING) < context.index(CARE_NOTES_HEADING)
    assert "stomach" in reply.lower()


def test_a_generated_answer_is_not_truncated_as_if_it_were_extractive(
    monkeypatch, tmp_path
):
    """Retrieval runs with the LLM off, so `mode` must be corrected afterwards.

    format_reply() trims anything not marked as model-generated to three
    sentences; leaving `mode` at "extractive" silently cut the reply.
    """
    from assistant.parent_voice import medical_chat_answer

    long_answer = " ".join(f"Sentence number {i} about her ulcer." for i in range(1, 7))
    orch = _assistant(monkeypatch, tmp_path, _FaithfulModel(reply=long_answer))
    res = orch.ask_medical(_query("where was her ulcer?"))

    mode = (res.get("mode") or "").lower()
    from_llm = "openai" in mode or "llm" in mode
    assert from_llm, f"generated answer mislabelled as {mode!r}"
    spoken = medical_chat_answer(res["answer"], from_llm=from_llm)
    assert spoken.count("Sentence number") == 6


def test_gestational_age_is_stated_as_the_child_s_birth(monkeypatch, tmp_path):
    """"GA 32.0w" was read as the mother being 32 weeks pregnant.

    The model then answered a question about the baby's stomach with advice on
    managing ulcers during pregnancy, so the digest must say what the number
    means rather than abbreviate it.
    """
    from assistant.memory.child_db import ChildMemoryDB
    from assistant.settings import reset_settings

    monkeypatch.setenv("NESTLING_CHILD_DB", str(tmp_path / "child.db"))
    reset_settings()
    db = ChildMemoryDB()
    cid = db.create_child(name="Monika", sex="female", gestational_age_weeks=32.0)
    digest = db.child_context_text(cid)

    assert "born at 32 weeks gestation" in digest
    assert "GA 32" not in digest, "the ambiguous abbreviation is back"
