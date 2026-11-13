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
    # It must not narrate where anything came from: that produced "the care
    # notes you shared focus on newborns" and a re-request for an age already
    # given. Asserted as a property of the prompt rather than one phrasing of
    # it, because the phrasing has had to change -- naming the words it must
    # not say is exactly how a small model learned to say them.
    assert "Do not describe where" in GROUNDED_SYSTEM
    assert "never restate or re-answer an" in GROUNDED_SYSTEM
    assert "already given" in GROUNDED_SYSTEM


def test_the_prompt_never_supplies_the_vocabulary_it_forbids():
    """A prompt that says "never mention care notes" teaches "care notes".

    Measured: a 1B model opened replies with "General Care Notes:" and "I
    cannot access your care notes or health records", both phrasings lifted
    from the instruction telling it not to. The words are gone from the
    prompt and from the section labels, so there is nothing to lift.
    """
    from assistant.llm.qwen_client import QwenClient

    seen = {}

    def capture(self, messages, **kw):
        seen["system"] = messages[0]["content"]
        seen["user"] = messages[1]["content"]
        return "ok"

    original = QwenClient.chat
    QwenClient.chat = capture
    try:
        QwenClient().answer_with_context("how much sleep?", "Sleep guidance.")
    finally:
        QwenClient.chat = original

    forbidden = ("care note", "my records", "your records", "health record")
    for prompt in (GROUNDED_SYSTEM, seen["system"], seen["user"]):
        low = prompt.lower()
        for word in forbidden:
            assert word not in low, f"{word!r} is in the prompt: {prompt}"


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


def test_a_question_about_the_conversation_is_not_answered_with_a_checklist():
    """Asked "what is the photo", it recited a clinician photo-prep checklist.

    The corpus matched a page about preparing photographs for a clinician and
    the model asked when the spots started and whether they blanch -- having
    just described them itself.
    """
    assert "refers to something from earlier in this conversation" in GROUNDED_SYSTEM
    assert "never ask them for details it already contains" in GROUNDED_SYSTEM
