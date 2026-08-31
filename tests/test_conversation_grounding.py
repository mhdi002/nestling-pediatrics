"""Would a model that reads the prompt answer correctly?

Every other conversation test stubs the model with something that ignores its
input, so they prove the right facts are RETRIEVED. They cannot catch a prompt
that contains the answer but is arranged so a careful reader would still get
it wrong -- the ulcer bug was exactly that shape: the fact was present, but
sat beside guidance that looked equally authoritative.

So the stub here answers only from what it is handed, by looking for the
question's terms in the labelled sections and reporting which section it drew
from. It is not a language model and does not pretend to be. It is a strict
reader, and a prompt a strict reader cannot answer from is a prompt worth
fixing before a 4B model ever sees it.
"""

from __future__ import annotations

import re

import pytest

from assistant.agent.grounding import CARE_NOTES_HEADING, PARENT_NOTES_HEADING
from assistant.settings import reset_settings
from tests import scenarios

ALL = scenarios.many(10, start=200)


def _sections(context: str) -> dict[str, str]:
    """Split a labelled context back into its parts."""
    out: dict[str, str] = {}
    current = None
    for line in (context or "").splitlines():
        heading = re.match(r"^\[([^\]]+)\]$", line.strip())
        if heading:
            current = heading.group(1)
            out.setdefault(current, "")
        elif current:
            out[current] += line + "\n"
    return out


class FaithfulReader:
    """Answers strictly from the context, and says where the answer came from."""

    vision_ready = True
    ready = True

    def __init__(self):
        self.calls: list[dict] = []

    def answer_with_context(self, query, context, *, system=None):
        self.calls.append({"query": query, "context": context, "system": system})
        sections = _sections(context or "")
        terms = [t for t in re.findall(r"[\w؀-ۿ]+", (query or "").lower())
                 if len(t) > 3]
        for heading, body in sections.items():
            low = body.lower()
            hits = [t for t in terms if t in low]
            if hits:
                # Quote the line that matched, the way a grounded reply would.
                for line in body.splitlines():
                    if any(t in line.lower() for t in hits):
                        return f"[from {heading}] {line.strip()}"
        return "[from nothing] I do not have that."

    def chat(self, *a, **k):
        return "ok"


@pytest.fixture
def agent(monkeypatch, tmp_path):
    monkeypatch.setenv("NESTLING_CHILD_DB", str(tmp_path / "child.db"))
    monkeypatch.setenv("NESTLING_CHAT_DB", str(tmp_path / "chat.db"))
    monkeypatch.setenv("NESTLING_MEMORY_DB", str(tmp_path / "memory.db"))
    monkeypatch.setenv("NESTLING_CHILD_MEMORY_ENABLED", "0")
    reset_settings()

    import assistant.llm.qwen_client as qc

    model = FaithfulReader()
    monkeypatch.setattr(qc, "llm_enabled", lambda: True)
    monkeypatch.setattr(qc, "get_qwen", lambda: model)

    from assistant.agent.orchestrator import ParentAssistant

    orch = ParentAssistant()
    if not orch.medical.load():
        pytest.skip("medical index not built")
    orch.use_llm = True
    orch._model = model
    yield orch
    reset_settings()


@pytest.mark.parametrize("sc", ALL, ids=lambda s: f"seed{s.seed}")
def test_a_recall_question_is_answerable_from_the_parent_s_own_notes(agent, sc):
    """The answer must be reachable in the section about this child.

    Not merely present somewhere in the prompt: present where a reader looking
    for facts about this child would find it.
    """
    cid = agent.db.create_child(name=sc.name, sex="female", gestational_age_weeks=39.0)
    sid = agent.chat_memory.create_session(child_id=cid)
    agent.chat(sid, sc.allergy_fact, child_id=cid)

    agent._model.calls.clear()
    out = agent.chat(sid, sc.allergy_question, child_id=cid)
    reply = out.get("reply") or ""

    assert sc.allergen.lower() in reply.lower(), f"{sc}\nreply: {reply[:300]!r}"
    assert PARENT_NOTES_HEADING in reply, (
        f"{sc}\nanswered from the wrong section: {reply[:200]!r}"
    )


@pytest.mark.parametrize("sc", ALL, ids=lambda s: f"seed{s.seed}")
def test_a_guidance_question_still_gets_the_care_notes(agent, sc):
    """The counterpart: a general question must arrive with guidance attached.

    This asserts the prompt is adequate, not which section a reader picks.
    The stub returns the first section whose text matches, and the transcript
    can match first simply by containing an earlier reply on the topic -- a
    real model has the whole prompt and a system instruction telling it which
    source is for guidance. Which one it actually uses is a question for a
    live model; whether the guidance was there to use is answerable here.
    """
    cid = agent.db.create_child(name=sc.name, sex="male", gestational_age_weeks=39.0)
    sid = agent.chat_memory.create_session(child_id=cid)
    agent.chat(sid, sc.condition_fact, child_id=cid)

    agent._model.calls.clear()
    agent.chat(sid, "how often should I feed a two month old?", child_id=cid)
    grounded = [c for c in agent._model.calls
                if PARENT_NOTES_HEADING in (c["context"] or "")]
    assert grounded, f"{sc}\nno grounded call"
    context = grounded[-1]["context"]
    assert CARE_NOTES_HEADING in context, f"{sc}\nno guidance in the prompt"
    section = context.split(CARE_NOTES_HEADING, 1)[1]
    assert any(word in section.lower() for word in ("feed", "milk", "breast")), (
        f"{sc}\nguidance was present but off-topic: {section[:200]!r}"
    )


@pytest.mark.parametrize("sc", ALL[:6], ids=lambda s: f"seed{s.seed}")
def test_the_child_s_own_facts_are_never_below_general_guidance(agent, sc):
    """Ordering is what stopped the ulcer being answered as a vaccine site."""
    cid = agent.db.create_child(name=sc.name, sex="female", gestational_age_weeks=39.0)
    sid = agent.chat_memory.create_session(child_id=cid)
    agent.chat(sid, sc.condition_fact, child_id=cid)

    agent._model.calls.clear()
    agent.chat(sid, sc.where_question, child_id=cid)
    grounded = [c for c in agent._model.calls
                if PARENT_NOTES_HEADING in (c["context"] or "")]
    assert grounded, f"{sc}\nno grounded call"
    context = grounded[-1]["context"]
    if CARE_NOTES_HEADING in context:
        assert context.index(PARENT_NOTES_HEADING) < context.index(CARE_NOTES_HEADING), sc


@pytest.mark.parametrize("sc", ALL[:6], ids=lambda s: f"seed{s.seed}")
def test_a_question_about_something_never_mentioned_is_not_invented(agent, sc):
    """A strict reader must find nothing, rather than something adjacent."""
    cid = agent.db.create_child(name=sc.name, sex="male", gestational_age_weeks=39.0)
    sid = agent.chat_memory.create_session(child_id=cid)
    agent.chat(sid, sc.allergy_fact, child_id=cid)

    agent._model.calls.clear()
    out = agent.chat(sid, "which orthodontist did we register with?", child_id=cid)
    reply = (out.get("reply") or "").lower()
    assert "orthodontist" not in reply or "do not have" in reply, (
        f"{sc}\ninvented an answer: {reply[:250]!r}"
    )


def test_the_reader_stub_actually_fails_when_the_fact_is_absent(agent):
    """A test double that always succeeds proves nothing -- check it can fail."""
    model = agent._model
    answer = model.answer_with_context(
        "what is she allergic to?",
        f"[{PARENT_NOTES_HEADING}]\n- she sleeps well\n",
    )
    assert "do not have" in answer, answer
