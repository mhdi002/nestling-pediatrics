"""Only the parent's words may become a durable fact about their child.

The deterministic extractor has always skipped assistant turns; the model
path did not, and fed it the whole transcript. That is how an invention
becomes a record: told "she has cradle cap on her tummy", a small model
answered about a tummy ache, the reply was consolidated, and from then on the
child's file said she had one. Both paths now read the same turns.
"""

from __future__ import annotations

import pytest

from assistant.memory.backends.native import NativeMemoryBackend
from assistant.memory.consolidation import extract_with_llm
from assistant.memory.system import MemorySystem
from assistant.memory.types import Episode
from assistant.settings import reset_settings
from tests import scenarios

OWNER = "owner-1"
CHILD = "child-1"


@pytest.fixture
def mem(tmp_path, monkeypatch):
    monkeypatch.setenv("NESTLING_MEMORY_DB", str(tmp_path / "memory.db"))
    reset_settings()
    system = MemorySystem(NativeMemoryBackend(tmp_path / "memory.db"))
    yield system
    system.close()
    reset_settings()


ALL = scenarios.many(10, start=400)


@pytest.mark.parametrize("sc", ALL, ids=lambda s: f"seed{s.seed}")
def test_the_extractor_is_never_shown_the_assistant_s_own_prose(monkeypatch, sc):
    """Whatever the reply said, the extractor must not see it."""
    import assistant.llm.qwen_client as qc

    seen = {}

    class Fake:
        def answer_with_context(self, query, context, *, system=None):
            seen["context"] = context
            return "[]"

    monkeypatch.setattr(qc, "llm_enabled", lambda: True)
    monkeypatch.setattr(qc, "get_qwen", lambda: Fake())

    invention = f"it sounds like {sc.pronoun} has a completely different problem"
    episodes = [
        Episode(id="1", session_id="s", role="user", content=sc.condition_fact),
        Episode(id="2", session_id="s", role="assistant", content=invention),
        Episode(id="3", session_id="s", role="user", content=sc.allergy_fact),
    ]
    extract_with_llm(episodes)
    assert invention not in seen["context"], f"{sc}\n{seen['context']}"
    assert sc.condition_fact in seen["context"], f"{sc}\n{seen['context']}"
    assert sc.allergy_fact in seen["context"], f"{sc}\n{seen['context']}"


@pytest.mark.parametrize("sc", ALL, ids=lambda s: f"seed{s.seed}")
def test_a_reply_the_parent_never_confirmed_does_not_become_a_fact(mem, sc, monkeypatch):
    """End to end, through the store, with the model unavailable."""
    import assistant.llm.qwen_client as qc

    monkeypatch.setattr(qc, "llm_enabled", lambda: False)
    invention = f"{sc.pronoun} clearly has something else entirely"
    mem.observe(session_id="s1", role="user", content=sc.condition_fact,
                subject=CHILD, owner_user_id=OWNER)
    mem.observe(session_id="s1", role="assistant", content=invention,
                subject=CHILD, owner_user_id=OWNER)
    mem.maybe_consolidate(session_id="s1", subject=CHILD,
                          owner_user_id=OWNER, force=True)

    facts = " ".join(
        r.text for r in mem.semantic.recall(subject=CHILD, owner_user_id=OWNER, limit=50)
    ).lower()
    assert "something else entirely" not in facts, facts
    assert sc.condition.lower() in facts, facts


def test_an_all_assistant_stretch_consolidates_to_nothing(monkeypatch):
    import assistant.llm.qwen_client as qc

    monkeypatch.setattr(qc, "llm_enabled", lambda: True)
    monkeypatch.setattr(qc, "get_qwen", lambda: pytest.fail("should not be asked"))
    episodes = [Episode(id="1", session_id="s", role="assistant", content="anything")]
    assert extract_with_llm(episodes) == []
