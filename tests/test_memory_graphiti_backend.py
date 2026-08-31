"""The Graphiti backend must never lose a fact, whatever the graph does.

graphiti-core is not installed in CI and the deployment host may not be able
to install it, so these tests exercise the contract with the graph stubbed:
the verbatim record is kept natively, the graph only reorders and enriches,
and every graph failure degrades instead of raising into a chat turn.
"""

from __future__ import annotations

import pytest

from assistant.memory.backends.graphiti_backend import GraphitiMemoryBackend
from assistant.memory.types import SEMANTIC, Episode, MemoryRecord
from assistant.settings import reset_settings

OWNER = "user-1"
CHILD = "child-1"


@pytest.fixture
def backend(tmp_path, monkeypatch):
    monkeypatch.setenv("NESTLING_MEMORY_DB", str(tmp_path / "memory.db"))
    reset_settings()
    b = GraphitiMemoryBackend({"group_prefix": "test"})
    yield b
    b.mirror.close()
    reset_settings()


def _fact(text: str) -> MemoryRecord:
    return MemoryRecord(text=text, kind=SEMANTIC, subject=CHILD, owner_user_id=OWNER)


def test_a_fact_is_kept_natively_even_when_the_graph_is_down(backend):
    """The graph is an index over the record, never the record itself."""
    backend._checked, backend._ok = True, False  # graph unavailable
    backend.remember_fact(_fact("she has an ulcer in her stomach"))
    hits = backend.recall_facts(CHILD, owner_user_id=OWNER)
    assert any("ulcer" in h.text for h in hits)


def test_a_graph_write_failure_does_not_reach_the_caller(backend, monkeypatch):
    backend._checked, backend._ok = True, True
    monkeypatch.setattr(
        backend, "_run", lambda coro: (_ for _ in ()).throw(RuntimeError("graph exploded"))
    )
    # Must not raise: a chat turn cannot fail because extraction did.
    backend.remember_fact(_fact("he is allergic to peanuts"))
    backend.add_episode(
        Episode(role="user", content="he is allergic to peanuts",
                session_id="s1", subject=CHILD, owner_user_id=OWNER)
    )
    assert any("peanut" in h.text for h in backend.recall_facts(CHILD, owner_user_id=OWNER))
    assert backend.count_episodes(session_id="s1", owner_user_id=OWNER) == 1


def test_a_graph_search_failure_falls_back_to_native_ranking(backend, monkeypatch):
    backend._checked, backend._ok = True, True
    backend.remember_fact(_fact("she has an ulcer in her stomach"))
    monkeypatch.setattr(
        backend, "_run", lambda coro: (_ for _ in ()).throw(RuntimeError("search down"))
    )
    hits = backend.recall_facts(CHILD, owner_user_id=OWNER, query="ulcer")
    assert any("ulcer" in h.text for h in hits)


def test_graph_relevance_reorders_the_verbatim_records(backend, monkeypatch):
    """The graph decides order; the parent still reads their own words back."""
    backend._checked, backend._ok = True, True
    backend.remember_fact(_fact("he sleeps through the night"))
    backend.remember_fact(_fact("we go to Mehr hospital"))

    class _Edge:
        def __init__(self, fact):
            self.fact = fact

    monkeypatch.setattr(
        backend, "_run", lambda coro: [_Edge("the child attends we go to Mehr hospital")]
    )
    hits = backend.recall_facts(CHILD, owner_user_id=OWNER, query="which hospital")
    assert hits[0].text == "we go to Mehr hospital"


def test_a_graph_paraphrase_with_no_verbatim_record_is_not_surfaced(backend, monkeypatch):
    """An extracted edge is the graph's wording, not the parent's."""
    backend._checked, backend._ok = True, True
    backend.remember_fact(_fact("she has an ulcer in her stomach"))

    class _Edge:
        fact = "PATIENT_HAS_CONDITION(child, gastric_ulcer)"

    monkeypatch.setattr(backend, "_run", lambda coro: [_Edge()])
    hits = backend.recall_facts(CHILD, owner_user_id=OWNER, query="ulcer")
    assert all("PATIENT_HAS_CONDITION" not in h.text for h in hits)
    assert any("ulcer in her stomach" in h.text for h in hits)


def test_accounts_get_separate_graph_partitions(backend):
    """Scoping must be structural, not a filter applied after searching."""
    mine = backend._group(CHILD, OWNER)
    theirs = backend._group(CHILD, "attacker")
    assert mine != theirs
    assert OWNER in mine


def test_an_unavailable_graph_is_reported_not_raised(backend, monkeypatch):
    monkeypatch.setattr(
        backend, "_client", lambda: (_ for _ in ()).throw(RuntimeError("no graph db"))
    )
    backend._checked = False
    assert backend.available() is False


def test_forgetting_removes_the_native_record(backend):
    backend._checked, backend._ok = True, False
    backend.remember_fact(_fact("confidential"))
    assert backend.forget(subject=CHILD, owner_user_id=OWNER) >= 1
    assert backend.recall_facts(CHILD, owner_user_id=OWNER) == []


def test_the_backend_satisfies_the_shared_interface():
    from assistant.memory.backends.base import MemoryBackend

    assert isinstance(GraphitiMemoryBackend({}), MemoryBackend)
