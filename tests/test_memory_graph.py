"""The child's profile as a graph.

The question that motivates all of this is "which hospital treated her
ulcer?". Flat facts answer it only if both words land in one sentence; a
graph answers it by walking child -> has_condition -> ulcer and
child -> seen_at -> Mehr hospital.
"""

from __future__ import annotations

import pytest

from assistant.memory.backends.native import NativeMemoryBackend
from assistant.memory.extraction import extract_deterministic, ingest
from assistant.memory.graph import ProfileGraph
from assistant.memory.semantic import SemanticMemory
from assistant.settings import reset_settings

OWNER = "user-1"
CHILD = "child-1"


@pytest.fixture
def graph(tmp_path):
    g = ProfileGraph(tmp_path / "graph.db")
    yield g
    g.close()


@pytest.fixture
def semantic(tmp_path, monkeypatch):
    monkeypatch.setenv("NESTLING_MEMORY_DB", str(tmp_path / "memory.db"))
    reset_settings()
    backend = NativeMemoryBackend(tmp_path / "memory.db")
    mem = SemanticMemory(backend)
    yield mem
    backend.close()
    reset_settings()


# ---------------------------------------------------------------------------
# Extraction
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sentence,relation,entity",
    [
        ("she has an ulcer in her stomach", "has_condition", "ulcer"),
        ("he is allergic to peanuts", "allergic_to", "peanuts"),
        ("we saw a doctor at Mehr hospital", "seen_at", "Mehr hospital"),
        ("she is taking amoxicillin", "takes", "amoxicillin"),
    ],
)
def test_a_sentence_becomes_a_triple_without_the_model(sentence, relation, entity):
    """The fallback matters: a 4B model's structured output is unreliable."""
    triples = extract_deterministic(sentence)
    assert triples, f"nothing extracted from {sentence!r}"
    assert any(
        t["relation"] == relation and entity.lower() in t["dst"].lower() for t in triples
    ), triples


def test_a_clinic_name_is_not_swallowed_by_the_words_before_it():
    """Matched case-insensitively this captured "a doctor at Mehr hospital"."""
    triples = extract_deterministic("we saw a doctor at Mehr hospital")
    clinic = next(t for t in triples if t["relation"] == "seen_at")
    assert clinic["dst"] == "Mehr hospital"


def test_an_ordinary_sentence_produces_no_triples():
    assert extract_deterministic("I saw her smile today") == []
    assert extract_deterministic("thank you so much") == []


# ---------------------------------------------------------------------------
# The graph
# ---------------------------------------------------------------------------


def test_the_same_entity_is_one_node_however_it_is_capitalised(graph):
    a = graph.upsert_node(subject=CHILD, label="Mehr hospital", type="clinic")
    b = graph.upsert_node(subject=CHILD, label="Mehr Hospital", type="clinic")
    assert a == b, "one clinic became two nodes"


def test_walking_from_a_condition_reaches_the_clinic(graph):
    """The question this whole design exists for."""
    for sentence in ("she has an ulcer in her stomach", "we saw a doctor at Mehr hospital"):
        ingest(graph, sentence, subject=CHILD, owner_user_id=OWNER, use_llm=False)

    rendered = graph.render(subject=CHILD, text="which hospital treated her ulcer?",
                            owner_user_id=OWNER)
    assert "Mehr hospital" in rendered, rendered


def test_a_question_about_nothing_known_walks_nowhere(graph):
    ingest(graph, "she has an ulcer in her stomach", subject=CHILD,
           owner_user_id=OWNER, use_llm=False)
    assert graph.render(subject=CHILD, text="what is the weather", owner_user_id=OWNER) == ""


def test_a_retracted_edge_stops_being_current_but_stays_walkable(graph):
    """Changing clinic does not un-attend the first one."""
    src = graph.upsert_node(subject=CHILD, label=CHILD, type="child")
    dst = graph.upsert_node(subject=CHILD, label="Razi clinic", type="clinic")
    edge = graph.add_edge(subject=CHILD, src=src, relation="seen_at", dst=dst)

    assert graph.retract_edge(edge)
    assert graph.edges(subject=CHILD) == []
    assert graph.edges(subject=CHILD, include_retracted=True), "history was destroyed"


def test_one_account_cannot_walk_another_s_graph(graph):
    ingest(graph, "she has an ulcer in her stomach", subject=CHILD,
           owner_user_id=OWNER, use_llm=False)
    assert graph.render(subject=CHILD, text="ulcer", owner_user_id="attacker") == ""


def test_two_children_with_the_same_name_stay_separate(graph):
    ingest(graph, "she has an ulcer in her stomach", subject="child-a",
           owner_user_id=OWNER, use_llm=False)
    assert graph.nodes(subject="child-b", owner_user_id=OWNER) == []


def test_forgetting_a_child_empties_their_graph(graph):
    ingest(graph, "he is allergic to peanuts", subject=CHILD,
           owner_user_id=OWNER, use_llm=False)
    assert graph.forget(subject=CHILD, owner_user_id=OWNER) >= 2
    assert graph.nodes(subject=CHILD, owner_user_id=OWNER) == []


# ---------------------------------------------------------------------------
# Wired into semantic memory
# ---------------------------------------------------------------------------


def test_remembering_a_fact_also_grows_the_graph(semantic):
    semantic.remember("he is allergic to peanuts", subject=CHILD,
                      owner_user_id=OWNER, use_llm=False)
    assert semantic.graph is not None
    labels = {n["label"].lower() for n in semantic.graph.nodes(subject=CHILD, owner_user_id=OWNER)}
    assert "peanuts" in labels


def test_related_answers_across_two_separate_facts(semantic):
    """Neither sentence contains both "ulcer" and "hospital"."""
    semantic.remember("she has an ulcer in her stomach", subject=CHILD,
                      owner_user_id=OWNER, use_llm=False)
    semantic.remember("we saw a doctor at Mehr hospital", subject=CHILD,
                      owner_user_id=OWNER, use_llm=False)

    related = semantic.related(subject=CHILD, question="which hospital treated her ulcer?",
                               owner_user_id=OWNER)
    assert "Mehr hospital" in related, related


def test_a_broken_graph_does_not_stop_a_fact_being_stored(semantic, monkeypatch):
    """The sentence is the record; the graph is only an index over it."""
    class _Broken:
        def __getattr__(self, name):
            raise RuntimeError("graph is down")

    monkeypatch.setattr(type(semantic), "graph", property(lambda self: _Broken()))
    record = semantic.remember("she has an ulcer", subject=CHILD,
                               owner_user_id=OWNER, use_llm=False)
    assert record is not None
    assert semantic.recall(subject=CHILD, owner_user_id=OWNER)


def test_the_graph_can_be_switched_off_in_config(semantic, monkeypatch):
    from assistant import refdata

    cfg = dict(refdata.memory_config())
    cfg["graph"] = {**cfg.get("graph", {}), "enabled": False}
    monkeypatch.setattr(refdata, "memory_config", lambda: cfg)
    monkeypatch.setattr("assistant.memory.semantic.log", __import__("logging").getLogger("t"))
    semantic._graph_ready = False
    semantic._graph = None
    assert semantic.graph is None


# ---------------------------------------------------------------------------
# Extraction must be grammatical, not a vocabulary
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "sentence,expected",
    [
        # None of these appear in any list in the codebase.
        ("she has bronchiolitis on her chest", "bronchiolitis"),
        ("he has colic", "colic"),
        ("he developed the croup last night", "croup"),
        ("she got conjunctivitis at nursery", "conjunctivitis"),
        # ...and the original example still works.
        ("she has an ulcer in her stomach", "ulcer"),
    ],
)
def test_a_condition_is_captured_without_naming_it_in_advance(sentence, expected):
    """The pattern used to enumerate conditions, so anything unlisted vanished."""
    triples = extract_deterministic(sentence)
    conditions = [t["dst"] for t in triples if t["relation"] == "has_condition"]
    assert conditions, f"nothing captured from {sentence!r}"
    assert expected in conditions[0], conditions


def test_a_condition_containing_and_is_truncated_by_the_fallback():
    """A known limit of the pattern fallback, recorded rather than hidden.

    "hand foot and mouth" is one condition; "a rash and a fever" is two. The
    clause boundary cannot tell them apart without semantics, so it stops at
    "and" -- keeping the pair separate at the cost of truncating the compound
    name to "hand foot". A partial node still matches and still links; the LLM
    extraction path, which is tried first, gets the full name right. Widening
    the pattern to fix this would merge genuinely separate conditions, which
    is the worse error for a medical record.
    """
    triples = extract_deterministic("he has hand foot and mouth")
    assert triples[0]["dst"] == "hand foot"

    two = extract_deterministic("she has a rash and a fever")
    assert two[0]["dst"] == "rash", "two conditions were merged into one"


def test_the_article_is_not_left_attached_to_the_condition():
    """"(?:an?|some|the )?" matched the "a" of "an ulcer", leaving "n ulcer"."""
    triples = extract_deterministic("she has an ulcer in her stomach")
    assert triples[0]["dst"] == "ulcer"


def test_a_body_part_is_not_mistaken_for_a_medication():
    """Bare "on" read "a rash ON his back" as a drug called "his back"."""
    triples = extract_deterministic("he has a rash on his back")
    assert all(t["relation"] != "takes" for t in triples), triples


def test_a_trailing_time_phrase_is_not_part_of_the_condition():
    triples = extract_deterministic("he developed the croup last night")
    assert triples[0]["dst"] == "croup"


def test_ordinary_chat_produces_no_entities():
    """Every filler line from the scenario pool must stay out of the graph."""
    from tests import scenarios

    for line in scenarios.CHATTER:
        assert extract_deterministic(line) == [], line
