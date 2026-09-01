"""Clinician-only procedures must never be read back to a parent as advice.

Part of the corpus comes from manuals written for skilled health workers
(WHO's PCPNC guide, the IYCF counselling course). Those books contain
procedures a parent must never be walked through -- newborn ventilation, IV
access, IM drug doses. Retrieval had no notion of audience, so a frightened
parent could be handed a resuscitation instruction as if it were advice.
"""

from __future__ import annotations

import pytest

from assistant.rag import audience
from assistant.rag.stores import MedicalRAG
from assistant.settings import get_settings, reset_settings

EMERGENCIES = [
    "my baby is not breathing",
    "my newborn is blue and floppy",
    "my baby is gasping and limp",
    "there is heavy bleeding after the birth",
    "my baby is having convulsions",
    "the baby came out not breathing",
]

ORDINARY = [
    "what foods are good for her?",
    "how often should I feed a two month old",
    "tell me about iron for breastfed babies",
    "when should my baby get vaccines",
    "my toddler has a rash on his arm",
    "how much should a 6 month old sleep",
    "what are the danger signs I should watch for",
    "how do I bathe my newborn",
    "when do babies start talking",
    "is my baby gaining enough weight",
    "what should I do about nappy rash",
    "how do I burp my baby",
]


@pytest.fixture(scope="module")
def rag():
    r = MedicalRAG()
    if not r.load():
        pytest.skip("medical index not built")
    return r


def _cited_docs(rag, out):
    for cite in out.get("citations") or []:
        yield next(d for d in rag.store.docs if d["id"] == cite["id"])


def test_every_source_declares_an_audience(rag):
    """A new document must be classified deliberately, not by omission."""
    undeclared = sorted(
        {
            d.get("source")
            for d in rag.store.docs
            if not audience.is_source_declared(d.get("source"))
        }
    )
    assert not undeclared, f"undeclared sources in the index: {undeclared}"


def test_the_corpus_keeps_both_audiences(rag):
    """Nothing is deleted -- clinician content stays indexed, just scoped."""
    labels = {d.get("audience") for d in rag.store.docs}
    assert audience.PARENT in labels
    assert audience.CLINICIAN in labels


@pytest.mark.parametrize("question", EMERGENCIES + ORDINARY)
def test_no_question_ever_returns_clinician_text(rag, question):
    """The safety guarantee, and it does not depend on any score heuristic."""
    out = rag.answer(question, use_llm=False)
    for doc in _cited_docs(rag, out):
        assert audience.is_parent_facing(doc), f"{doc['id']} is clinician-only"


@pytest.mark.parametrize("question", ORDINARY)
def test_ordinary_questions_still_get_an_answer(rag, question):
    """Scoping must not quietly empty the corpus."""
    out = rag.answer(question, use_llm=False)
    assert (out.get("answer") or "").strip()
    assert out.get("mode") != "emergency_escalation", "answered as an emergency"


def test_escalation_is_not_decided_by_retrieval_scores():
    """The score comparison that answered "what foods are good for her?" as an
    emergency is gone, not merely switched off.

    Its clinician/parent scores (11.94 / 8.74) outrank four of six genuine
    emergencies, so no threshold on that measure separates the classes.
    Anything reintroducing a score-based trigger here should fail this.
    """
    reset_settings()
    assert get_settings().nestling_urgent_escalation_enabled is True
    assert not hasattr(audience, "clinician_only_topic")


@pytest.mark.parametrize("question", EMERGENCIES)
def test_the_scope_filter_holds_even_when_a_turn_escalates(rag, question):
    """Escalation replaces the answer; it must not relax the audience filter."""
    out = rag.answer(question, use_llm=False)
    for doc in _cited_docs(rag, out):
        assert audience.is_parent_facing(doc), f"{doc['id']} is clinician-only"


def test_retrieval_is_scoped_even_without_going_through_load():
    """An unlabelled corpus fails closed per chunk, which would empty it.

    Loading the store directly used to leave every chunk unlabelled, so the
    filter dropped all of them and every question answered "I couldn't find a
    matching care note".
    """
    r = MedicalRAG()
    if not r.store.load():
        pytest.skip("medical index not built")
    hits = r.retrieve("tell me about iron for breastfed babies", top_k=5)
    assert hits, "scoping emptied the corpus"
    assert all(audience.is_parent_facing(h) for h in hits)
