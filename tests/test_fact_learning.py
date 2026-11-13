"""How long-term memory decides what to keep from a parent's message.

The model lists the facts a message states; these rules, which need no model,
decide whether each one is new, a correction, or already known.
"""

from __future__ import annotations

import json

from assistant.rlm.memory import LongTermMemory, _already_known, _grounded, _same_subject

SAVED = [
    {"id": "f1", "text": "Darya has a fever since last night."},
    {"id": "f2", "text": "Darya started crawling last week."},
    {"id": "f3", "text": "Darya has mild eczema and is breastfed with some formula at nursery."},
]


def test_a_reworded_fact_is_already_known():
    assert _already_known("Darya started crawling last week", SAVED)
    # Contained in a longer saved fact, though shorter than it.
    assert _already_known("Darya still has mild eczema.", SAVED)


def test_a_fact_that_adds_something_is_new():
    assert not _already_known("Darya now pulls herself up to stand.", SAVED)
    assert not _already_known("Darya is allergic to peanuts.", SAVED)


def test_a_correction_replaces_only_a_fact_about_the_same_thing():
    assert _same_subject("Darya's fever has gone.", SAVED[0]["text"], SAVED)
    # Standing does not make crawling untrue; the child's name alone is no link.
    assert not _same_subject("Darya is now able to stand.", SAVED[1]["text"], SAVED)


def test_grounding_sets_aside_what_saved_facts_establish():
    established = {"darya", "has", "fever", "since", "last", "night"}
    # A parent says "she", not the name: the new words carry the check.
    assert _grounded("Darya is now able to stand.", "she now pulls herself up to stand", established)
    # A fact the message never stated is not grounded in it.
    assert not _grounded("Darya is allergic to peanuts.", "what should she eat?", established)


def _memory(tmp_path, monkeypatch):
    monkeypatch.setenv("NESTLING_DATA_DIR", str(tmp_path))
    from assistant.settings import get_settings

    get_settings.cache_clear()
    return LongTermMemory()


def _model(facts):
    def complete(*_a, **_k):
        return json.dumps({"facts": facts})

    return complete


def test_learning_adds_new_facts_corrects_old_ones_and_skips_repeats(tmp_path, monkeypatch):
    m = _memory(tmp_path, monkeypatch)
    u = "acct|child"
    for row in SAVED:
        m._add(u, None, row["text"], "general")
    ids = {f["text"]: f["id"] for f in m.list_facts(u)}
    fever, crawling = ids[SAVED[0]["text"]], ids[SAVED[1]["text"]]

    m.learn_facts(
        user_id=u,
        workspace_id=None,
        user_text="Her fever is gone now. She now pulls herself up to stand and still has mild eczema.",
        assistant_text="",
        complete_fn=_model([
            {"text": "Darya's fever has gone.", "changes": fever},
            # Wrongly linked by the model: must not erase the crawling fact.
            {"text": "Darya now pulls herself up to stand.", "changes": crawling},
            {"text": "Darya still has mild eczema.", "changes": ""},
        ]),
    )
    texts = sorted(f["text"] for f in m.list_facts(u))
    assert "Darya's fever has gone." in texts
    assert SAVED[0]["text"] not in texts
    assert SAVED[1]["text"] in texts
    assert "Darya now pulls herself up to stand." in texts
    assert len(texts) == 4, texts


def test_a_question_saves_nothing_even_if_the_model_echoes_saved_facts(tmp_path, monkeypatch):
    m = _memory(tmp_path, monkeypatch)
    u = "acct|child"
    for row in SAVED:
        m._add(u, None, row["text"], "general")
    m.learn_facts(
        user_id=u,
        workspace_id=None,
        user_text="What should she eat at her age?",
        assistant_text="",
        complete_fn=_model([{"text": r["text"], "changes": ""} for r in SAVED]),
    )
    assert len(m.list_facts(u)) == 3
