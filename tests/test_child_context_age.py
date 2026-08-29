"""
The child digest must state the age, not leave the model to infer it.

A parent asked what a 2-month-old should eat and was told what a 13-month-old
needs. The digest carried only gestational and postmenstrual weeks
("GA 32.0w", "40.69w"), so the model had to derive a chronological age and
invented one. The age is now stated outright, and it comes from the stored
growth row -- computed with gestational age in scope -- or from date of birth.
"""

from __future__ import annotations

import pytest

from assistant.memory.child_db import ChildMemoryDB


@pytest.fixture()
def db(tmp_path):
    d = ChildMemoryDB(tmp_path / "children.db")
    try:
        yield d
    finally:
        d.close()


def test_digest_states_chronological_age(db):
    cid = db.create_child("monika", "female", gestational_age_weeks=32)
    db.add_growth(cid, 40.69, "weight", 3.2, age_months=2.0)
    ctx = db.child_context_text(cid)
    assert "months old" in ctx, ctx
    assert "2.0 months old" in ctx, ctx
    # The prematurity gap must not be lost: 40.69w PMA is 2 months of life,
    # not 9, and both numbers appear in the digest.
    assert "13" not in ctx


def test_age_from_date_of_birth_when_no_growth_row(db):
    from datetime import date, timedelta

    born = date.today() - timedelta(days=120)
    cid = db.create_child("Ada", "female", date_of_birth=born.isoformat())
    ctx = db.child_context_text(cid)
    assert "months old" in ctx, ctx


def test_no_age_claimed_when_unknowable(db):
    """Better to say nothing than to state an age that was guessed."""
    cid = db.create_child("Sam", "male", gestational_age_weeks=39)
    ctx = db.child_context_text(cid)
    assert "months old" not in ctx, ctx


def test_remembered_concern_survives_alongside_the_age(db):
    """The digest must carry both; forgetting either produced a live failure."""
    cid = db.create_child("monika", "female", gestational_age_weeks=32)
    db.add_growth(cid, 40.69, "weight", 3.2, age_months=2.0)
    db.remember_note(cid, "she has some ulcer in her stomach")
    ctx = db.child_context_text(cid)
    assert "ulcer" in ctx.lower(), ctx
    assert "months old" in ctx, ctx


def test_malformed_dob_does_not_raise(db):
    cid = db.create_child("Nil", "male", date_of_birth="not-a-date")
    assert isinstance(db.child_context_text(cid), str)
