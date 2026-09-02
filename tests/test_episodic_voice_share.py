"""An assistant turn may not crowd out the parent's own words.

The assistant writes four sentences where a parent writes one line. Sharing a
single per-line cap, its earlier reply took most of the episodic budget -- and
because that reply is generated, whatever it guessed came back into the next
prompt sitting beside things the parent actually said. Measured on a real
model, a reply that had mistaken bronchiolitis for a rash kept the whole
conversation on rashes.

The property below is stated for any parent and any wording: nothing is
asserted about particular sentences, only about whose voice gets the room.
"""

from __future__ import annotations

import pytest

from assistant.memory.backends.native import NativeMemoryBackend
from assistant.memory.system import MemorySystem
from assistant.settings import get_settings, reset_settings
from tests import scenarios

OWNER = "owner-1"
CHILD = "child-1"
SESSION = "s1"


@pytest.fixture
def mem(tmp_path, monkeypatch):
    monkeypatch.setenv("NESTLING_MEMORY_DB", str(tmp_path / "memory.db"))
    reset_settings()
    system = MemorySystem(NativeMemoryBackend(tmp_path / "memory.db"))
    yield system
    system.close()
    reset_settings()


def _converse(mem, sc):
    """A parent stating short facts, each answered at length."""
    for fact in sc.facts:
        mem.observe(session_id=SESSION, role="user", content=fact,
                    subject=CHILD, owner_user_id=OWNER)
        mem.observe(session_id=SESSION, role="assistant",
                    content=" ".join(sc.chatter) * 4,
                    subject=CHILD, owner_user_id=OWNER)


ALL = scenarios.many(12)


@pytest.mark.parametrize("sc", ALL, ids=lambda s: f"seed{s.seed}")
def test_an_older_reply_never_takes_more_room_than_the_parent(mem, sc):
    _converse(mem, sc)
    rendered = mem.episodic.render(
        session_id=SESSION, owner_user_id=OWNER, query=sc.where_question
    )
    said = [len(line) for line in rendered.splitlines()
            if line.upper().startswith("USER")]
    replies = [len(line) for line in rendered.splitlines()
               if line.upper().startswith("ASSISTANT")]
    assert said and replies, rendered
    # The last reply keeps its full length; every earlier one is held to the
    # longest thing the parent said.
    assert sorted(replies)[0] <= max(said) + len("ASSISTANT: "), f"{sc}\n{rendered}"


@pytest.mark.parametrize("sc", ALL, ids=lambda s: f"seed{s.seed}")
def test_the_latest_reply_survives_intact_for_a_bare_follow_up(mem, sc):
    """"is that good?" refers to what was just said, so keep that one whole."""
    _converse(mem, sc)
    rendered = mem.episodic.render(
        session_id=SESSION, owner_user_id=OWNER, query="is that good?"
    )
    replies = [line for line in rendered.splitlines()
               if line.upper().startswith("ASSISTANT")]
    cap = get_settings().nestling_memory_line_chars
    assert replies, rendered
    assert max(len(r) for r in replies) >= cap, f"{sc}\n{rendered}"


@pytest.mark.parametrize("sc", ALL, ids=lambda s: f"seed{s.seed}")
def test_what_the_parent_said_is_never_truncated_away(mem, sc):
    """Compressing replies is only worth doing if the facts then fit."""
    _converse(mem, sc)
    rendered = mem.episodic.render(
        session_id=SESSION, owner_user_id=OWNER, query=sc.allergy_question,
        budget_chars=900,
    ).lower()
    assert sc.allergen.lower() in rendered, f"{sc}\n{rendered}"
