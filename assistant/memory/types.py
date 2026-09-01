"""The shapes every memory layer and backend agrees on.

Four kinds of memory, kept apart because they answer different questions and
decay at different rates:

  PROCEDURAL  how the assistant should behave. Durable, small, and rarely
              about one child -- house rules plus what a parent has corrected
              us on ("stop repeating yourself").
  SEMANTIC    durable facts about a child or parent: an allergy, a diagnosis,
              a clinic, a preference. Survives every session.
  EPISODIC    what was actually said, turn by turn. Large, mostly stale, and
              worth retrieving by relevance rather than replaying wholesale.

A fact is not simply true; it is true for a while. A child grows out of an
age band, an eczema patch clears, a parent switches clinic. So a record
carries `valid_from`/`valid_to` and is superseded rather than overwritten,
which keeps "where was her ulcer?" answerable after the ulcer heals and stops
a stale fact from being asserted as current.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from typing import Any

PROCEDURAL = "procedural"
SEMANTIC = "semantic"
EPISODIC = "episodic"

KINDS = (PROCEDURAL, SEMANTIC, EPISODIC)


def utc_now() -> str:
    """Timestamp with microsecond precision, because ordering depends on it.

    At second resolution every turn of a conversation shared one timestamp --
    six consecutive calls returned the same string -- so "ORDER BY created_at"
    was arbitrary among them. That made recall non-deterministic run to run,
    could show the exchange out of order, and mattered most to consolidation,
    which slices episodes[folded:] on the assumption that the order is stable
    and would otherwise fold the wrong turns.
    """
    return datetime.now(timezone.utc).isoformat(timespec="microseconds")


def new_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True)
class MemoryRecord:
    """One remembered thing, whatever backend holds it."""

    text: str
    kind: str
    # Whose memory this is: a child_id for child facts, a user id for parent
    # preferences. Scoping is by subject AND owner, never by subject alone --
    # a child id guessed by another account must not reach this store.
    subject: str = ""
    owner_user_id: str | None = None
    id: str = field(default_factory=new_id)
    created_at: str = field(default_factory=utc_now)
    # Where it came from, so a consolidated claim can be told apart from
    # something the parent said outright.
    source: str = "chat"
    # 0..1. Something stated by the parent outranks something inferred.
    confidence: float = 1.0
    valid_from: str | None = None
    valid_to: str | None = None
    attributes: dict[str, Any] = field(default_factory=dict)

    def is_current(self, at: str | None = None) -> bool:
        """A superseded fact is still remembered, just no longer asserted."""
        if self.valid_to is None:
            return True
        return (at or utc_now()) < self.valid_to

    def superseded(self, at: str | None = None) -> "MemoryRecord":
        return replace(self, valid_to=at or utc_now())

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "text": self.text,
            "kind": self.kind,
            "subject": self.subject,
            "owner_user_id": self.owner_user_id,
            "created_at": self.created_at,
            "source": self.source,
            "confidence": self.confidence,
            "valid_from": self.valid_from,
            "valid_to": self.valid_to,
            "attributes": dict(self.attributes),
        }

    @classmethod
    def from_dict(cls, row: dict[str, Any]) -> "MemoryRecord":
        known = {
            "id", "text", "kind", "subject", "owner_user_id", "created_at",
            "source", "confidence", "valid_from", "valid_to", "attributes",
        }
        data = {k: v for k, v in row.items() if k in known}
        data.setdefault("attributes", {})
        if not isinstance(data["attributes"], dict):
            data["attributes"] = {}
        data["confidence"] = float(data.get("confidence") or 1.0)
        return cls(**data)


ORIGINAL = "original"
ORIGINAL_LANG = "original_lang"


@dataclass(frozen=True)
class Episode:
    """One conversational turn, kept verbatim."""

    role: str
    content: str
    session_id: str
    subject: str = ""
    owner_user_id: str | None = None
    id: str = field(default_factory=new_id)
    created_at: str = field(default_factory=utc_now)
    attributes: dict[str, Any] = field(default_factory=dict)

    @property
    def spoken(self) -> str:
        """What the parent actually typed, which is not always `content`.

        A Persian turn reaches the agent translated, because the corpus and
        the retrieval index are English and a Persian sentence matches nothing
        in them. `content` is therefore the English the agent worked from --
        the right thing to search and the right thing to put in the prompt.

        It is the wrong thing to keep as the record. A child's history is what
        the parent said, and "دخترم کهیر دارد" is that record, not its gloss:
        machine translation drifts, and the gloss cannot be checked against
        anything once the original is gone. So the original rides along in
        `attributes` and is what this returns when there is one.
        """
        original = (self.attributes or {}).get(ORIGINAL)
        return original if isinstance(original, str) and original.strip() else self.content

    def as_line(self, cap: int | None = None) -> str:
        text = " ".join((self.content or "").split())
        if cap and len(text) > cap:
            text = text[: max(0, cap - 1)].rstrip() + "…"
        return f"{(self.role or '?').upper()}: {text}"
