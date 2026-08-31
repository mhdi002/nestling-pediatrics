"""Semantic memory: durable facts about a child or a parent.

What survives the session. An allergy, a diagnosis, the clinic they attend, a
preference about how the parent wants things explained. Retrieved by relevance
to the question in hand, not replayed wholesale, because a prompt carrying
every fact ever recorded is how a question about an ulcer gets answered with
whatever else was in the pile.

Facts are superseded, never overwritten. A child outgrows an age band, a rash
clears, a family changes clinic -- and the old fact stays retrievable as
history while stopping short of being asserted as current.
"""

from __future__ import annotations

import logging

from assistant.memory.types import SEMANTIC, MemoryRecord, utc_now
from assistant.settings import get_settings

log = logging.getLogger(__name__)


def _clean(text: str) -> str:
    return " ".join((text or "").split())


class SemanticMemory:
    """Durable facts, scoped to one child and one account."""

    def __init__(self, backend):
        self.backend = backend

    def remember(
        self,
        text: str,
        *,
        subject: str,
        owner_user_id: str | None = None,
        source: str = "chat",
        confidence: float = 1.0,
        attributes: dict | None = None,
        supersedes: str | None = None,
    ) -> MemoryRecord | None:
        text = _clean(text)
        if not text or not subject:
            return None
        if supersedes:
            self.backend.supersede_fact(supersedes)
        record = MemoryRecord(
            text=text,
            kind=SEMANTIC,
            subject=subject,
            owner_user_id=owner_user_id,
            source=source,
            confidence=float(confidence),
            valid_from=utc_now(),
            attributes=dict(attributes or {}),
        )
        try:
            return self.backend.remember_fact(record)
        except Exception as exc:  # a memory write must not break a chat turn
            log.warning("Could not store semantic fact: %s", exc)
            return None

    def recall(
        self,
        *,
        subject: str,
        owner_user_id: str | None = None,
        query: str = "",
        limit: int | None = None,
    ) -> list[MemoryRecord]:
        if not subject:
            return []
        limit = limit or get_settings().nestling_child_memory_events
        try:
            return self.backend.recall_facts(
                subject, owner_user_id=owner_user_id, query=query, limit=limit
            )
        except Exception as exc:
            log.warning("Could not recall semantic facts: %s", exc)
            return []

    def render(
        self,
        *,
        subject: str,
        owner_user_id: str | None = None,
        query: str = "",
        budget_chars: int | None = None,
    ) -> str:
        """Facts as prompt lines, most relevant first, truncated to budget.

        Ordered by relevance then confidence, so something the parent stated
        outranks something we inferred during consolidation when only one of
        them fits.
        """
        settings = get_settings()
        cap = settings.nestling_memory_line_chars
        records = self.recall(
            subject=subject, owner_user_id=owner_user_id, query=query, limit=50
        )
        records = sorted(records, key=lambda r: -float(r.confidence or 0))
        lines: list[str] = []
        used = 0
        for r in records:
            text = r.text if len(r.text) <= cap else r.text[: cap - 1].rstrip() + "…"
            line = f"- {text}"
            if budget_chars is not None and used + len(line) + 1 > budget_chars:
                break
            lines.append(line)
            used += len(line) + 1
        return "\n".join(lines)
