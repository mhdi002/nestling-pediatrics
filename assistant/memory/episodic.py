"""Episodic memory: what was actually said.

The largest and least useful-per-byte of the four, which is exactly why it
needs care. Replaying the last N turns is cheap and usually wrong: a parent
asking about an ulcer mentioned four sessions ago is not served by twelve
recent turns about sleep, and those twelve turns crowd out the one line that
answers them.

So recall is by relevance to the current question, falling back to recency
only when there is no question to match against. Turns are kept verbatim --
consolidation distils them into semantic memory separately, and a distilled
claim never replaces the record of what was said.
"""

from __future__ import annotations

import logging

from assistant.memory.types import Episode
from assistant.settings import get_settings

log = logging.getLogger(__name__)


class EpisodicMemory:
    """Conversational turns, recalled by relevance."""

    def __init__(self, backend):
        self.backend = backend

    def record(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        subject: str = "",
        owner_user_id: str | None = None,
        attributes: dict | None = None,
    ) -> Episode | None:
        content = (content or "").strip()
        if not content or not session_id:
            return None
        episode = Episode(
            role=role,
            content=content,
            session_id=session_id,
            subject=subject,
            owner_user_id=owner_user_id,
            attributes=dict(attributes or {}),
        )
        try:
            return self.backend.add_episode(episode)
        except Exception as exc:  # never break a chat turn over a memory write
            log.warning("Could not store episode: %s", exc)
            return None

    def recall(
        self,
        *,
        session_id: str | None = None,
        subject: str = "",
        owner_user_id: str | None = None,
        query: str = "",
        limit: int | None = None,
    ) -> list[Episode]:
        limit = limit or get_settings().nestling_history_window
        try:
            return self.backend.recall_episodes(
                session_id=session_id,
                subject=subject,
                owner_user_id=owner_user_id,
                query=query,
                limit=limit,
            )
        except Exception as exc:
            log.warning("Could not recall episodes: %s", exc)
            return []

    def recall_across_sessions(
        self,
        *,
        subject: str,
        owner_user_id: str | None = None,
        query: str,
        limit: int | None = None,
    ) -> list[Episode]:
        """Relevant turns about this child from ANY session.

        A new session starts with no history, but the child's thread did not
        start with it: what the parent said last week is still the answer to
        what they are asking now.
        """
        if not subject or not query.strip():
            return []
        return self.recall(
            subject=subject, owner_user_id=owner_user_id, query=query, limit=limit
        )

    def count(self, *, session_id: str, owner_user_id: str | None = None) -> int:
        try:
            return self.backend.count_episodes(
                session_id=session_id, owner_user_id=owner_user_id
            )
        except Exception as exc:
            log.warning("Could not count episodes: %s", exc)
            return 0

    def render(
        self,
        *,
        session_id: str | None = None,
        subject: str = "",
        owner_user_id: str | None = None,
        query: str = "",
        budget_chars: int | None = None,
    ) -> str:
        """Turns as prompt lines, oldest first so the exchange reads forwards."""
        cap = get_settings().nestling_memory_line_chars
        episodes = self.recall(
            session_id=session_id,
            subject=subject,
            owner_user_id=owner_user_id,
            query=query,
            limit=50,
        )
        lines = [e.as_line(cap) for e in episodes]
        if budget_chars is None:
            return "\n".join(lines)
        # Trim from the front: when the budget bites, the oldest turn is the
        # one worth losing.
        kept: list[str] = []
        used = 0
        for line in reversed(lines):
            if used + len(line) + 1 > budget_chars:
                break
            kept.append(line)
            used += len(line) + 1
        return "\n".join(reversed(kept))
