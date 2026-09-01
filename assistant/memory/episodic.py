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

from assistant.memory.types import ORIGINAL, ORIGINAL_LANG, Episode
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
        original: str = "",
    ) -> Episode | None:
        """Store one turn.

        `original` is the parent's own sentence when `content` is a
        translation of it. Both are kept: `content` is what retrieval matches
        and what the prompt carries, because the index and the care corpus are
        English; `original` is the record, because a child's history is what
        the parent said and a translation of it is a copy that can drift. See
        Episode.spoken.
        """
        content = (content or "").strip()
        if not content or not session_id:
            return None
        attrs = dict(attributes or {})
        original = (original or "").strip()
        # Only when it says something `content` does not. Storing an identical
        # copy of every English turn would double the store for nothing.
        if original and original != content:
            from assistant.runtime_translate import detect_lang

            attrs[ORIGINAL] = original
            attrs.setdefault(ORIGINAL_LANG, detect_lang(original))
        episode = Episode(
            role=role,
            content=content,
            session_id=session_id,
            subject=subject,
            owner_user_id=owner_user_id,
            attributes=attrs,
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
        """Turns as prompt lines, oldest first so the exchange reads forwards.

        Recent turns are always included and relevance is added on top, rather
        than replacing them. Filtering purely by relevance broke follow-ups:
        "and what about at night?" matched "night" somewhere unrelated, the
        filter engaged, and the turn that said what the conversation was about
        was dropped for scoring zero. A conversation needs its recent context
        whether or not it shares words with the question; relevance is for
        reaching further back than that.
        """
        settings = get_settings()
        cap = settings.nestling_memory_line_chars
        recent = self.recall(
            session_id=session_id,
            subject=subject,
            owner_user_id=owner_user_id,
            limit=settings.nestling_history_window,
        )
        episodes = list(recent)
        relevant: set[str] = set()
        if query.strip():
            seen = {e.id for e in episodes}
            for hit in self.recall(
                session_id=session_id,
                subject=subject,
                owner_user_id=owner_user_id,
                query=query,
                limit=50,
            ):
                relevant.add(hit.id)
                if hit.id not in seen:
                    episodes.append(hit)
                    seen.add(hit.id)
            # Chronological, so the exchange still reads forwards.
            episodes.sort(key=lambda e: e.created_at)
        if budget_chars is None:
            return "\n".join(e.as_line(cap) for e in episodes)

        # Trimming used to work from the front, on the reasoning that the
        # oldest turn is the one worth losing. That silently destroyed the
        # feature it was meant to serve: a turn found by relevance is old by
        # definition -- that is why it was not in the recent window -- so it
        # sorted to the front and was the first thing dropped. A parent asking
        # about something said earlier got back only the last few lines of
        # small talk.
        #
        # So spend the budget on the turns the question actually matched
        # first, and fill what is left with recent context, oldest dropped
        # first. Order is restored afterwards, so the exchange still reads
        # forwards whichever turns survived.
        chosen: set[str] = set()
        used = 0
        for episode in episodes:
            if episode.id not in relevant:
                continue
            line = episode.as_line(cap)
            if used + len(line) + 1 > budget_chars:
                break
            chosen.add(episode.id)
            used += len(line) + 1
        for episode in reversed(episodes):
            if episode.id in chosen:
                continue
            line = episode.as_line(cap)
            if used + len(line) + 1 > budget_chars:
                break
            chosen.add(episode.id)
            used += len(line) + 1
        return "\n".join(e.as_line(cap) for e in episodes if e.id in chosen)
