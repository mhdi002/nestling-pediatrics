"""What a memory backend must do, and nothing about how.

Two implementations sit behind this: a native store built on the SQLite and
BM25 machinery this project already runs, and a Graphiti temporal graph. The
interface is deliberately small so a backend that is unreachable -- a graph
database that did not start, an embedding model the host cannot fetch -- can
be swapped for the native one without the layers above noticing.

Every method takes an owner. Scoping by subject alone once let one account
read another's child, so a backend that ignores `owner_user_id` is a bug
rather than a simplification.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from assistant.memory.types import Episode, MemoryRecord


@runtime_checkable
class MemoryBackend(Protocol):
    """Storage and recall for semantic facts and episodic turns."""

    name: str

    def available(self) -> bool:
        """Whether this backend can serve requests right now.

        Checked before use rather than assumed, so a backend whose service is
        down degrades instead of raising into a parent's chat turn.
        """

    # -- semantic ---------------------------------------------------------
    def remember_fact(self, record: MemoryRecord) -> MemoryRecord:
        """Store a durable fact, superseding any it replaces."""

    def recall_facts(
        self,
        subject: str,
        *,
        owner_user_id: str | None = None,
        query: str = "",
        limit: int = 10,
        include_superseded: bool = False,
    ) -> list[MemoryRecord]:
        """Facts about `subject`, most relevant first when `query` is given."""

    def supersede_fact(self, record_id: str, *, at: str | None = None) -> bool:
        """Mark a fact no longer current. It stays retrievable as history."""

    # -- episodic ---------------------------------------------------------
    def add_episode(self, episode: Episode) -> Episode:
        """Record one conversational turn."""

    def recall_episodes(
        self,
        *,
        session_id: str | None = None,
        subject: str = "",
        owner_user_id: str | None = None,
        query: str = "",
        limit: int = 12,
    ) -> list[Episode]:
        """Turns, by relevance when `query` is given and recency otherwise.

        Relevance matters more than recency here: a parent asking about an
        ulcer mentioned four sessions ago is not served by the last twelve
        turns about sleep.
        """

    def count_episodes(self, *, session_id: str, owner_user_id: str | None = None) -> int:
        """How many turns a session holds, for the consolidation trigger."""

    def forget(self, *, subject: str = "", session_id: str = "", owner_user_id: str | None = None) -> int:
        """Delete memory for a subject or session. Returns rows removed."""
