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
    """Durable facts, scoped to one child and one account.

    Facts are kept twice on purpose: as the parent's own sentence, and as
    entities and relationships in a profile graph. The sentence is what gets
    read back to them; the graph is what lets a question walk from a condition
    to the clinic that treated it, which no amount of keyword overlap does.
    """

    def __init__(self, backend, graph=None):
        self.backend = backend
        self._graph = graph
        self._graph_ready = graph is not None

    @property
    def graph(self):
        """The profile graph, built beside the fact store on first use."""
        if self._graph_ready:
            return self._graph
        self._graph_ready = True
        try:
            from assistant.memory.graph import ProfileGraph
            from assistant.refdata import memory_config

            if not ((memory_config() or {}).get("graph") or {}).get("enabled", True):
                self._graph = None
            else:
                path = getattr(self.backend, "path", None)
                self._graph = ProfileGraph(path) if path else None
        except Exception as exc:
            log.warning("Profile graph unavailable: %s", exc)
            self._graph = None
        return self._graph

    def _ingest_graph(self, record, *, use_llm: bool) -> None:
        """Add a fact's entities to the graph. Never breaks the write."""
        graph = self.graph
        if graph is None or record is None:
            return
        try:
            from assistant.memory.extraction import ingest

            ingest(
                graph,
                record.text,
                subject=record.subject,
                owner_user_id=record.owner_user_id,
                fact_id=record.id,
                use_llm=use_llm,
            )
        except Exception as exc:
            log.warning("Could not add fact to the profile graph: %s", exc)

    def close(self) -> None:
        """Release the profile graph's connection.

        It is opened lazily on first use and was never closed, so every
        SemanticMemory left a SQLite handle open. Harmless in a long-running
        server, which builds one, but it made test teardown flaky on Windows:
        the temporary directory could not be removed while the file was held.
        """
        graph = self._graph
        self._graph = None
        self._graph_ready = False
        if graph is not None:
            try:
                graph.close()
            except Exception:  # noqa: BLE001 - closing must not raise
                pass

    def related(
        self,
        *,
        subject: str,
        question: str,
        owner_user_id: str | None = None,
        budget_chars: int | None = None,
    ) -> str:
        """What the graph knows around the entities this question mentions."""
        graph = self.graph
        if graph is None or not subject:
            return ""
        try:
            from assistant.refdata import memory_config

            hops = int(((memory_config() or {}).get("graph") or {}).get("hops") or 2)
            return graph.render(
                subject=subject,
                text=question,
                owner_user_id=owner_user_id,
                budget_chars=budget_chars,
                hops=hops,
            )
        except Exception as exc:
            log.warning("Profile graph lookup failed: %s", exc)
            return ""

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
        use_llm: bool = True,
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
            stored = self.backend.remember_fact(record)
        except Exception as exc:  # a memory write must not break a chat turn
            log.warning("Could not store semantic fact: %s", exc)
            return None
        self._ingest_graph(stored, use_llm=use_llm)
        return stored

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
