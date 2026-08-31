"""The memory system: one object the agent talks to.

Holds the four layers and the backend they share, and is the only place that
knows how a turn flows through them:

    remember a turn  ->  episodic
    every N turns    ->  consolidation  ->  semantic
    building a reply ->  procedural + semantic + episodic, within a budget

Backend selection is deliberate about failure. A backend that cannot serve --
a graph database that did not start, an embedding model the host cannot fetch
-- is replaced by the native store rather than raising into a parent's chat
turn. Memory degrades; it does not break the conversation.
"""

from __future__ import annotations

import logging

from assistant.memory.assembly import assemble, budget
from assistant.memory.consolidation import Consolidator, due
from assistant.memory.episodic import EpisodicMemory
from assistant.memory.procedural import ProceduralMemory
from assistant.memory.semantic import SemanticMemory
from assistant.refdata import memory_config
from assistant.settings import get_settings

log = logging.getLogger(__name__)


def build_backend(name: str | None = None):
    """Resolve a backend by name from config/memory.yaml, or fall back.

    Never raises: an unusable backend is a reason to use the native store, not
    a reason for the app to fail to start.
    """
    from assistant.memory.backends.native import NativeMemoryBackend

    settings = get_settings()
    name = (name or settings.nestling_memory_backend or "native").strip()
    declared = (memory_config() or {}).get("backends") or {}
    spec = declared.get(name) or {}
    driver = str(spec.get("driver") or name).lower()

    if driver == "native":
        return NativeMemoryBackend()

    if driver == "graphiti":
        try:
            from assistant.memory.backends.graphiti_backend import GraphitiMemoryBackend

            backend = GraphitiMemoryBackend(spec)
            if backend.available():
                return backend
            log.warning("Graphiti backend unavailable -- using native memory")
        except Exception as exc:
            log.warning("Graphiti backend could not start (%s) -- using native", exc)
        return NativeMemoryBackend()

    log.warning("Unknown memory backend %r -- using native", name)
    return NativeMemoryBackend()


class MemorySystem:
    """Procedural, semantic and episodic memory over one backend."""

    def __init__(self, backend=None):
        self.backend = backend if backend is not None else build_backend()
        self.procedural = ProceduralMemory(self.backend)
        self.semantic = SemanticMemory(self.backend)
        self.episodic = EpisodicMemory(self.backend)
        self.consolidator = Consolidator(self.semantic, self.episodic)

    @property
    def backend_name(self) -> str:
        return getattr(self.backend, "name", "unknown")

    # -- writing ----------------------------------------------------------
    def observe(
        self,
        *,
        session_id: str,
        role: str,
        content: str,
        subject: str = "",
        owner_user_id: str | None = None,
        attributes: dict | None = None,
    ):
        """Record one turn."""
        return self.episodic.record(
            session_id=session_id,
            role=role,
            content=content,
            subject=subject,
            owner_user_id=owner_user_id,
            attributes=attributes,
        )

    def maybe_consolidate(
        self,
        *,
        session_id: str,
        subject: str,
        owner_user_id: str | None = None,
        use_llm: bool = True,
        force: bool = False,
    ) -> list[str]:
        """Fold episodes into durable facts once enough has been said.

        The watermark lives on the session so a stretch is folded exactly
        once: moving it before extraction would lose turns if the model call
        failed, so it moves only after facts are stored.
        """
        if not subject:
            return []
        total = self.episodic.count(session_id=session_id, owner_user_id=owner_user_id)
        folded = self._watermark(session_id)
        if not force and not due(total - folded):
            return []
        episodes = self.episodic.recall(
            session_id=session_id, owner_user_id=owner_user_id, limit=500
        )
        pending = episodes[folded:] if folded < len(episodes) else []
        if not pending:
            return []
        stored = self.consolidator.consolidate(
            session_id=session_id,
            subject=subject,
            owner_user_id=owner_user_id,
            episodes=pending,
            use_llm=use_llm,
        )
        self._set_watermark(session_id, total)
        return stored

    # The watermark is kept with the facts so a backend swap keeps it.
    def _watermark(self, session_id: str) -> int:
        records = self.backend.recall_facts(
            f"session:{session_id}", limit=1, include_superseded=True
        )
        if not records:
            return 0
        try:
            return int(records[0].attributes.get("consolidated_upto") or 0)
        except (TypeError, ValueError):
            return 0

    def _set_watermark(self, session_id: str, value: int) -> None:
        from assistant.memory.types import SEMANTIC, MemoryRecord

        subject = f"session:{session_id}"
        for existing in self.backend.recall_facts(subject, limit=10, include_superseded=True):
            self.backend.supersede_fact(existing.id)
        self.backend.remember_fact(
            MemoryRecord(
                text=f"consolidated {value} turns",
                kind=SEMANTIC,
                subject=subject,
                source="watermark",
                attributes={"consolidated_upto": int(value)},
            )
        )

    # -- reading ----------------------------------------------------------
    def context_for(
        self,
        *,
        question: str,
        session_id: str | None = None,
        subject: str = "",
        owner_user_id: str | None = None,
        intents: set[str] | None = None,
        working: str = "",
        total_chars: int | None = None,
    ):
        """The labelled, budgeted context for this turn."""
        caps = budget(total_chars)
        procedural = self.procedural.render(
            intents=intents, owner_user_id=owner_user_id, budget_chars=caps.procedural
        )
        semantic = (
            self.semantic.render(
                subject=subject,
                owner_user_id=owner_user_id,
                query=question,
                budget_chars=caps.semantic,
            )
            if subject
            else ""
        )
        episodic = self.episodic.render(
            session_id=session_id,
            owner_user_id=owner_user_id,
            query=question,
            budget_chars=caps.episodic,
        )
        return assemble(
            procedural=procedural,
            semantic=semantic,
            episodic=episodic,
            working=working,
            total_chars=total_chars,
        )

    def forget(
        self, *, subject: str = "", session_id: str = "", owner_user_id: str | None = None
    ) -> int:
        return self.backend.forget(
            subject=subject, session_id=session_id, owner_user_id=owner_user_id
        )

    def close(self) -> None:
        """Release every connection this system opened."""
        for closer in (getattr(self.semantic, "close", None),
                       getattr(self.backend, "close", None)):
            if callable(closer):
                try:
                    closer()
                except Exception:  # noqa: BLE001 - closing must not raise
                    pass
