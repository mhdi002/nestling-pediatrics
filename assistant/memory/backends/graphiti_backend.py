"""Memory in a Graphiti temporal knowledge graph.

Graphiti is the open-source engine underneath Zep. Zep's own Community
Edition is deprecated and its hosted service would mean sending a child's
medical history to a third party, so this talks to Graphiti directly and
keeps everything on the host: entity extraction and embeddings both go
through the OpenAI-compatible endpoint this project already serves, so no
request leaves the machine.

What the graph buys over the native store is relationships and time. It
learns that a child, a clinic and a diagnosis are connected, and it records
when each became true, so "which hospital did I take her to?" can be answered
from a link rather than from a keyword that happens to appear in the same
sentence.

Three things are handled defensively, because all three have already happened
on the deployment host:

  The graph database may not be running. `available()` is checked before use
  and a failure means the native backend serves the turn instead.

  The model is small. Graphiti's own documentation warns that entity
  extraction needs reliable structured output and that smaller models may not
  give it. A failed extraction here is expected, not exceptional: the turn is
  still stored verbatim so nothing is lost, and only the graph enrichment is
  skipped.

  The package may be absent. `graphiti-core` is imported lazily, so a host
  that could not install it runs the native backend rather than failing at
  import time.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any

from assistant.memory.backends.native import NativeMemoryBackend
from assistant.memory.types import Episode, MemoryRecord
from assistant.settings import get_settings

log = logging.getLogger(__name__)


class GraphitiMemoryBackend:
    """Temporal knowledge graph, with the native store as its system of record.

    The native store is not a fallback bolted on the side: it is where the
    verbatim record lives. The graph adds relationships and recall over it.
    Writing to both means a graph that is wiped, corrupted or simply switched
    off never loses what a parent told us -- which for medical history is the
    property that matters most.
    """

    name = "graphiti"

    def __init__(self, spec: dict[str, Any] | None = None):
        self.spec = dict(spec or {})
        self.mirror = NativeMemoryBackend()
        self._graphiti = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.RLock()
        self._checked = False
        self._ok = False

    # -- lifecycle --------------------------------------------------------
    def _run(self, coro):
        """Run a coroutine on this backend's own loop.

        Graphiti's API is async and the app's request path is not, so the loop
        lives on a dedicated thread. Creating one per call would reconnect to
        the graph database on every turn.
        """
        with self._lock:
            if self._loop is None:
                self._loop = asyncio.new_event_loop()
                self._thread = threading.Thread(
                    target=self._loop.run_forever, name="graphiti-loop", daemon=True
                )
                self._thread.start()
        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result(timeout=get_settings().nestling_llm_timeout)

    def _client(self):
        if self._graphiti is not None:
            return self._graphiti
        from graphiti_core import Graphiti  # imported lazily on purpose
        from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
        from graphiti_core.llm_client.config import LLMConfig
        from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient

        settings = get_settings()
        llm_url = (self.spec.get("llm_base_url") or settings.nestling_llm_url or "").rstrip("/")
        if not llm_url:
            raise RuntimeError("no LLM endpoint configured for Graphiti")
        embed_url = (self.spec.get("embedder_base_url") or llm_url).rstrip("/")
        # vLLM ignores the key but the client requires one to be present.
        api_key = settings.nestling_llm_api_key or "not-needed"

        llm_config = LLMConfig(
            api_key=api_key,
            base_url=f"{llm_url}/v1",
            model=settings.nestling_llm_model,
            small_model=settings.nestling_llm_model,
        )
        embedder = OpenAIEmbedder(
            config=OpenAIEmbedderConfig(
                api_key=api_key,
                base_url=f"{embed_url}/v1",
                embedding_model=settings.nestling_memory_embedding_model,
            )
        )
        self._graphiti = Graphiti(
            graph_driver=self._graph_driver(),
            llm_client=OpenAIGenericClient(config=llm_config),
            embedder=embedder,
        )
        return self._graphiti

    def _graph_driver(self):
        settings = get_settings()
        kind = str(self.spec.get("graph_driver") or "falkordb").lower()
        uri = self.spec.get("graph_uri") or settings.nestling_memory_graph_uri
        if not uri:
            raise RuntimeError("no graph database URI configured")
        if kind == "falkordb":
            from graphiti_core.driver.falkordb_driver import FalkorDriver

            return FalkorDriver(uri=uri)
        if kind == "neo4j":
            from graphiti_core.driver.neo4j_driver import Neo4jDriver

            return Neo4jDriver(
                uri=uri,
                user=settings.nestling_memory_graph_user,
                password=settings.nestling_memory_graph_password,
            )
        raise RuntimeError(f"unsupported graph driver: {kind}")

    def available(self) -> bool:
        """Probed once. A graph that is not there is a reason to use native."""
        if self._checked:
            return self._ok
        self._checked = True
        try:
            self._run(self._client().build_indices_and_constraints())
            self._ok = True
            log.info("Graphiti memory backend ready")
        except Exception as exc:
            self._ok = False
            log.warning("Graphiti backend not available: %s", exc)
        return self._ok

    def _group(self, subject: str, owner_user_id: str | None) -> str:
        """Graph partition key.

        Owner is part of it, so one account's subgraph is not merely filtered
        out of another's results but never searched in the first place.
        """
        prefix = self.spec.get("group_prefix") or "nestling"
        return f"{prefix}:{owner_user_id or 'shared'}:{subject or 'general'}"

    def _tolerant(self) -> bool:
        return bool(self.spec.get("tolerate_extraction_failure", True))

    # -- semantic ---------------------------------------------------------
    def remember_fact(self, record: MemoryRecord) -> MemoryRecord:
        stored = self.mirror.remember_fact(record)
        if not self.available():
            return stored
        try:
            from graphiti_core.nodes import EpisodeType

            self._run(
                self._client().add_episode(
                    name=f"fact:{record.id}",
                    episode_body=record.text,
                    source=EpisodeType.text,
                    source_description=record.source or "fact",
                    reference_time=self._when(record.valid_from or record.created_at),
                    group_id=self._group(record.subject, record.owner_user_id),
                )
            )
        except Exception as exc:
            if not self._tolerant():
                raise
            log.warning("Graphiti could not ingest a fact (kept natively): %s", exc)
        return stored

    def recall_facts(
        self,
        subject: str,
        *,
        owner_user_id: str | None = None,
        query: str = "",
        limit: int = 10,
        include_superseded: bool = False,
    ) -> list[MemoryRecord]:
        """Graph search when there is a query, native ordering otherwise.

        The graph is asked first because it can follow a relationship the
        native index cannot, and its hits are matched back to the verbatim
        records so the caller always gets the parent's own words rather than
        an extracted paraphrase.
        """
        native = self.mirror.recall_facts(
            subject,
            owner_user_id=owner_user_id,
            query=query,
            limit=limit,
            include_superseded=include_superseded,
        )
        if not query.strip() or not self.available():
            return native
        try:
            edges = self._run(
                self._client().search(
                    query=query,
                    group_ids=[self._group(subject, owner_user_id)],
                    num_results=limit,
                )
            )
        except Exception as exc:
            log.warning("Graphiti search failed (using native results): %s", exc)
            return native
        return self._merge(edges, native, limit)

    @staticmethod
    def _merge(edges, native: list[MemoryRecord], limit: int) -> list[MemoryRecord]:
        """Order native records by what the graph found relevant.

        A graph hit that has no verbatim counterpart is dropped rather than
        surfaced: an extracted edge is the graph's paraphrase, and a parent
        should be read back their own words.
        """
        facts = [
            (getattr(e, "fact", "") or "").lower() for e in (edges or [])
        ]
        if not facts:
            return native
        ranked: list[MemoryRecord] = []
        seen: set[str] = set()
        for fact in facts:
            for record in native:
                if record.id in seen:
                    continue
                text = record.text.lower()
                if text and (text in fact or fact in text):
                    ranked.append(record)
                    seen.add(record.id)
                    break
        for record in native:
            if record.id not in seen:
                ranked.append(record)
        return ranked[:limit]

    def supersede_fact(self, record_id: str, *, at: str | None = None) -> bool:
        # Graphiti carries its own validity intervals; the mirror is the one
        # this project reads for currency.
        return self.mirror.supersede_fact(record_id, at=at)

    # -- episodic ---------------------------------------------------------
    def add_episode(self, episode: Episode) -> Episode:
        stored = self.mirror.add_episode(episode)
        if not self.available():
            return stored
        try:
            from graphiti_core.nodes import EpisodeType

            self._run(
                self._client().add_episode(
                    name=f"turn:{episode.id}",
                    episode_body=f"{episode.role}: {episode.content}",
                    source=EpisodeType.message,
                    source_description="chat turn",
                    reference_time=self._when(episode.created_at),
                    group_id=self._group(episode.subject, episode.owner_user_id),
                )
            )
        except Exception as exc:
            if not self._tolerant():
                raise
            log.warning("Graphiti could not ingest a turn (kept natively): %s", exc)
        return stored

    def recall_episodes(
        self,
        *,
        session_id: str | None = None,
        subject: str = "",
        owner_user_id: str | None = None,
        query: str = "",
        limit: int = 12,
    ) -> list[Episode]:
        # Turn recall stays native: it must return the exact words said, in
        # order, which is what the mirror holds.
        return self.mirror.recall_episodes(
            session_id=session_id,
            subject=subject,
            owner_user_id=owner_user_id,
            query=query,
            limit=limit,
        )

    def count_episodes(self, *, session_id: str, owner_user_id: str | None = None) -> int:
        return self.mirror.count_episodes(session_id=session_id, owner_user_id=owner_user_id)

    def forget(
        self, *, subject: str = "", session_id: str = "", owner_user_id: str | None = None
    ) -> int:
        """Deleting must reach the graph too, or 'forget' is a lie."""
        removed = self.mirror.forget(
            subject=subject, session_id=session_id, owner_user_id=owner_user_id
        )
        if subject and self.available():
            try:
                self._run(
                    self._client().remove_episodes_by_group_id(
                        self._group(subject, owner_user_id)
                    )
                )
            except Exception as exc:
                log.warning("Graphiti group not removed for %s: %s", subject, exc)
        return removed

    @staticmethod
    def _when(stamp: str | None):
        from datetime import datetime, timezone

        if not stamp:
            return datetime.now(timezone.utc)
        try:
            parsed = datetime.fromisoformat(stamp)
        except ValueError:
            return datetime.now(timezone.utc)
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
