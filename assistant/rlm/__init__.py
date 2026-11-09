"""Nestling RLM: FastAPI is a client of the TypeScript IDE orchestrator.

Knowledge lives as files under ``knowledge/`` and is read through the TS RLM
REPL. Embeddings are used only for user long-term memory, never for book RAG.
"""

from __future__ import annotations

from assistant.rlm.host import RlmHost, get_rlm_host

__all__ = ["RlmHost", "get_rlm_host"]
