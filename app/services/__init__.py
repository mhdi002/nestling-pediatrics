"""Service layer: ParentAssistant factory + shared app Services container."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from assistant.agent.orchestrator import ParentAssistant
from assistant.memory.chat_memory import ChatMemory, ChatMemoryDB
from assistant.memory.child_db import ChildMemoryDB


@dataclass
class Services:
    db: ChildMemoryDB
    chat: ChatMemory
    assistant: ParentAssistant

    def close(self) -> None:
        try:
            self.assistant.close()
        except Exception:
            try:
                self.db.close()
            except Exception:
                pass
            try:
                self.chat.close()
            except Exception:
                pass


_services: Services | None = None


def load_models_enabled() -> bool:
    """True when NESTLING_LOAD_MODELS=1 (loads optional xLAM)."""
    return os.environ.get("NESTLING_LOAD_MODELS", "0") == "1"


def get_assistant(
    child_db: ChildMemoryDB | None = None,
    chat_db: ChatMemory | ChatMemoryDB | None = None,
    use_xlam: bool | None = None,
    use_pleias: bool | None = None,
) -> ParentAssistant:
    """Factory — xLAM only when NESTLING_LOAD_MODELS=1; generative RAG follows NESTLING_USE_LLM."""
    load = load_models_enabled()
    return ParentAssistant(
        db=child_db,
        chat_memory=chat_db,
        use_xlam=load if use_xlam is None else use_xlam,
        # None → ParentAssistant enables sidecar LLM via llm_enabled()
        use_pleias=use_pleias,
    )


def chat(
    asst: ParentAssistant,
    session_id: str,
    message: str,
    child_id: str | None = None,
) -> dict[str, Any]:
    return asst.chat(session_id, message, child_id=child_id)


def create_session(asst: ParentAssistant, child_id: str | None = None) -> str:
    return asst.start_session(child_id=child_id)


def create_services(
    child_db_path: Path | None = None,
    chat_db_path: Path | None = None,
    use_xlam: bool | None = None,
    use_pleias: bool | None = None,
) -> Services:
    load = load_models_enabled()
    db = ChildMemoryDB(path=child_db_path)
    chat_mem = ChatMemory(path=chat_db_path)
    assistant = ParentAssistant(
        db=db,
        chat_memory=chat_mem,
        use_xlam=load if use_xlam is None else use_xlam,
        # Do not force use_pleias=False when LOAD_MODELS=0 — that disabled Qwen RAG.
        use_pleias=use_pleias,
    )
    return Services(db=db, chat=chat_mem, assistant=assistant)


def get_services() -> Services:
    global _services
    if _services is None:
        _services = create_services()
    return _services


def peek_services() -> Services | None:
    return _services


def set_services(services: Services | None) -> None:
    global _services
    _services = services


__all__ = [
    "Services",
    "get_assistant",
    "create_session",
    "chat",
    "create_services",
    "get_services",
    "peek_services",
    "set_services",
    "load_models_enabled",
]
