"""Shared application services (DB, chat memory, orchestrator)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from assistant.agent.orchestrator import ParentAssistant
from assistant.memory.chat_memory import ChatMemory
from assistant.memory.child_db import ChildMemoryDB


@dataclass
class Services:
    db: ChildMemoryDB
    chat: ChatMemory
    assistant: ParentAssistant


_services: Services | None = None


def create_services(
    child_db_path: Path | None = None,
    chat_db_path: Path | None = None,
) -> Services:
    db = ChildMemoryDB(path=child_db_path)
    chat = ChatMemory(path=chat_db_path)
    assistant = ParentAssistant(
        db=db,
        chat_memory=chat,
        use_xlam=False,
        use_pleias=False,
    )
    return Services(db=db, chat=chat, assistant=assistant)


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
