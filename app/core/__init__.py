"""App core settings — re-exports assistant.config for a clean layout."""

from assistant.config import (  # noqa: F401
    CHAT_DB_PATH,
    CHILD_DB_PATH,
    EN_DIR,
    EXTRACTED,
    OVERLAY_DIR,
    ROOT,
)

__all__ = [
    "ROOT",
    "EXTRACTED",
    "EN_DIR",
    "CHILD_DB_PATH",
    "CHAT_DB_PATH",
    "OVERLAY_DIR",
]
