"""Load and render prompt templates from the prompts/ directory."""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path

from assistant.settings import get_settings


def prompts_dir() -> Path:
    s = get_settings()
    override = getattr(s, "rlm_prompts_dir", None)
    if override:
        return Path(override).resolve()
    return Path(s.nestling_root).resolve() / "prompts"


@lru_cache(maxsize=64)
def get_prompt(rel: str) -> str:
    path = prompts_dir() / rel
    if not path.is_file():
        raise FileNotFoundError(f"Missing prompt template: {path}")
    return path.read_text(encoding="utf-8")


def render(template: str, **kwargs: object) -> str:
    """Replace ``{{key}}`` placeholders. Unknown keys stay as-is."""

    def repl(m: re.Match[str]) -> str:
        key = m.group(1).strip()
        if key in kwargs:
            return str(kwargs[key])
        return m.group(0)

    return re.sub(r"\{\{\s*([a-zA-Z0-9_]+)\s*\}\}", repl, template)


def pediatric_system_overlay() -> str:
    """Extra system rules for Nestling — loaded from prompts if present, else built-in file."""
    rel = "nestling/pediatric.md"
    path = prompts_dir() / rel
    if path.is_file():
        return path.read_text(encoding="utf-8")
    return (
        "You are Nestling, a parent-facing pediatric assistant.\n"
        "Never invent growth percentiles, z-scores, or screening scores — call clinical tools.\n"
        "For care guidance (feeding, sleep, iron, milestones, danger signs), search and read "
        "files under the knowledge/ workspace with fs.* and search.grep.\n"
        "Speak to parents clearly; do not dump clinician-only protocols.\n"
        "When the parent writes in Persian, answer in Persian; otherwise match their language.\n"
    )


def clear_prompt_cache() -> None:
    get_prompt.cache_clear()
