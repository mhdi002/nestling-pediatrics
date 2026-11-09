"""Canonical checkpoint format shared by session + in-turn compaction."""

from __future__ import annotations

import re

CHECKPOINT_SECTIONS = (
    "Goal",
    "Constraints & Preferences",
    "Facts & Values",
    "Tool Results",
    "Requests answered",
    "Open requests",
    "Dead ends",
)


def parse_checkpoint(text: str) -> dict[str, str]:
    out = {s: "" for s in CHECKPOINT_SECTIONS}
    current: str | None = None
    buf: list[str] = []

    def flush() -> None:
        nonlocal current
        if current:
            out[current] = "\n".join(buf).strip()
        buf.clear()

    for line in (text or "").split("\n"):
        m = re.match(r"^#{1,3}\s*(.+?)\s*$", line)
        heading = m.group(1) if m else None
        match = None
        if heading:
            for s in CHECKPOINT_SECTIONS:
                if s.lower() == heading.lower():
                    match = s
                    break
        if match:
            flush()
            current = match
        elif current:
            buf.append(line)
    flush()
    return out


def normalize_checkpoint(text: str) -> str:
    stripped = re.sub(r"<think>[\s\S]*?</think>", "", text or "").strip()
    parsed = parse_checkpoint(stripped)
    any_section = any(parsed[s] for s in CHECKPOINT_SECTIONS)
    if not any_section:
        parts = []
        for s in CHECKPOINT_SECTIONS:
            body = stripped or "(none)" if s == "Facts & Values" else "(none)"
            parts.append(f"## {s}\n{body}")
        return "\n\n".join(parts)
    return "\n\n".join(f"## {s}\n{parsed[s] or '(none)'}" for s in CHECKPOINT_SECTIONS)


def is_empty_checkpoint(text: str) -> bool:
    parsed = parse_checkpoint(text)
    return all(
        (not parsed[s]) or re.match(r"^\(none\)$", parsed[s].strip(), re.I)
        for s in CHECKPOINT_SECTIONS
    )
