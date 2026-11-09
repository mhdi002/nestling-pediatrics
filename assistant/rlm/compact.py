"""Session and in-turn summarizer helpers."""

from __future__ import annotations

from typing import Callable

from assistant.rlm.checkpoint import (
    CHECKPOINT_SECTIONS,
    is_empty_checkpoint,
    normalize_checkpoint,
    parse_checkpoint,
)
from assistant.rlm.prompts import get_prompt
from assistant.rlm.tokens import clip_chars, estimate_tokens


def digest_checkpoint(previous: str | None, body: list[dict[str, str]]) -> str:
    lines = [
        f"- {m['role']}: {clip_chars((m.get('content') or '').replace(chr(10), ' ').strip(), 400)}"
        for m in body
    ][-60:]
    prev = parse_checkpoint(previous) if previous else None
    tools_prev = ""
    if prev and prev.get("Tool Results") and not is_empty_checkpoint("## Tool Results\n" + prev["Tool Results"]):
        tools_prev = prev["Tool Results"]
    tools = "\n".join([x for x in [tools_prev, *lines] if x])
    merged = []
    for s in CHECKPOINT_SECTIONS:
        if s == "Tool Results":
            body_s = tools or "(none)"
        else:
            body_s = (prev.get(s) if prev else "") or "(none)"
        merged.append(f"## {s}\n{body_s}")
    return normalize_checkpoint("\n\n".join(merged))


def build_summarizer_messages(
    previous: str | None,
    body: list[dict[str, str]],
) -> list[dict[str, str]]:
    system = get_prompt("session_compact/system.md")
    if previous:
        system = f"{system}\n\nEarlier compacted summary:\n{previous}"
    msgs: list[dict[str, str]] = [{"role": "system", "content": system}]
    for m in body:
        last = msgs[-1]
        if last["role"] == m["role"] and last["role"] != "system":
            last["content"] += f"\n\n{m['content']}"
        else:
            msgs.append({"role": m["role"], "content": m["content"]})
    instruction = get_prompt("session_compact/summarizer.md")
    last = msgs[-1]
    if last["role"] == "user":
        last["content"] += f"\n\n---\n{instruction}"
    else:
        msgs.append({"role": "user", "content": instruction})
    return msgs


def summarize_in_chunks(
    *,
    complete_fn: Callable[..., str],
    previous: str | None,
    body: list[dict[str, str]],
    max_tokens: int,
    context_length: int,
) -> str:
    overhead = (
        estimate_tokens(get_prompt("session_compact/system.md"))
        + estimate_tokens(get_prompt("session_compact/summarizer.md"))
        + 256
    )
    prev = previous
    i = 0
    shrink = 1
    while i < len(body):
        room = max(
            512,
            int((context_length - max_tokens - overhead - estimate_tokens(prev or "")) / shrink),
        )
        chunk: list[dict[str, str]] = []
        used = 0
        for j in range(i, len(body)):
            m = body[j]
            cost = estimate_tokens(m.get("content") or "") + 8
            if chunk and used + cost > room:
                break
            if cost > room:
                chunk.append(
                    {
                        "role": m["role"],
                        "content": clip_chars(m.get("content") or "", room * 3),
                    }
                )
            else:
                chunk.append(m)
            used += min(cost, room)
        try:
            next_text = complete_fn(
                build_summarizer_messages(prev, chunk),
                max_tokens=max_tokens,
                temperature=0.2,
            )
            next_text = normalize_checkpoint(next_text)
        except Exception:
            if shrink < 16:
                shrink *= 2
                continue
            next_text = digest_checkpoint(prev, chunk)
        prev = digest_checkpoint(prev, chunk) if is_empty_checkpoint(next_text) else next_text
        i += len(chunk)
        shrink = 1
    return prev or normalize_checkpoint("")
