"""OpenAI-compatible streaming client for the RLM engine."""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from typing import Any, Callable, Iterator

from assistant.settings import get_settings

log = logging.getLogger(__name__)


class ContextOverflowError(Exception):
    pass


class ModelWindowError(Exception):
    def __init__(self, message: str, context_length: int | None = None) -> None:
        super().__init__(message)
        self.context_length = context_length


def _url() -> str:
    s = get_settings()
    return (s.nestling_llm_url or "").rstrip("/")


def _model() -> str:
    return get_settings().nestling_llm_model


def stream_chat(
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float = 0.3,
    enable_thinking: bool | None = None,
    signal_aborted: Callable[[], bool] | None = None,
) -> Iterator[dict[str, Any]]:
    """Yield chunks: {type: text|thinking|usage, ...}."""
    s = get_settings()
    base = _url()
    if not base:
        raise RuntimeError("NESTLING_LLM_URL is not set")
    payload: dict[str, Any] = {
        "model": _model(),
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    # Chat template kwargs (disable thinking for Qwen-family when configured).
    raw = (s.nestling_llm_chat_template_kwargs or "").strip()
    if raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict) and parsed:
                if enable_thinking is False:
                    parsed = {**parsed, "enable_thinking": False}
                payload["chat_template_kwargs"] = parsed
        except ValueError:
            pass
    elif enable_thinking is False:
        payload["chat_template_kwargs"] = {"enable_thinking": False}

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{base}/v1/chat/completions",
        data=data,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {s.nestling_llm_api_key or 'EMPTY'}",
        },
        method="POST",
    )
    stall = float(s.rlm_model_stall_sec)
    try:
        with urllib.request.urlopen(req, timeout=stall) as resp:
            for raw_line in resp:
                if signal_aborted and signal_aborted():
                    break
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                if line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    continue
                body = line[5:].strip()
                if body == "[DONE]":
                    break
                try:
                    obj = json.loads(body)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    u = obj["usage"]
                    yield {
                        "type": "usage",
                        "inputTokens": int(u.get("prompt_tokens") or 0),
                        "outputTokens": int(u.get("completion_tokens") or 0),
                    }
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = (choices[0] or {}).get("delta") or {}
                reasoning = delta.get("reasoning") or delta.get("reasoning_content")
                if reasoning:
                    yield {"type": "thinking", "text": str(reasoning)}
                content = delta.get("content")
                if content:
                    yield {"type": "text", "text": str(content)}
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:800]
        low = detail.lower()
        if exc.code == 400 and ("context" in low or "token" in low):
            raise ContextOverflowError(detail) from exc
        raise RuntimeError(f"LLM HTTP {exc.code}: {detail}") from exc


def complete(
    messages: list[dict[str, str]],
    *,
    max_tokens: int,
    temperature: float = 0.2,
) -> str:
    text = ""
    for c in stream_chat(
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        enable_thinking=False,
    ):
        if c.get("type") == "text":
            text += c.get("text") or ""
    return text
