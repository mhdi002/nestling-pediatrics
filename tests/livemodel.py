"""Driving a REAL model from the test suite.

Every memory test until now either passed ``use_llm=False`` or stubbed the
client with something that ignores its input, so the LLM half of
consolidation and graph extraction -- the half that runs in production -- had
never executed once. The parsers that read the model's reply were written
against imagined output.

This module lets a test talk to whatever OpenAI-compatible server is actually
running, and skip cleanly when there is none, so CI without a sidecar stays
green.

WHY THE HARNESS REWRITES THE PROMPT
Qwen3.x reasons before answering. The deployed vLLM is told not to, via
``chat_template_kwargs {"enable_thinking": false}``, and obeys. Ollama's
OpenAI shim silently ignores that field -- measured on ollama 0.32.14 against
qwen3-vl:4b, where the following all reasoned and returned EMPTY content:

    plain                      2413 chars of reasoning, finish_reason=length
    {"think": false}           2420
    {"reasoning_effort":"none"} 2551
    {"reasoning_effort":"low"}  2504
    chat_template_kwargs       2473
    "/no_think" on the user     2467
    "/no_think" on BOTH system
      and user                 1551 chars, finish_reason=stop, answered

Only the last one leaves room for an answer, so that is what the harness
sends, together with a floor under max_tokens. Both are properties of *this
server*, not of the product: nothing here changes what Nestling sends to its
own sidecar. Everything downstream of the HTTP call -- the JSON parsers,
consolidation, graph ingest, the grounded prompt -- runs exactly as shipped.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request

import pytest

# Local Ollama by default; override to point the live tests somewhere else.
DEFAULT_URL = "http://localhost:11434"

# Preference order for the text model. The deployed model is Qwen3.5-4B, so a
# local qwen3.5 is the closest thing to what ships; the vision build is the
# fallback and is what the photo tests need anyway.
TEXT_PREFERENCES = ("qwen3.5:4b", "qwen3.5-4b", "qwen3.5", "qwen3-vl:4b", "qwen3")

# A served model whose name says nothing about vision is not worth probing
# with an image; the capability probe in QwenClient decides for the rest.
VISION_PREFERENCES = ("qwen3-vl", "vl", "vision")

# Reasoning burns the budget before the answer starts. Measured above: the
# smallest prompt needed ~1550 characters of reasoning (roughly 400 tokens)
# before it produced anything, and the consolidation prompt needs several
# times that. 3000 leaves room for the reasoning plus a JSON array; below
# ~1500 the reply came back empty every time.
TOKEN_FLOOR = 3000

# A local 4B on CPU/small GPU took 90-280 seconds for the prompts here, so the
# product's 60s ceiling would time out every call. Live tests only.
TIMEOUT_SECONDS = 900.0

_NO_THINK = "/no_think"


def base_url() -> str:
    return (os.environ.get("NESTLING_TEST_LLM_URL") or DEFAULT_URL).rstrip("/")


def served_models(url: str | None = None) -> list[str]:
    """Model ids the server offers, or [] when it is not there."""
    url = (url or base_url()).rstrip("/")
    try:
        with urllib.request.urlopen(f"{url}/v1/models", timeout=3) as r:
            body = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, OSError, ValueError):
        return []
    return [str(m.get("id") or "") for m in (body.get("data") or []) if m.get("id")]


def _pick(models: list[str], preferences: tuple[str, ...]) -> str | None:
    for want in preferences:
        for name in models:
            if name.lower().startswith(want.lower()):
                return name
    return None


def pick_text_model(models: list[str]) -> str | None:
    return _pick(models, TEXT_PREFERENCES)


def pick_vision_model(models: list[str]) -> str | None:
    for want in VISION_PREFERENCES:
        for name in models:
            if want.lower() in name.lower():
                return name
    return None


def live_model(monkeypatch, *, vision: bool = False) -> str:
    """Point Nestling at the local server, or skip the test.

    Returns the model id in use. The caller gets a fully configured settings
    object and a QwenClient whose transport is real; only the prompt shaping
    described in the module docstring is added.
    """
    url = base_url()
    models = served_models(url)
    if not models:
        pytest.skip(f"no OpenAI-compatible model server at {url}")
    text = pick_text_model(models)
    if not text:
        pytest.skip(f"no Qwen-family text model served at {url} (have {models})")
    chosen = pick_vision_model(models) if vision else text
    if vision and not chosen:
        pytest.skip(f"no vision-capable model served at {url} (have {models})")

    monkeypatch.setenv("NESTLING_LLM_URL", url)
    monkeypatch.setenv("NESTLING_VISION_LLM_URL", url)
    monkeypatch.setenv("NESTLING_USE_LLM", "1")
    monkeypatch.setenv("NESTLING_LLM_MODEL", text)
    monkeypatch.setenv("NESTLING_VISION_MODEL", chosen if vision else text)
    monkeypatch.setenv("NESTLING_LLM_TIMEOUT", str(TIMEOUT_SECONDS))

    from assistant.llm import qwen_client
    from assistant.settings import reset_settings

    reset_settings()
    qwen_client.reset_qwen()
    _patch_for_reasoning_server(monkeypatch, qwen_client)
    return chosen


def _patch_for_reasoning_server(monkeypatch, qwen_client) -> None:
    """Add the /no_think markers and the token floor, nothing else."""
    real_chat = qwen_client.QwenClient.chat
    real_image = qwen_client.QwenClient.analyze_image

    def chat(self, messages, *, temperature=None, top_p=None, max_tokens=None):
        return real_chat(
            self,
            [_mark(m) for m in messages],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max(TOKEN_FLOOR, max_tokens or 0),
        )

    def analyze_image(self, *, image_bytes, prompt, mime="image/png", max_tokens=None):
        return real_image(
            self,
            image_bytes=image_bytes,
            prompt=f"{_NO_THINK}\n{prompt}",
            mime=mime,
            max_tokens=max(TOKEN_FLOOR, max_tokens or 0),
        )

    monkeypatch.setattr(qwen_client.QwenClient, "chat", chat)
    monkeypatch.setattr(qwen_client.QwenClient, "analyze_image", analyze_image)


def _mark(message: dict) -> dict:
    """Prefix a text message with /no_think, leaving multimodal parts alone."""
    content = message.get("content")
    if not isinstance(content, str) or content.startswith(_NO_THINK):
        return message
    return {**message, "content": f"{_NO_THINK}\n{content}"}


def reset_after(monkeypatch) -> None:  # pragma: no cover - teardown helper
    from assistant.llm import qwen_client
    from assistant.settings import reset_settings

    reset_settings()
    qwen_client.reset_qwen()
