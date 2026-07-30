"""OpenAI-compatible client used by Nestling text + vision paths."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Any

import base64

from assistant.settings import get_settings


def _strip_reasoning(text: str) -> str:
    """Qwen3.x often emits hidden/visible chain-of-thought — keep the parent reply only."""
    import re

    t = (text or "").strip()
    if not t:
        return ""
    # XML-style think blocks
    t = re.sub(r"<think>[\s\S]*?</think>", "", t, flags=re.I)
    t = re.sub(r"<thinking>[\s\S]*?</thinking>", "", t, flags=re.I)
    # Visible "Thinking Process:" dumps before the real answer
    if re.search(r"(?i)^\s*thinking process\s*:", t):
        # Prefer text after a clear answer marker
        for pat in (
            r"(?is)\*\*final answer:?\*\*\s*",
            r"(?is)\bfinal answer:?\s*",
            r"(?is)\banswer:?\s*",
            r"(?is)\breply:?\s*",
        ):
            m = re.search(pat, t)
            if m:
                t = t[m.end() :].strip()
                break
        else:
            # Drop leading bullet analysis until a normal paragraph starts
            parts = re.split(r"\n\s*\n", t)
            kept = []
            for p in parts:
                if re.search(r"(?i)thinking process|analyze the request|constraints:|input data", p):
                    continue
                kept.append(p)
            t = "\n\n".join(kept).strip() or t
    return t.strip()


def llm_base_url() -> str:
    return (os.environ.get("NESTLING_LLM_URL") or os.environ.get("LLM_URL") or "").rstrip("/")


def vision_base_url() -> str:
    return (
        os.environ.get("NESTLING_VISION_LLM_URL")
        or os.environ.get("NESTLING_VISION_URL")
        or os.environ.get("VISION_LLM_URL")
        or llm_base_url()
    ).rstrip("/")


def llm_model_path() -> str:
    return (
        os.environ.get("NESTLING_LLM_MODEL_PATH")
        or os.environ.get("NESTLING_LLM_HF_PATH")
        or os.environ.get("NESTLING_LLM_GGUF")
        or os.environ.get("NESTLING_LLM_GGUF_PATH", "")
    )


def llm_enabled() -> bool:
    if os.environ.get("NESTLING_USE_LLM", "1") == "0":
        return False
    return bool(llm_base_url())


class QwenClient:
    """Chat completions against an OpenAI-compatible server."""

    def __init__(self, base_url: str | None = None, timeout: float = 180.0):
        settings = get_settings()
        self.base_url = (base_url or llm_base_url()).rstrip("/")
        self.vision_url = vision_base_url()
        self.timeout = timeout
        self.model = os.environ.get("NESTLING_LLM_MODEL", settings.nestling_llm_model)
        self.vision_model = os.environ.get("NESTLING_VISION_MODEL", settings.nestling_vision_model)
        self.model_path = llm_model_path()

    @property
    def ready(self) -> bool:
        if not self.base_url:
            return False
        for path in ("/health", "/v1/models"):
            try:
                req = urllib.request.Request(f"{self.base_url}{path}", method="GET")
                with urllib.request.urlopen(req, timeout=1.5) as r:
                    if r.status == 200:
                        return True
            except Exception:
                continue
        return False

    @property
    def vision_ready(self) -> bool:
        if not self.vision_url:
            return False
        for path in ("/health", "/v1/models"):
            try:
                req = urllib.request.Request(f"{self.vision_url}{path}", method="GET")
                with urllib.request.urlopen(req, timeout=1.5) as r:
                    if r.status == 200:
                        return True
            except Exception:
                continue
        return False

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        temperature: float = 0.7,
        top_p: float = 0.95,
        max_tokens: int = 512,
    ) -> str:
        if not self.base_url:
            raise RuntimeError("NESTLING_LLM_URL is not set")
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
            # Qwen3.x: suppress visible chain-of-thought when supported by the server
            "chat_template_kwargs": {"enable_thinking": False},
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                body = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"LLM HTTP {exc.code}: {detail[:400]}") from exc
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM empty response: {body!r}")
        msg = choices[0].get("message") or {}
        return _strip_reasoning((msg.get("content") or "").strip())

    def analyze_image(
        self,
        *,
        image_bytes: bytes,
        prompt: str,
        mime: str = "image/png",
        max_tokens: int = 700,
    ) -> str:
        if not self.vision_url:
            raise RuntimeError("NESTLING_VISION_LLM_URL is not set")
        data_url = (
            f"data:{mime};base64," + base64.b64encode(image_bytes).decode("ascii")
        )
        payload = {
            "model": self.vision_model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are Nestling, a pediatric parent assistant with vision. "
                        "Educational only, no diagnosis."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": data_url}},
                    ],
                },
            ],
            "temperature": 0.2,
            "top_p": 0.9,
            "max_tokens": max_tokens,
        }
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            f"{self.vision_url}/v1/chat/completions",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                body = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Vision HTTP {exc.code}: {detail[:400]}") from exc
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"Vision empty response: {body!r}")
        msg = choices[0].get("message") or {}
        return _strip_reasoning((msg.get("content") or "").strip())

    def answer_with_context(self, query: str, context: str, *, system: str | None = None) -> str:
        sys = system or (
            "You are Nestling, a warm pediatric parent assistant. "
            "Answer ONLY using the provided care notes. Be concise, conversational, and clear. "
            "Answer ONLY the current parent question's topic — do not rehash earlier topics "
            "(e.g. feeding/meals/milk) unless the parent asks about them now. "
            "Paraphrase in your own words — never paste the care notes verbatim. "
            "Vary wording naturally for each question. "
            "If the parent question states a Known chronological age in months, use ONLY that age "
            "(e.g. say ~13 months / toddler). Never invent a different age from care-note section "
            "titles like 'Feeding 7–9 months'. "
            "Do NOT include chain-of-thought, analysis steps, or 'Thinking Process' — reply to the parent only. "
            "Never invent drug doses. Always remind parents this is not a diagnosis and to see a clinician when worried."
        )
        # Keep prompts short for small-GPU max_model_len budgets.
        ctx = (context or "").strip()
        if len(ctx) > 1800:
            ctx = ctx[:1800] + "…"
        q = (query or "").strip()
        if len(q) > 400:
            q = q[:400] + "…"
        user = (
            f"Parent question:\n{q}\n\n"
            f"Care notes (ground your reply in these; do not invent beyond them):\n{ctx}\n\n"
            "Reply in plain parent language (2–4 short paragraphs). "
            "Stay on the current question only; do not add unrelated prior topics. "
            "Do not copy headings or dump the notes; speak like a helpful chat."
        )
        return self.chat(
            [
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
            temperature=0.75,
            max_tokens=320,
        )


_CLIENT: QwenClient | None = None


def get_qwen() -> QwenClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = QwenClient()
    return _CLIENT
