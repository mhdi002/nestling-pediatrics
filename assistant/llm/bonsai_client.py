"""OpenAI-compatible client for PrismML Bonsai-27B via llama-server."""

from __future__ import annotations

import base64
import json
import os
import urllib.error
import urllib.request
from typing import Any


def bonsai_base_url() -> str:
    return (os.environ.get("NESTLING_BONSAI_URL") or os.environ.get("BONSAI_URL") or "").rstrip("/")


def bonsai_enabled() -> bool:
    return bool(bonsai_base_url()) and os.environ.get("NESTLING_USE_BONSAI", "1") != "0"


class BonsaiClient:
    """Chat + optional vision against llama-server (/v1/chat/completions)."""

    def __init__(self, base_url: str | None = None, timeout: float = 180.0):
        self.base_url = (base_url or bonsai_base_url()).rstrip("/")
        self.timeout = timeout
        self.model = os.environ.get("NESTLING_BONSAI_MODEL", "Bonsai-27B-Q1_0")

    @property
    def ready(self) -> bool:
        if not self.base_url:
            return False
        try:
            req = urllib.request.Request(f"{self.base_url}/health", method="GET")
            with urllib.request.urlopen(req, timeout=3) as r:
                return r.status == 200
        except Exception:
            try:
                req = urllib.request.Request(f"{self.base_url}/v1/models", method="GET")
                with urllib.request.urlopen(req, timeout=3) as r:
                    return r.status == 200
            except Exception:
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
            raise RuntimeError("NESTLING_BONSAI_URL is not set")
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": 20,
            "max_tokens": max_tokens,
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
            raise RuntimeError(f"Bonsai HTTP {exc.code}: {detail[:400]}") from exc
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"Bonsai empty response: {body!r}")
        msg = choices[0].get("message") or {}
        return (msg.get("content") or "").strip()

    def answer_with_context(self, query: str, context: str, *, system: str | None = None) -> str:
        sys = system or (
            "You are Nestling, a warm pediatric parent assistant. "
            "Answer ONLY using the provided care notes. Be concise, conversational, and clear. "
            "Never invent drug doses. Always remind parents this is not a diagnosis and to see a clinician when worried."
        )
        user = (
            f"Parent question:\n{query}\n\n"
            f"Care notes (use these; do not invent beyond them):\n{context}\n\n"
            "Reply in plain parent language (2–5 short paragraphs)."
        )
        return self.chat(
            [
                {"role": "system", "content": sys},
                {"role": "user", "content": user},
            ],
            max_tokens=640,
        )

    def analyze_image(
        self,
        image_bytes: bytes,
        *,
        mime: str = "image/png",
        prompt: str = "",
        context: str = "",
    ) -> str:
        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{mime};base64,{b64}"
        sys = (
            "You are Nestling, a pediatric parent assistant with vision. "
            "Describe what you see carefully. Suggest common educational possibilities "
            "(e.g. viral rash patterns such as hand-foot-mouth) ONLY as possibilities, never a diagnosis. "
            "List red flags for urgent care. Encourage photographing for the clinician. "
            "Do not prescribe medications or doses."
        )
        user_text = prompt.strip() or (
            "A parent sent this photo of their child. Describe the skin findings and "
            "give calm parent guidance. Remind them to contact a pediatrician."
        )
        if context:
            user_text += f"\n\nRelevant care notes:\n{context[:2500]}"
        content = [
            {"type": "text", "text": user_text},
            {"type": "image_url", "image_url": {"url": data_url}},
        ]
        return self.chat(
            [
                {"role": "system", "content": sys},
                {"role": "user", "content": content},
            ],
            max_tokens=700,
        )


_CLIENT: BonsaiClient | None = None


def get_bonsai() -> BonsaiClient:
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = BonsaiClient()
    return _CLIENT
