#!/usr/bin/env python3
"""OpenAI-compatible chat server for Bonsai GGUF (+ optional mmproj) via llama-cpp-python."""
from __future__ import annotations

import base64
import os
import re
from typing import Any

from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

MODELS_DIR = os.environ.get("MODELS_DIR", "/models")
GGUF = os.environ.get("BONSAI_GGUF", "Bonsai-27B-Q1_0.gguf")
MMPROJ = os.environ.get("BONSAI_MMPROJ", "Bonsai-27B-mmproj-Q8_0.gguf")
N_CTX = int(os.environ.get("BONSAI_CTX", "4096"))
N_GPU = int(os.environ.get("BONSAI_NGL", "0"))
MODEL_ID = os.environ.get("NESTLING_BONSAI_MODEL", "Bonsai-27B-Q1_0")

app = FastAPI(title="Nestling Bonsai")
_llm = None
_load_error: str | None = None


def _data_url_to_bytes(url: str) -> tuple[bytes, str] | None:
    m = re.match(r"^data:([^;]+);base64,(.+)$", url, re.S)
    if not m:
        return None
    return base64.b64decode(m.group(2)), m.group(1)


def get_llm():
    global _llm, _load_error
    if _llm is not None:
        return _llm
    if _load_error:
        raise RuntimeError(_load_error)
    from llama_cpp import Llama

    path = os.path.join(MODELS_DIR, GGUF)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Missing model file: {path}")
    kwargs: dict[str, Any] = {
        "model_path": path,
        "n_ctx": N_CTX,
        "n_gpu_layers": N_GPU,
        "verbose": False,
    }
    mmproj = os.path.join(MODELS_DIR, MMPROJ)
    if os.environ.get("BONSAI_ENABLE_VISION", "1") == "1" and os.path.isfile(mmproj):
        kwargs["chat_handler"] = None
        try:
            from llama_cpp.llama_chat_format import Llava15ChatHandler

            kwargs["chat_handler"] = Llava15ChatHandler(clip_model_path=mmproj)
        except Exception:
            # Fall back to plain text if vision handler unavailable
            kwargs.pop("chat_handler", None)
    _llm = Llama(**kwargs)
    return _llm


class ChatMessage(BaseModel):
    role: str
    content: Any


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.95
    max_tokens: int = 512


@app.get("/health")
def health():
    ready = False
    detail = None
    try:
        get_llm()
        ready = True
    except Exception as exc:
        detail = str(exc)
    return {"status": "ok" if ready else "loading", "ready": ready, "model": MODEL_ID, "detail": detail}


@app.get("/v1/models")
def models():
    return {"data": [{"id": MODEL_ID, "object": "model"}]}


def _flatten_content(content: Any) -> tuple[str, list[bytes]]:
    if isinstance(content, str):
        return content, []
    if not isinstance(content, list):
        return str(content), []
    texts: list[str] = []
    images: list[bytes] = []
    for part in content:
        if not isinstance(part, dict):
            continue
        if part.get("type") == "text":
            texts.append(str(part.get("text") or ""))
        elif part.get("type") == "image_url":
            url = (part.get("image_url") or {}).get("url") or ""
            parsed = _data_url_to_bytes(url)
            if parsed:
                images.append(parsed[0])
    return "\n".join(texts).strip(), images


@app.post("/v1/chat/completions")
def chat(req: ChatRequest):
    try:
        llm = get_llm()
    except Exception as exc:
        return JSONResponse({"error": str(exc)}, status_code=503)

    messages = []
    for m in req.messages:
        text, images = _flatten_content(m.content)
        if images:
            # llama-cpp multimodal: pass image as content list when chat_handler present
            content: Any = [{"type": "text", "text": text}]
            for img in images:
                b64 = base64.b64encode(img).decode("ascii")
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{b64}"},
                    }
                )
            messages.append({"role": m.role, "content": content})
        else:
            messages.append({"role": m.role, "content": text})

    out = llm.create_chat_completion(
        messages=messages,
        temperature=req.temperature,
        top_p=req.top_p,
        max_tokens=req.max_tokens,
    )
    return out


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("BONSAI_PORT", "8080")))
