"""API auth helpers — optional API key gate."""

from __future__ import annotations

import hmac

from fastapi import Header, HTTPException, Request

from assistant.settings import get_settings


def require_api_key(
    request: Request,
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    """
    When NESTLING_API_KEY is set, require matching X-API-Key header
    (or Authorization: Bearer <key>). Health remains open.
    """
    expected = get_settings().nestling_api_key
    if not expected:
        return
    path = request.url.path
    if path.endswith("/health") or path.rstrip("/").endswith("/api/health"):
        return
    bearer = None
    auth = request.headers.get("authorization") or ""
    if auth.lower().startswith("bearer "):
        bearer = auth[7:].strip()
    provided = x_api_key or bearer
    # Constant-time compare so a wrong key cannot be recovered by timing.
    if not provided or not hmac.compare_digest(provided, expected):
        raise HTTPException(
            status_code=401,
            detail={"error": "unauthorized", "detail": "Invalid or missing API key"},
        )
