"""Nestling API routes under /api (mounted by app.main)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Iterator

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from app.api.auth import (
    authenticate,
    current_user,
    hash_password,
    login_enabled,
    make_token,
    require_api_key,
)
from app.services import get_services
from assistant.config import OVERLAY_DIR, UPLOAD_DIR
from assistant.refdata import clinical_bounds
from assistant.settings import get_settings
from assistant.tools.clinical import (
    dispatch_tool,
    growth_percentile_curves,
    list_asq_questions,
    list_mchat_questions,
)

log = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_api_key)])


def _err(status: int, error: str, detail: str | None = None) -> HTTPException:
    return HTTPException(status_code=status, detail={"error": error, "detail": detail or error})


# The chat admission gate is built at startup (app.main lifespan) once settings
# and the sidecar sizing are known, and injected here. Until then it is a
# disabled no-op, so tests and the app-only path behave exactly as before.
from app.concurrency import ChatGate, GateBusy  # noqa: E402

_CHAT_GATE = ChatGate(0, 0, 0.0)


def set_chat_gate(gate: ChatGate) -> None:
    global _CHAT_GATE
    _CHAT_GATE = gate


def get_chat_gate() -> ChatGate:
    return _CHAT_GATE


def _busy_error(exc: GateBusy) -> HTTPException:
    """503 with a Retry-After hint, not a 500 or a silent timeout.

    Shedding load honestly is the whole point: a parent who is told to retry in
    a few seconds is better served than one whose request sits in a queue until
    it 504s, and the difference is what keeps the GPU working on turns someone
    is still waiting for.
    """
    return HTTPException(
        status_code=503,
        detail={
            "error": "busy",
            "detail": "The assistant is at capacity right now. Please retry shortly.",
            "retry_after": round(exc.retry_after, 1),
        },
        headers={"Retry-After": str(max(1, int(exc.retry_after)))},
    )


def _require_owned_child(svc, child_id: str | None, owner_user_id: str | None) -> None:
    """
    Guard a child_id passed in a request body/path against the signed-in account.

    A signed-in account may only reach its own children; a child_id it does not
    own is answered with the same 404 as a missing one, so ids cannot be probed
    for existence and one family can never read or mutate another's record.

    API-key / unauthenticated callers (owner_user_id is None) keep the historical
    unscoped access — the same convention the read routes already follow.
    """
    if owner_user_id and child_id and svc.db.get_child(child_id, owner_user_id=owner_user_id) is None:
        raise _err(404, "not_found", f"Unknown child_id: {child_id}")


# --- request bodies ---


_BOUNDS = clinical_bounds()
GA_MIN_WEEKS = float(_BOUNDS.get("intergrowth_weeks_min", 27))
GA_MAX_WEEKS = float(_BOUNDS.get("intergrowth_weeks_max", 64))
PRETERM_GA_WEEKS = float(_BOUNDS.get("preterm_ga_threshold_weeks", 37))
WHO_AGE_MONTHS_MAX = float(_BOUNDS.get("who_age_months_max", 24))
ASQ_AGE_MONTHS_MAX = int(_BOUNDS.get("asq_age_months_max", 72))
# Generous outer bound for a free-text `weeks` field; the clinical tools apply the
# real per-chart limits and return a parent-readable error.
MAX_WEEKS_INPUT = 1000.0
MAX_AGE_MONTHS_INPUT = 1200.0
MAX_VALUE_INPUT = 1000.0
NAME_MAX_CHARS = 120
NOTES_MAX_CHARS = 4000
MESSAGE_MAX_CHARS = 8000
UI_LANGS = {"fa", "en"}
# Characters that flush an SSE token chunk early so the UI streams word-by-word.
STREAM_BREAK_CHARS = " \n.,!?;:"


class ChildCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=NAME_MAX_CHARS)
    sex: str = Field(..., min_length=1, max_length=NAME_MAX_CHARS)
    date_of_birth: str | None = None
    gestational_age_weeks: float | None = Field(
        None, ge=GA_MIN_WEEKS - 10, le=GA_MAX_WEEKS
    )
    notes: str = Field("", max_length=NOTES_MAX_CHARS)


class SessionCreate(BaseModel):
    child_id: str | None = None
    title: str | None = Field(None, max_length=NAME_MAX_CHARS)


class ChatBody(BaseModel):
    # Optional: UI may chat before creating a session — we auto-create.
    session_id: str | None = None
    message: str = Field(..., min_length=1, max_length=MESSAGE_MAX_CHARS)
    child_id: str | None = None
    ui_lang: str | None = None  # 'fa' | 'en'


class GrowthBody(BaseModel):
    sex: str = Field(..., min_length=1)
    measure: str = Field(..., min_length=1)
    weeks: float | None = Field(None, ge=0, le=MAX_WEEKS_INPUT)
    value: float = Field(..., gt=0, le=MAX_VALUE_INPUT)
    child_id: str | None = None
    age_months: float | None = Field(None, ge=0, le=MAX_AGE_MONTHS_INPUT)
    gestational_age_weeks: float | None = Field(None, ge=GA_MIN_WEEKS - 10, le=GA_MAX_WEEKS)


class AsqScoreBody(BaseModel):
    domain_answers: dict[str, list[str]]
    age_months: int = Field(..., ge=0, le=ASQ_AGE_MONTHS_MAX)
    child_id: str | None = None


class MchatScoreBody(BaseModel):
    answers: dict[str, str]
    child_id: str | None = None


class LoginBody(BaseModel):
    username: str = Field(min_length=1, max_length=128)
    password: str = Field(min_length=1, max_length=512)


# Brute-force throttle for the one unauthenticated write endpoint. In-memory
# and per-process, which is sufficient because the app runs a single worker by
# default; a multi-worker or multi-replica deployment needs the shared limiter
# in nginx (see docker/nginx/nestling.conf.template) as the real control.
_LOGIN_ATTEMPTS: dict[str, list[float]] = {}
_LOGIN_MAX_ATTEMPTS = 10
_LOGIN_WINDOW_SECONDS = 300.0


def _login_rate_limited(client_ip: str) -> bool:
    import time as _time

    now = _time.monotonic()
    hits = [t for t in _LOGIN_ATTEMPTS.get(client_ip, []) if now - t < _LOGIN_WINDOW_SECONDS]
    # Bound the dict so a spray of spoofed IPs cannot grow it without limit.
    if len(_LOGIN_ATTEMPTS) > 2048:
        _LOGIN_ATTEMPTS.clear()
    hits.append(now)
    _LOGIN_ATTEMPTS[client_ip] = hits
    return len(hits) > _LOGIN_MAX_ATTEMPTS


@router.get("/auth/config")
def auth_config():
    """Tells the UI whether to show a sign-in form. Never exposes credentials."""
    return {"login_required": login_enabled()}


@router.post("/auth/register")
def register(body: LoginBody, request: Request):
    """Create an account. Each account only ever sees the children it creates."""
    client_ip = (request.client.host if request.client else None) or "unknown"
    if _login_rate_limited(client_ip):
        raise _err(429, "too_many_attempts", "Too many attempts. Try again in a few minutes.")
    username = body.username.strip()
    if len(username) < 3:
        raise _err(400, "invalid_username", "Username must be at least 3 characters")
    if len(body.password) < 8:
        raise _err(400, "weak_password", "Password must be at least 8 characters")
    svc = get_services()
    user_id = svc.db.create_user(username, hash_password(body.password))
    if user_id is None:
        raise _err(409, "username_taken", "That username is already registered")
    settings = get_settings()
    return {
        "token": make_token(user_id),
        "expires_in": settings.nestling_session_ttl_hours * 3600,
        "username": username,
    }


@router.post("/auth/login")
def login(body: LoginBody, request: Request):
    """Exchange username/password for a bearer token used by the web UI."""
    if not login_enabled():
        raise _err(400, "login_disabled", "This deployment has no login configured")
    client_ip = (request.client.host if request.client else None) or "unknown"
    if _login_rate_limited(client_ip):
        raise _err(429, "too_many_attempts", "Too many sign-in attempts. Try again in a few minutes.")
    user_id = authenticate(body.username, body.password)
    if not user_id:
        # Deliberately identical message for bad user vs bad password.
        raise _err(401, "invalid_credentials", "Incorrect username or password")
    settings = get_settings()
    return {
        # The token subject is the user id, so data scoping follows the account
        # rather than a display name that could later change.
        "token": make_token(user_id),
        "expires_in": settings.nestling_session_ttl_hours * 3600,
        "username": body.username,
    }


@router.get("/ready")
def ready():
    """Fast readiness for load balancers — no LLM probe, no external I/O."""
    return {"status": "ready", "service": "nestling"}


@router.get("/health")
def health():
    """Liveness + short LLM readiness probe (1.5s, never blocks Docker health long)."""
    llm = {"configured": False, "ready": False, "url": None}
    vision = {"configured": False, "ready": False, "url": None}
    try:
        from assistant.llm.qwen_client import (
            get_qwen,
            llm_base_url,
            llm_enabled,
            llm_model_path,
            vision_base_url,
        )

        if llm_enabled():
            client = get_qwen()
            ready = bool(client.ready)
            vision_ready = bool(client.vision_ready) if vision_base_url() else False
            llm = {
                "configured": True,
                "ready": ready,
                "probed": True,
                "url": llm_base_url(),
                "model": client.model,
                "model_path": llm_model_path() or None,
            }
            vision_url = vision_base_url()
            vision = {
                "configured": bool(vision_url),
                "ready": vision_ready,
                "probed": True,
                "url": vision_url,
                "model": client.vision_model,
                "shared_endpoint": bool(vision_url) and vision_url == llm_base_url(),
            }
    except Exception as exc:
        llm["error"] = str(exc)
        vision["error"] = str(exc)
    # Admission-gate occupancy, so an operator can see whether the app is
    # shedding load and size the fleet from real numbers rather than guesses.
    gate = get_chat_gate().stats()
    capacity = {
        "chat_gate_enabled": gate.enabled,
        "max_inflight": gate.max_inflight,
        "max_waiting": gate.max_waiting,
        "inflight": gate.inflight,
        "waiting": gate.waiting,
    }
    return {"status": "ok", "service": "nestling", "llm": llm, "vision": vision,
            "capacity": capacity}


@router.post("/children")
def create_child(body: ChildCreate, request: Request):
    svc = get_services()
    try:
        cid = svc.db.create_child(
            body.name,
            body.sex,
            owner_user_id=current_user(request),
            date_of_birth=body.date_of_birth,
            gestational_age_weeks=body.gestational_age_weeks,
            notes=body.notes,
        )
    except Exception as exc:
        raise _err(400, "create_child_failed", str(exc)) from exc
    return {"child_id": cid, "child": svc.db.get_child(cid)}


@router.get("/children")
def list_children(request: Request):
    # Scoped to the signed-in account so one family cannot enumerate another's
    # children. API-key callers (no user) still see everything, as before.
    return {"children": get_services().db.list_children(owner_user_id=current_user(request))}


@router.get("/children/{child_id}")
def get_child(child_id: str, request: Request):
    child = get_services().db.get_child(child_id, owner_user_id=current_user(request))
    if not child:
        # Same 404 whether the child is missing or owned by someone else, so
        # ids cannot be probed for existence.
        raise _err(404, "not_found", f"Unknown child_id: {child_id}")
    return {"child": child}


@router.get("/children/{child_id}/dossier")
def child_dossier(child_id: str, request: Request):
    """Full child record for UI + agent: profile, growth, screenings, chart overlays."""
    svc = get_services()
    # Scoped to the signed-in account: the dossier carries the full medical
    # record (notes, growth, screenings), so it must not be readable across
    # accounts by knowing a child_id. Same 404 whether missing or someone else's.
    child = svc.db.get_child(child_id, owner_user_id=current_user(request))
    if not child:
        raise _err(404, "not_found", f"Unknown child_id: {child_id}")
    growth = svc.db.growth_history(child_id)
    screens = svc.db.screenings(child_id)
    ga = child.get("gestational_age_weeks")
    maturity = (
        "preterm"
        if ga is not None and float(ga) < PRETERM_GA_WEEKS
        else ("term" if ga is not None else "unknown")
    )
    overlay_limit = get_settings().nestling_dossier_overlay_limit
    overlays = []
    try:
        newest_first = sorted(
            OVERLAY_DIR.glob(f"overlay_{child_id}_*.png"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        for p in newest_first[:overlay_limit]:
            overlays.append({"filename": p.name, "url": f"/api/overlays/{p.name}"})
    except OSError as exc:
        log.warning("Could not list overlays for child %s: %s", child_id, exc)
    return {
        "ok": True,
        "child_id": child_id,
        "profile": child,
        "maturity": maturity,
        "growth": growth,
        "screenings": screens,
        "overlays": overlays,
        "summary": (
            f"{child.get('name')} ({child.get('sex')}, GA {ga}w, {maturity}): "
            f"{len(growth)} growth, {len(screens)} screening(s)."
        ),
    }


@router.post("/sessions")
def create_session(body: SessionCreate, request: Request):
    svc = get_services()
    owner_user_id = current_user(request)
    # Every other route that accepts a child_id guards it; this one did not,
    # so an account could open a session bound to another family's child. The
    # chat and dossier paths re-check ownership, so no medical text leaked
    # through it -- but the session then carried the victim's child_id and
    # title into the attacker's own session list, and a later path that
    # trusted session.child_id would have leaked outright. Found by a
    # route-driven probe against the live server; it is the same IDOR class
    # as the four closed earlier, on a route that slipped the net.
    _require_owned_child(svc, body.child_id, owner_user_id)
    sid = svc.chat.create_session(
        child_id=body.child_id, title=body.title, owner_user_id=owner_user_id
    )
    return {"session_id": sid, "child_id": body.child_id}


@router.get("/sessions")
def list_sessions(
    request: Request,
    child_id: str | None = None,
    limit: int | None = Query(None, ge=1),
):
    # Scoped to the signed-in account: chat history is health data and must not
    # be readable across accounts.
    svc = get_services()
    return {
        "sessions": svc.chat.list_sessions(
            child_id=child_id, limit=limit, owner_user_id=current_user(request)
        )
    }


@router.delete("/sessions")
def clear_all_sessions(request: Request):
    """
    Delete every chat session for the signed-in account.

    Requires an authenticated user: without one there is no safe scope to
    delete, and wiping unscoped would take other families' conversations with
    it. Children, growth measurements and screenings are deliberately left
    alone -- this clears conversation history, not the medical record.
    """
    user_id = current_user(request)
    if not user_id:
        raise _err(
            401, "sign_in_required", "Sign in to clear your chat history"
        )
    removed = get_services().chat.delete_all_sessions(user_id)
    return {"deleted": removed}


@router.delete("/sessions/{session_id}")
def delete_session(session_id: str, request: Request):
    """Delete one conversation. Ownership is enforced in the store as well."""
    ok = get_services().chat.delete_session(session_id, owner_user_id=current_user(request))
    if not ok:
        # Same 404 whether it is missing or someone else's, so ids cannot be
        # probed for existence.
        raise _err(404, "session_not_found", session_id)
    return {"deleted": 1}


@router.get("/sessions/{session_id}")
def get_session(session_id: str, request: Request, limit: int | None = Query(None, ge=1)):
    svc = get_services()
    s = svc.chat.get_session(session_id)
    if not s:
        raise _err(404, "session_not_found", session_id)
    user_id = current_user(request)
    owner = s.get("owner_user_id")
    # Same 404 as "missing" so session ids cannot be probed for existence.
    # Strict: an unowned legacy session is not readable by an account either.
    if user_id and owner != user_id:
        raise _err(404, "session_not_found", session_id)
    # Bounded by default: a long-running session must not serialize its whole history.
    max_history = get_settings().nestling_history_response_limit
    history_limit = min(limit, max_history) if limit else max_history
    return {
        "session": s,
        "history": svc.chat.get_history(session_id, limit=history_limit),
    }


def _validate_image_bytes(raw: bytes) -> tuple[bytes, str]:
    """Decode with Pillow, strip EXIF, re-encode as PNG. Reject non-images."""
    from io import BytesIO

    from PIL import Image

    settings = get_settings()
    try:
        img = Image.open(BytesIO(raw))
        # Reject decompression bombs before load() allocates the full bitmap.
        w, h = img.size
        if w * h > settings.nestling_max_image_pixels:
            raise _err(
                400,
                "image_too_large",
                f"Image exceeds {settings.nestling_max_image_pixels} pixels ({w}x{h}).",
            )
        img.load()
        img = img.convert("RGBA") if img.mode in {"P", "RGBA"} else img.convert("RGB")
        # Drop EXIF by reconstructing
        out = BytesIO()
        fmt = "PNG"
        img.save(out, format=fmt, optimize=True)
        return out.getvalue(), "image/png"
    except HTTPException:
        raise
    except Exception as exc:
        raise _err(400, "invalid_image", f"Could not decode image: {exc}") from exc


def _run_vision_turn(
    *,
    clean: bytes,
    mime: str,
    fname: str,
    caption: str,
    session_id: str | None,
    child_id: str | None,
    ui_lang: str | None,
    owner_user_id: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Blocking half of the vision turn (SQLite + model call), run off the event loop."""
    svc = get_services()
    # A child_id from the form must belong to the caller (same rule as text chat).
    _require_owned_child(svc, child_id, owner_user_id)
    sid = session_id
    existing = svc.chat.get_session(sid) if sid else None
    # Never append a photo turn to another account's session (see _run_chat_turn).
    if existing and owner_user_id and existing.get("owner_user_id") != owner_user_id:
        existing = None
        sid = None
    if not sid or not existing:
        # Must carry the owner: an unowned session is invisible to the account
        # that created it under the per-user scoping.
        sid = svc.chat.create_session(child_id=child_id, owner_user_id=owner_user_id)
    user_msg = f"[photo:{fname}] {caption}" if caption else f"[photo:{fname}]"
    svc.chat.add_message(sid, "user", user_msg)
    out = svc.assistant.analyze_parent_photo(clean, mime=mime, prompt=caption, ui_lang=ui_lang)
    svc.chat.add_message(sid, "assistant", out.get("reply") or "")
    return sid, out


def _gated_vision_turn(**kwargs):
    """The vision turn under the admission gate.

    Wrapped so both the wait for a slot and the turn itself run in the worker
    thread `run_in_threadpool` gives it -- entering the gate from the async
    caller would block the event loop on the semaphore.
    """
    with _CHAT_GATE.admit():
        return _run_vision_turn(**kwargs)


@router.post("/chat/vision")
async def chat_vision(
    request: Request,
    message: str = Form(""),
    session_id: str | None = Form(None),
    child_id: str | None = Form(None),
    ui_lang: str | None = Form(None),
    image: UploadFile = File(...),
):
    """Parent photo + optional caption -> vision model + medically grounded RAG response."""
    settings = get_settings()
    allowed = settings.allowed_upload_types
    content_type = (image.content_type or "").split(";")[0].strip().lower()
    if allowed and content_type not in allowed:
        # 400 (not 415) to keep the existing error contract the SPA handles.
        raise _err(
            400,
            "unsupported_media_type",
            f"Allowed image types: {', '.join(sorted(allowed))}. Got {content_type or 'none'}.",
        )
    max_bytes = settings.nestling_max_upload_bytes
    # Read one byte past the cap so an oversized upload is rejected without
    # buffering the whole body in memory.
    raw = await image.read(max_bytes + 1)
    if not raw:
        raise _err(400, "empty_image", "No image bytes received.")
    if len(raw) > max_bytes:
        raise _err(400, "image_too_large", f"Max image size is {max_bytes} bytes.")

    clean, mime = await run_in_threadpool(_validate_image_bytes, raw)
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    from uuid import uuid4

    fname = f"{uuid4().hex}.png"
    await run_in_threadpool((UPLOAD_DIR / fname).write_bytes, clean)

    caption = (message or "").strip()
    try:
        sid, out = await run_in_threadpool(
            _gated_vision_turn,
            clean=clean,
            mime=mime,
            fname=fname,
            caption=caption,
            session_id=session_id,
            child_id=child_id,
            ui_lang=ui_lang,
            owner_user_id=current_user(request),
        )
    except GateBusy as exc:
        raise _busy_error(exc) from None
    return {
        "session_id": sid,
        "child_id": child_id,
        "reply": out.get("reply"),
        "reply_lang": out.get("reply_lang"),
        "intents": ["vision", "medical"],
        "vision": {
            "mode": out.get("mode"),
            "model": out.get("model"),
            "upload": fname,
            "disclaimer": out.get("disclaimer"),
        },
        "medical_rag": out.get("medical_rag"),
        "tool_results": [],
    }


def _normalized_ui_lang(ui_lang: str | None) -> str | None:
    lang = (ui_lang or "").strip().lower()
    return lang if lang in UI_LANGS else None


def _run_chat_turn(body: ChatBody, owner_user_id: str | None = None) -> dict[str, Any]:
    """Full chat turn. Sync on purpose — FastAPI runs `def` routes in a worker thread."""
    svc = get_services()
    # A child_id supplied by the caller must belong to the caller, or the turn
    # would read another family's child into slots and reflect it in the reply.
    _require_owned_child(svc, body.child_id, owner_user_id)
    sid = body.session_id
    existing = svc.chat.get_session(sid) if sid else None
    # A session id owned by another account must never be reused: doing so would
    # append this turn to their history and echo their child's slots back. Treat
    # a non-owned (or unowned legacy) session as absent and open a fresh one that
    # belongs to the caller. Unauthenticated / API-key callers (no owner) keep
    # the historical behaviour of reusing whatever id they pass.
    if existing and owner_user_id and existing.get("owner_user_id") != owner_user_id:
        existing = None
        sid = None
    if not sid or not existing:
        # Auto-create when UI has no session yet, or stale localStorage id after DB reset.
        sid = svc.chat.create_session(child_id=body.child_id, owner_user_id=owner_user_id)
    out = svc.assistant.chat(
        sid,
        body.message,
        child_id=body.child_id,
        ui_lang=_normalized_ui_lang(body.ui_lang),
        owner_user_id=owner_user_id,
    )
    # Auto-title from first user message
    s = svc.chat.get_session(sid) or {}
    if not (s.get("title") or "").strip():
        svc.chat.set_title(
            sid, (body.message or "").strip()[: get_settings().nestling_session_title_chars]
        )
    return out


@router.post("/chat")
def chat(body: ChatBody, request: Request):
    try:
        with _CHAT_GATE.admit():
            return _run_chat_turn(body, current_user(request))
    except GateBusy as exc:
        # At capacity: shed this turn fast with a Retry-After rather than let
        # it queue behind the GPU until it times out.
        raise _busy_error(exc) from None
    except HTTPException:
        # Deliberate 4xx (e.g. an ownership 404) must reach the client as-is,
        # not be masked as a generic 500.
        raise
    except Exception as exc:
        log.exception("Chat turn failed")
        raise _err(500, "chat_failed", str(exc)) from exc


@router.post("/chat/stream")
def chat_stream(body: ChatBody, request: Request):
    """
    SSE stream: runs the full chat turn, then streams the reply text in chunks,
    then emits a final `result` event with the full JSON payload.
    """
    chunk_chars = get_settings().nestling_stream_chunk_chars
    owner_user_id = current_user(request)
    # Reject a child_id the caller does not own before the stream opens, so it
    # surfaces as a 404 rather than mid-stream (the reuse of a non-owned session
    # id is handled inside _run_chat_turn).
    _require_owned_child(get_services(), body.child_id, owner_user_id)

    # Admission and the turn's compute happen before the stream opens, so a
    # capacity rejection is a clean 503 rather than an error event mid-stream,
    # and the slot is held only for the work -- not for however long the client
    # takes to read the already-computed reply.
    try:
        with _CHAT_GATE.admit():
            out = _run_chat_turn(body, owner_user_id)
    except GateBusy as exc:
        raise _busy_error(exc) from None

    def gen() -> Iterator[str]:
        try:
            reply = out.get("reply") or ""
            # Stream reply in word-ish chunks for UX (tools already finished)
            buf = ""
            for ch in reply:
                buf += ch
                if ch in STREAM_BREAK_CHARS or len(buf) >= chunk_chars:
                    yield f"event: token\ndata: {json.dumps({'text': buf}, ensure_ascii=False)}\n\n"
                    buf = ""
            if buf:
                yield f"event: token\ndata: {json.dumps({'text': buf}, ensure_ascii=False)}\n\n"
            yield f"event: result\ndata: {json.dumps(out, ensure_ascii=False, default=str)}\n\n"
            yield "event: done\ndata: {}\n\n"
        except Exception as exc:
            log.exception("Streaming chat turn failed")
            yield f"event: error\ndata: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


@router.get("/growth/curves")
def growth_curves(
    sex: str = "male",
    measure: str = "weight",
    chart_standard: str | None = None,
    gestational_age_weeks: float | None = None,
    age_max: float | None = None,
):
    """JSON percentile curves (P3/10/50/90/97) for client SVG charts."""
    out = growth_percentile_curves(
        sex,
        measure,
        chart_standard=chart_standard,
        gestational_age_weeks=gestational_age_weeks,
        age_max=age_max,
    )
    if out.get("ok") is False:
        raise _err(400, "growth_curves_failed", out.get("error") or out.get("detail"))
    return out


@router.post("/growth")
def growth(body: GrowthBody, request: Request):
    svc = get_services()
    # A child_id here reaches the child's record (reads its GA, writes a growth
    # row). Reject one that does not belong to the signed-in account before any
    # of that happens, so growth cannot be written into another family's chart.
    _require_owned_child(svc, body.child_id, current_user(request))
    ga = body.gestational_age_weeks
    if body.child_id and ga is None:
        child = svc.db.get_child(body.child_id) or {}
        ga = child.get("gestational_age_weeks")
    if body.child_id:
        out = svc.assistant.record_growth_and_overlay(
            body.child_id,
            body.sex,
            body.measure,
            body.weeks,
            body.value,
            age_months=body.age_months,
        )
    else:
        out = dispatch_tool(
            "overlay_growth_on_chart",
            {
                "sex": body.sex,
                "measure": body.measure,
                "weeks": body.weeks,
                "value": body.value,
                "age_months": body.age_months,
                "gestational_age_weeks": ga,
            },
            db=svc.db,
        )
    if out.get("ok") is False:
        raise _err(400, "growth_failed", out.get("error") or out.get("detail"))
    # UI fetches charts via GET /api/overlays/{filename}
    fname = out.get("overlay_filename")
    if not fname and out.get("overlay_path"):
        from pathlib import Path

        fname = Path(out["overlay_path"]).name
        out["overlay_filename"] = fname
    if fname:
        out["overlay"] = fname
    return out


@router.get("/asq/{age}/questions")
def asq_questions(age: int):
    out = list_asq_questions(age)
    if out.get("ok") is False:
        raise _err(404, "asq_not_found", out.get("error") or out.get("detail"))
    return out


@router.post("/asq/score")
def asq_score(body: AsqScoreBody, request: Request):
    svc = get_services()
    _require_owned_child(svc, body.child_id, current_user(request))
    if body.child_id:
        return svc.assistant.run_asq_session(body.child_id, body.age_months, body.domain_answers)
    result = dispatch_tool(
        "score_asq_questionnaire", {"domain_answers": body.domain_answers}, db=svc.db
    )
    if result.get("ok") is False:
        raise _err(400, "asq_score_failed", result.get("error") or result.get("detail"))
    return {"instrument": "ASQ", "age_months": body.age_months, "result": result}


@router.get("/mchat/questions")
def mchat_questions():
    out = list_mchat_questions()
    if out.get("ok") is False:
        raise _err(404, "mchat_not_found", out.get("error") or out.get("detail"))
    return out


@router.post("/mchat/score")
def mchat_score(body: MchatScoreBody, request: Request):
    svc = get_services()
    _require_owned_child(svc, body.child_id, current_user(request))
    try:
        answers = {int(k): v for k, v in body.answers.items()}
    except (TypeError, ValueError) as exc:
        raise _err(
            400, "invalid_answers", "M-CHAT answer keys must be question numbers."
        ) from exc
    if body.child_id:
        return svc.assistant.run_mchat_session(body.child_id, answers)
    result = dispatch_tool("score_mchat", {"answers": body.answers}, db=svc.db)
    if result.get("ok") is False:
        raise _err(400, "mchat_score_failed", result.get("error") or result.get("detail"))
    return {"instrument": "M-CHAT-R", "result": result}


@router.get("/overlays/{filename}")
def get_overlay(filename: str):
    if "/" in filename or "\\" in filename or ".." in filename:
        raise _err(400, "invalid_filename", "Path separators not allowed")
    root = Path(OVERLAY_DIR).resolve()
    path = (root / filename).resolve()
    # Defence in depth: a drive-relative or absolute name would otherwise escape
    # the overlay directory even without a path separator (e.g. "C:secret.png").
    if path.parent != root:
        raise _err(400, "invalid_filename", "Filename must resolve inside the overlay directory")
    if not path.is_file():
        raise _err(404, "not_found", f"Overlay not found: {filename}")
    return FileResponse(path, media_type="image/png")
