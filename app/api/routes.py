"""Nestling API routes under /api (mounted by app.main)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field

from app.services import get_services
from assistant.config import OVERLAY_DIR, UPLOAD_DIR
from assistant.tools.clinical import (
    dispatch_tool,
    list_asq_questions,
    list_mchat_questions,
)

router = APIRouter()


def _err(status: int, error: str, detail: str | None = None) -> HTTPException:
    return HTTPException(status_code=status, detail={"error": error, "detail": detail or error})


# --- request bodies ---


class ChildCreate(BaseModel):
    name: str
    sex: str
    date_of_birth: str | None = None
    gestational_age_weeks: float | None = None
    notes: str = ""


class SessionCreate(BaseModel):
    child_id: str | None = None
    title: str | None = None


class ChatBody(BaseModel):
    # Optional: UI may chat before creating a session — we auto-create.
    session_id: str | None = None
    message: str = Field(..., min_length=1)
    child_id: str | None = None
    ui_lang: str | None = None  # 'fa' | 'en'


class GrowthBody(BaseModel):
    sex: str
    measure: str
    weeks: float | None = None
    value: float
    child_id: str | None = None
    age_months: float | None = None
    gestational_age_weeks: float | None = None


class AsqScoreBody(BaseModel):
    domain_answers: dict[str, list[str]]
    age_months: int
    child_id: str | None = None


class MchatScoreBody(BaseModel):
    answers: dict[str, str]
    child_id: str | None = None


@router.get("/health")
def health():
    bonsai = {"configured": False, "ready": False, "url": None}
    try:
        from assistant.llm.bonsai_client import bonsai_base_url, bonsai_enabled, get_bonsai

        if bonsai_enabled():
            client = get_bonsai()
            bonsai = {
                "configured": True,
                "ready": client.ready,
                "url": bonsai_base_url(),
                "model": client.model,
            }
    except Exception as exc:
        bonsai["error"] = str(exc)
    return {"status": "ok", "service": "nestling", "bonsai": bonsai}


@router.post("/children")
def create_child(body: ChildCreate):
    svc = get_services()
    try:
        cid = svc.db.create_child(
            body.name,
            body.sex,
            date_of_birth=body.date_of_birth,
            gestational_age_weeks=body.gestational_age_weeks,
            notes=body.notes,
        )
    except Exception as exc:
        raise _err(400, "create_child_failed", str(exc)) from exc
    return {"child_id": cid, "child": svc.db.get_child(cid)}


@router.get("/children")
def list_children():
    return {"children": get_services().db.list_children()}


@router.get("/children/{child_id}")
def get_child(child_id: str):
    child = get_services().db.get_child(child_id)
    if not child:
        raise _err(404, "not_found", f"Unknown child_id: {child_id}")
    return {"child": child}


@router.get("/children/{child_id}/dossier")
def child_dossier(child_id: str):
    """Full child record for UI + agent: profile, growth, screenings, chart overlays."""
    svc = get_services()
    child = svc.db.get_child(child_id)
    if not child:
        raise _err(404, "not_found", f"Unknown child_id: {child_id}")
    growth = svc.db.growth_history(child_id)
    screens = svc.db.screenings(child_id)
    ga = child.get("gestational_age_weeks")
    maturity = "preterm" if ga is not None and float(ga) < 37 else ("term" if ga is not None else "unknown")
    overlays = []
    try:
        for p in sorted(OVERLAY_DIR.glob(f"overlay_{child_id}_*.png"), key=lambda x: x.stat().st_mtime, reverse=True):
            overlays.append({"filename": p.name, "url": f"/api/overlays/{p.name}"})
    except Exception:
        pass
    return {
        "ok": True,
        "child_id": child_id,
        "profile": child,
        "maturity": maturity,
        "growth": growth,
        "screenings": screens,
        "overlays": overlays[:12],
        "summary": (
            f"{child.get('name')} ({child.get('sex')}, GA {ga}w, {maturity}): "
            f"{len(growth)} growth, {len(screens)} screening(s)."
        ),
    }


@router.post("/sessions")
def create_session(body: SessionCreate):
    svc = get_services()
    sid = svc.chat.create_session(child_id=body.child_id, title=body.title)
    return {"session_id": sid, "child_id": body.child_id}


@router.get("/sessions")
def list_sessions(child_id: str | None = None, limit: int = 40):
    svc = get_services()
    return {"sessions": svc.chat.list_sessions(child_id=child_id, limit=limit)}


@router.get("/sessions/{session_id}")
def get_session(session_id: str):
    svc = get_services()
    s = svc.chat.get_session(session_id)
    if not s:
        raise _err(404, "session_not_found", session_id)
    return {
        "session": s,
        "history": svc.chat.get_history(session_id),
    }


@router.post("/chat/vision")
async def chat_vision(
    message: str = Form(""),
    session_id: str | None = Form(None),
    child_id: str | None = Form(None),
    ui_lang: str | None = Form(None),
    image: UploadFile = File(...),
):
    """Parent photo + optional caption → vision (Bonsai mmproj) + medical RAG."""
    svc = get_services()
    raw = await image.read()
    if not raw:
        raise _err(400, "empty_image", "No image bytes received.")
    if len(raw) > 8_000_000:
        raise _err(400, "image_too_large", "Max image size is 8 MB.")
    mime = image.content_type or "image/jpeg"
    UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    from uuid import uuid4

    fname = f"{uuid4().hex}_{Path(image.filename or 'photo.jpg').name}"
    path = UPLOAD_DIR / fname
    path.write_bytes(raw)

    sid = session_id
    if not sid or not svc.chat.get_session(sid):
        sid = svc.chat.create_session(child_id=child_id)
    caption = (message or "").strip() or "Please look at this photo of my child."
    svc.chat.add_message(sid, "user", f"[photo:{fname}] {caption}")
    out = svc.assistant.analyze_parent_photo(
        raw, mime=mime, prompt=caption, ui_lang=ui_lang
    )
    svc.chat.add_message(sid, "assistant", out.get("reply") or "")
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


@router.post("/chat")
def chat(body: ChatBody):
    svc = get_services()
    sid = body.session_id
    if not sid or not svc.chat.get_session(sid):
        # Auto-create when UI has no session yet, or stale localStorage id after DB reset.
        sid = svc.chat.create_session(child_id=body.child_id)
    try:
        out = svc.assistant.chat(
            sid, body.message, child_id=body.child_id, ui_lang=body.ui_lang
        )
        # Auto-title from first user message
        s = svc.chat.get_session(sid) or {}
        if not (s.get("title") or "").strip():
            svc.chat.set_title(sid, (body.message or "").strip()[:60])
        return out
    except Exception as exc:
        raise _err(500, "chat_failed", str(exc)) from exc


@router.post("/growth")
def growth(body: GrowthBody):
    svc = get_services()
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
def asq_score(body: AsqScoreBody):
    svc = get_services()
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
def mchat_score(body: MchatScoreBody):
    svc = get_services()
    answers = {int(k): v for k, v in body.answers.items()}
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
    path = Path(OVERLAY_DIR) / filename
    if not path.is_file():
        raise _err(404, "not_found", f"Overlay not found: {filename}")
    return FileResponse(path, media_type="image/png")
