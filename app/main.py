"""Nestling FastAPI application — API under /api, static UI at /."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router as api_router
from app.services import create_services, peek_services, set_services
from assistant.settings import get_settings

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
WEB_DIR = ROOT / "web"


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Tests may inject Services before startup; don't overwrite them.
    if peek_services() is None:
        set_services(create_services())
    yield
    svc = peek_services()
    set_services(None)
    if svc is not None:
        try:
            svc.close()
        except Exception as exc:
            log.warning("Error during service shutdown: %s", exc)


app = FastAPI(
    title="Nestling",
    description="Pediatric parent assistant API",
    version="0.3.0",
    lifespan=lifespan,
)

_settings = get_settings()
_cors_origins = _settings.cors_origin_list
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    # Credentials cannot be combined with a wildcard origin; browsers reject the
    # response and Starlette would silently drop the header anyway.
    allow_credentials=_cors_origins != ["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    if isinstance(exc.detail, dict):
        return JSONResponse(status_code=exc.status_code, content=exc.detail)
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": "http_error", "detail": str(exc.detail)},
    )


@app.exception_handler(Exception)
async def unhandled_exception(request: Request, exc: Exception):
    # Log the traceback server-side; the client gets no internal details.
    log.exception("Unhandled error on %s %s", request.method, request.url.path)
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "detail": "Internal server error"},
    )


app.include_router(api_router, prefix="/api")

if WEB_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(WEB_DIR), html=True), name="web")
