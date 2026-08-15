from __future__ import annotations

from pathlib import Path

from assistant.refdata import asq_scoring
from assistant.settings import get_settings

_settings = get_settings()

# Every path derives from NESTLING_ROOT / NESTLING_DATA_DIR so a container or a
# test run can relocate state without touching code.
ROOT = Path(_settings.nestling_root).resolve()
EXTRACTED = ROOT / "extracted"
DATA = Path(_settings.nestling_data_dir) if _settings.nestling_data_dir else ROOT / "data"
EN_DIR = DATA / "en"
KNOWLEDGE_DIR = DATA / "knowledge"
CHILD_DB_PATH = Path(_settings.nestling_child_db or DATA / "children" / "children.db")
CHAT_DB_PATH = Path(_settings.nestling_chat_db or DATA / "children" / "chat.db")
CHILD_INDEX_DIR = DATA / "children" / "rag_index"
MEDICAL_INDEX_DIR = DATA / "knowledge" / "rag_index"
OVERLAY_DIR = DATA / "overlays"

# Model identifiers are never declared here — they come from assistant.settings
# (NESTLING_LLM_MODEL / NESTLING_VISION_MODEL / NESTLING_TOOL_MODEL) only.

UPLOAD_DIR = DATA / "uploads"
MODELS_DIR = DATA / "models" / "llm"

_asq = asq_scoring()
ASQ_SCORE_YES = int(_asq["score_yes"])
ASQ_SCORE_SOMETIMES = int(_asq["score_sometimes"])
ASQ_SCORE_NOT_YET = int(_asq["score_not_yet"])
ASQ_DEFAULT_CUTOFF = int(_asq["default_cutoff"])
ASQ_CUTOFF_SOURCE = str(_asq.get("cutoff_source", "unverified_default"))
