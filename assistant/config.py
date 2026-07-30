from __future__ import annotations

from pathlib import Path

from assistant.refdata import asq_scoring

ROOT = Path(__file__).resolve().parent.parent
EXTRACTED = ROOT / "extracted"
DATA = ROOT / "data"
EN_DIR = DATA / "en"
KNOWLEDGE_DIR = DATA / "knowledge"
CHILD_DB_PATH = DATA / "children" / "children.db"
CHAT_DB_PATH = DATA / "children" / "chat.db"
CHILD_INDEX_DIR = DATA / "children" / "rag_index"
MEDICAL_INDEX_DIR = DATA / "knowledge" / "rag_index"
OVERLAY_DIR = DATA / "overlays"

# Local LLM stack (OpenAI-compatible endpoint):
# - Unified text/vision model service: Qwen/Qwen3.5-4B
# Tool calling: Salesforce/xLAM-1b-fc-r (optional) or deterministic router.
XLAM_MODEL_ID = "Salesforce/xLAM-1b-fc-r"
TEXT_MODEL_ID = "Qwen/Qwen3.5-4B"
VISION_MODEL_ID = "Qwen/Qwen3.5-4B"
TEXT_GGUF = ""
VISION_GGUF = ""
VISION_MMPROJ = ""

# Backward compatibility for older imports/tests.
QWEN_MODEL_ID = TEXT_MODEL_ID
QWEN_GGUF = TEXT_GGUF

ALLOWED_HF_MODELS = (XLAM_MODEL_ID,)

UPLOAD_DIR = DATA / "uploads"
MODELS_DIR = DATA / "models" / "llm"

_asq = asq_scoring()
ASQ_SCORE_YES = int(_asq["score_yes"])
ASQ_SCORE_SOMETIMES = int(_asq["score_sometimes"])
ASQ_SCORE_NOT_YET = int(_asq["score_not_yet"])
ASQ_DEFAULT_CUTOFF = int(_asq["default_cutoff"])
ASQ_CUTOFF_SOURCE = str(_asq.get("cutoff_source", "unverified_default"))
