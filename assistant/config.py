from __future__ import annotations

from pathlib import Path

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

# ONLY these Hugging Face / local LLM backends are used for generation:
# - Optional: PrismML Bonsai-27B-Q1_0 via llama-server (NESTLING_BONSAI_URL)
# - Fallback HF: PleIAs/Pleias-RAG-1B
# Tool calling: Salesforce/xLAM-1b-fc-r (optional) or deterministic router
XLAM_MODEL_ID = "Salesforce/xLAM-1b-fc-r"
PLEIAS_RAG_MODEL_ID = "PleIAs/Pleias-RAG-1B"
BONSAI_MODEL_ID = "prism-ml/Bonsai-27B-gguf"
BONSAI_GGUF = "Bonsai-27B-Q1_0.gguf"
BONSAI_MMPROJ = "Bonsai-27B-mmproj-Q8_0.gguf"

ALLOWED_HF_MODELS = (XLAM_MODEL_ID, PLEIAS_RAG_MODEL_ID, BONSAI_MODEL_ID)

UPLOAD_DIR = DATA / "uploads"
MODELS_DIR = DATA / "models" / "bonsai"

ASQ_SCORE_YES = 10
ASQ_SCORE_SOMETIMES = 5
ASQ_SCORE_NOT_YET = 0
ASQ_DEFAULT_CUTOFF = 30
