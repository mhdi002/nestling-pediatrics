"""Central typed settings — all NESTLING_* / LLM env vars live here."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths (overrideable for tests and containers)
    nestling_root: Path = ROOT
    nestling_data_dir: Path | None = None
    nestling_child_db: Path | None = None
    nestling_chat_db: Path | None = None

    # Models
    nestling_load_models: bool = False
    nestling_use_llm: bool = True
    # Empty by default: the sidecar is opt-in, and an unreachable default URL
    # would make every request pay a failed connection probe.
    nestling_llm_url: str = ""
    nestling_vision_llm_url: str = ""
    nestling_llm_model: str = "Qwen/Qwen3.5-4B"
    nestling_vision_model: str = "Qwen/Qwen3.5-4B"
    nestling_tool_model: str = "Salesforce/xLAM-1b-fc-r"
    nestling_tool_max_new_tokens: int = 512
    nestling_llm_gguf: str = ""
    nestling_vision_gguf: str = ""
    nestling_vision_mmproj: str = ""
    nestling_use_dense: bool = False
    nestling_embedding_model: str = "BAAI/bge-m3"

    # LLM transport
    nestling_llm_timeout: float = 180.0
    nestling_llm_probe_timeout: float = 1.5
    nestling_llm_ready_cache_seconds: float = 5.0
    nestling_llm_error_detail_chars: int = 400

    # API / security
    nestling_cors_origins: str = "*"
    nestling_api_key: str | None = None
    nestling_max_upload_bytes: int = 8_000_000
    nestling_max_image_pixels: int = 40_000_000
    nestling_allowed_upload_types: str = "image/png,image/jpeg,image/webp,image/gif,image/bmp"
    nestling_session_list_limit: int = 40
    nestling_session_list_max_limit: int = 200
    nestling_dossier_overlay_limit: int = 12
    nestling_tool_overlay_limit: int = 8
    nestling_session_title_chars: int = 60
    nestling_stream_chunk_chars: int = 24

    # Runtime translation (deep-translator has no built-in timeout)
    nestling_translate_timeout: float = 6.0
    nestling_translate_workers: int = 4

    # Chat memory
    nestling_history_window: int = 12
    nestling_summary_trigger_turns: int = 16
    nestling_history_response_limit: int = 200
    nestling_summary_max_chars: int = 4000
    nestling_summary_fold_max_chars: int = 3500
    nestling_summary_fold_turns: int = 24
    nestling_summary_turn_chars: int = 180
    nestling_memory_recent_chars: int = 1200
    nestling_medical_query_chars: int = 500

    # LLM generation defaults
    llm_temperature: float = 0.7
    llm_top_p: float = 0.95
    llm_max_tokens_default: int = 512
    llm_max_tokens_rag: int = 640
    llm_max_tokens_vision: int = 700
    llm_max_tokens_chat: int = 320
    llm_rag_temperature: float = 0.75
    llm_vision_temperature: float = 0.2
    llm_vision_top_p: float = 0.9
    llm_prompt_context_chars: int = 1800
    llm_prompt_query_chars: int = 400
    llm_answer_max_chars: int = 1200

    # Intent routing
    nestling_router_llm_min_confidence: float = 0.7

    # Retrieval
    nestling_rag_top_k: int = 5
    nestling_rag_pool_multiplier: int = 5
    nestling_rag_pool_min: int = 25
    nestling_rag_query_multiplier: int = 3
    nestling_rag_query_min: int = 15
    nestling_rag_topic_hits: int = 3
    nestling_rag_speech_hits: int = 2
    nestling_rag_extract_chars: int = 420
    nestling_rag_extract_min_sentence_chars: int = 160
    nestling_rag_extract_keep_ratio: float = 0.75
    nestling_dense_rrf_k: int = 60
    nestling_dense_cache_size: int = 2048

    # Chart rendering
    chart_dpi: int = 140
    chart_figsize_w: float = 10.0
    chart_figsize_h: float = 6.0
    chart_curve_step_months: float = 0.5
    chart_curve_step_weeks: float = 0.5

    @property
    def cors_origin_list(self) -> list[str]:
        raw = (self.nestling_cors_origins or "*").strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]

    @property
    def allowed_upload_types(self) -> frozenset[str]:
        raw = (self.nestling_allowed_upload_types or "").strip()
        return frozenset(t.strip().lower() for t in raw.split(",") if t.strip())


@lru_cache
def get_settings() -> Settings:
    return Settings()


def reset_settings() -> None:
    """Drop the cached Settings so a changed environment is picked up (tests)."""
    get_settings.cache_clear()
