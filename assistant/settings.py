"""Central typed settings — all NESTLING_* / LLM env vars live here."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Paths (overrideable for tests)
    nestling_root: Path = ROOT
    nestling_child_db: Path | None = None
    nestling_chat_db: Path | None = None

    # Models
    nestling_load_models: bool = False
    nestling_use_llm: bool = True
    nestling_llm_url: str = "http://llm:8000"
    nestling_vision_llm_url: str = ""
    nestling_llm_model: str = "Qwen/Qwen3.5-4B"
    nestling_vision_model: str = "Qwen/Qwen3.5-4B"
    nestling_tool_model: str = "Salesforce/xLAM-1b-fc-r"
    nestling_llm_gguf: str = ""
    nestling_vision_gguf: str = ""
    nestling_vision_mmproj: str = ""
    nestling_use_dense: bool = False
    nestling_embedding_model: str = "BAAI/bge-m3"

    # API / security
    nestling_cors_origins: str = "*"
    nestling_api_key: str | None = None
    nestling_max_upload_bytes: int = 8_000_000
    nestling_session_list_limit: int = 40
    nestling_dossier_overlay_limit: int = 12
    nestling_tool_overlay_limit: int = 8

    # Chat memory
    nestling_history_window: int = 12
    nestling_summary_trigger_turns: int = 16

    # LLM generation defaults
    llm_temperature: float = 0.7
    llm_top_p: float = 0.95
    llm_max_tokens_rag: int = 640
    llm_max_tokens_vision: int = 700

    # Chart rendering
    chart_dpi: int = 140
    chart_figsize_w: float = 10.0
    chart_figsize_h: float = 6.0

    @property
    def cors_origin_list(self) -> list[str]:
        raw = (self.nestling_cors_origins or "*").strip()
        if raw == "*":
            return ["*"]
        return [o.strip() for o in raw.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
