"""
app/core/config.py
Central configuration loaded from environment variables.
Uses pydantic-settings so every value is type-validated at startup.
"""
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM providers ─────────────────────────────────────────────
    anthropic_api_key: str = Field(default="", description="Anthropic API key")
    openai_api_key: str = Field(default="", description="OpenAI API key")
    llm_provider: Literal["anthropic", "openai"] = "anthropic"
    llm_model: str = "claude-sonnet-4-20250514"
    embedding_model: str = "text-embedding-3-small"

    # ── Vector store ──────────────────────────────────────────────
    chroma_persist_dir: Path = Path("./data/chroma_db")
    chroma_collection_name: str = "enterprise_rag"

    # ── RAG pipeline tuning ───────────────────────────────────────
    chunk_size: int = Field(default=512, ge=128, le=4096)
    chunk_overlap: int = Field(default=64, ge=0, le=512)
    retriever_top_k: int = Field(default=4, ge=1, le=20)
    mmr_diversity_score: float = Field(default=0.3, ge=0.0, le=1.0)

    # ── Evaluation gates ──────────────────────────────────────────
    ragas_eval_dataset_path: Path = Path("./tests/eval/eval_dataset.json")
    hallucination_threshold: float = Field(default=0.10, ge=0.0, le=1.0)
    faithfulness_threshold: float = Field(default=0.80, ge=0.0, le=1.0)

    # ── LangSmith ─────────────────────────────────────────────────
    langchain_tracing_v2: bool = False
    langchain_api_key: str = ""
    langchain_project: str = "enterprise-rag-engine"

    # ── App settings ──────────────────────────────────────────────
    app_env: Literal["development", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = "INFO"
    docs_dir: Path = Path("./data/documents")
    max_upload_size_mb: int = Field(default=50, ge=1, le=500)

    @field_validator("chroma_persist_dir", "docs_dir", mode="before")
    @classmethod
    def ensure_path(cls, v):
        p = Path(v)
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def is_production(self) -> bool:
        return self.app_env == "production"

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_size_mb * 1024 * 1024


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Cached singleton — call this everywhere instead of instantiating Settings."""
    return Settings()
