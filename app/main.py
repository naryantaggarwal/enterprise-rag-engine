"""
app/main.py
FastAPI application entrypoint.

Run locally:
  uvicorn app.main:app --reload --port 8000

Production:
  uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 2
"""
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from app.api.routes import router
from app.core.config import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown logic."""
    settings = get_settings()
    logger.info(f"Starting enterprise-rag-engine [{settings.app_env}]")
    logger.info(f"LLM: {settings.llm_provider}/{settings.llm_model}")
    logger.info(f"Chunk size: {settings.chunk_size}, overlap: {settings.chunk_overlap}")
    yield
    logger.info("Shutting down enterprise-rag-engine")


def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Enterprise RAG Engine",
        description=(
            "Production-grade document Q&A with automated evaluation. "
            "Built with LangChain, ChromaDB, FastAPI, and RAGAS."
        ),
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────
    origins = (
        ["*"]
        if not settings.is_production
        else ["https://your-frontend-domain.com"]
    )
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Routes ────────────────────────────────────────────────────
    app.include_router(router, prefix="/api/v1")

    return app


app = create_app()
