"""
app/api/routes.py
FastAPI route handlers.

Endpoints:
  POST /ingest/file        - Upload and ingest a document
  POST /ingest/directory   - Ingest all docs in configured directory
  POST /query              - Ask a question
  GET  /collection/stats   - Vector store stats
  DELETE /collection       - Clear vector store (dev only)
  GET  /eval/latest        - Latest evaluation results
  POST /eval/run           - Trigger evaluation run
  GET  /health             - Health check
"""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Annotated, Optional

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse
from loguru import logger
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.ingestion import (
    clear_collection,
    get_collection_stats,
    ingest_directory,
    ingest_file,
)
from app.core.rag_chain import query_rag

router = APIRouter()


# ── Request / Response models ─────────────────────────────────────────────────

class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=2000)
    chat_history: list[list[str]] = Field(
        default_factory=list,
        description="List of [human, ai] message pairs for conversation context",
    )


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    query: str
    chunk_count: int


class EvalRequest(BaseModel):
    dataset_path: Optional[str] = None
    set_as_baseline: bool = False


# ── Ingestion endpoints ───────────────────────────────────────────────────────

@router.post("/ingest/file", tags=["ingestion"])
async def upload_and_ingest(file: UploadFile = File(...)):
    """
    Upload a document file and ingest it into the vector store.
    Supported: PDF, DOCX, TXT, MD
    """
    settings = get_settings()

    if file.size and file.size > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: {settings.max_upload_size_mb}MB",
        )

    allowed_extensions = {".pdf", ".docx", ".doc", ".txt", ".md", ".rst"}
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in allowed_extensions:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported file type: {suffix}. Allowed: {allowed_extensions}",
        )

    # Write to temp file (UploadFile is a stream, not a path)
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = Path(tmp.name)

    try:
        result = ingest_file(tmp_path, metadata={"original_filename": file.filename})
        # Rename to keep original name for source citations
        final_path = settings.docs_dir / file.filename
        tmp_path.rename(final_path)
        return {"status": "success", "result": result}
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        logger.error(f"Ingestion failed for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/ingest/directory", tags=["ingestion"])
async def ingest_all_documents():
    """Ingest all documents in the configured docs directory."""
    results = ingest_directory()
    total = len(results)
    ingested = sum(1 for r in results if r.get("status") == "ingested")
    return {
        "status": "complete",
        "total_files": total,
        "ingested": ingested,
        "skipped": sum(1 for r in results if r.get("status") == "skipped"),
        "errors": sum(1 for r in results if r.get("status") == "error"),
        "details": results,
    }


# ── Query endpoint ────────────────────────────────────────────────────────────

@router.post("/query", response_model=QueryResponse, tags=["query"])
async def ask_question(request: QueryRequest):
    """
    Ask a question against the ingested documents.
    Returns the answer with source citations and retrieved chunk info.
    """
    # Convert [[human, ai], ...] to [(human, ai), ...]
    history_tuples = [tuple(pair) for pair in request.chat_history if len(pair) == 2]

    response = query_rag(
        question=request.question,
        chat_history=history_tuples,
    )

    return QueryResponse(
        answer=response.answer,
        sources=response.sources,
        query=response.query,
        chunk_count=len(response.retrieved_chunks),
    )


# ── Collection management ─────────────────────────────────────────────────────

@router.get("/collection/stats", tags=["collection"])
async def collection_stats():
    """Return vector store statistics."""
    return get_collection_stats()


@router.delete("/collection", tags=["collection"])
async def clear_vector_store():
    """Clear the entire vector store. Development only."""
    settings = get_settings()
    if settings.is_production:
        raise HTTPException(
            status_code=403,
            detail="Cannot clear collection in production environment",
        )
    return clear_collection()


# ── Evaluation endpoints ──────────────────────────────────────────────────────

@router.get("/eval/latest", tags=["evaluation"])
async def get_latest_eval():
    """Return the most recent evaluation summary."""
    summary_path = Path("./eval_results/latest_summary.json")
    if not summary_path.exists():
        raise HTTPException(
            status_code=404,
            detail="No evaluation results found. Run /eval/run first.",
        )
    return json.loads(summary_path.read_text())


@router.post("/eval/run", tags=["evaluation"])
async def run_evaluation(request: EvalRequest, background_tasks: BackgroundTasks):
    """
    Trigger an evaluation run in the background.
    Results are written to eval_results/ and accessible via /eval/latest.
    """
    from app.evaluation.ragas_eval import run_evaluation as _run_eval
    from app.evaluation.regression_tracker import record_run, set_baseline

    def _background_eval():
        dataset_path = Path(request.dataset_path) if request.dataset_path else None
        summary = _run_eval(dataset_path=dataset_path)
        record_run(summary)
        if request.set_as_baseline:
            set_baseline(summary)

    background_tasks.add_task(_background_eval)
    return {"status": "evaluation_started", "message": "Check /eval/latest for results"}


# ── Health check ──────────────────────────────────────────────────────────────

@router.get("/health", tags=["health"])
async def health_check():
    settings = get_settings()
    stats = get_collection_stats()
    return {
        "status": "healthy",
        "environment": settings.app_env,
        "llm_provider": settings.llm_provider,
        "llm_model": settings.llm_model,
        "vector_store": stats,
    }
