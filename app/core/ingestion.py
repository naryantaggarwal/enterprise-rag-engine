"""
app/core/ingestion.py
Document ingestion pipeline: load → split → embed → store.

Supports PDF, DOCX, TXT, and Markdown files.
Chunking strategy uses RecursiveCharacterTextSplitter with configurable
size and overlap; tuning these is the #1 lever on RAG quality.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import (
    PyPDFLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
)
from langchain_chroma import Chroma
from loguru import logger

from app.core.config import get_settings
from app.core.embeddings import get_embeddings


def _loader_for(path: Path):
    """Return the right LangChain loader based on file extension."""
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return PyPDFLoader(str(path))
    elif suffix in {".docx", ".doc"}:
        return UnstructuredWordDocumentLoader(str(path))
    elif suffix in {".txt", ".md", ".rst"}:
        return TextLoader(str(path), encoding="utf-8")
    else:
        raise ValueError(f"Unsupported file type: {suffix}")


def _file_hash(path: Path) -> str:
    """SHA-256 of file contents — used to skip already-ingested docs."""
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            sha.update(chunk)
    return sha.hexdigest()


def _get_vector_store(embeddings) -> Chroma:
    settings = get_settings()
    return Chroma(
        collection_name=settings.chroma_collection_name,
        embedding_function=embeddings,
        persist_directory=str(settings.chroma_persist_dir),
    )


def ingest_file(file_path: Path, metadata: Optional[dict] = None) -> dict:
    """
    Ingest a single document file into ChromaDB.

    Returns a summary dict with chunk count and whether the doc was
    already present (deduplication via file hash stored in metadata).
    """
    settings = get_settings()
    embeddings = get_embeddings()
    vector_store = _get_vector_store(embeddings)

    file_hash = _file_hash(file_path)

    # ── Deduplication check ───────────────────────────────────────
    existing = vector_store.get(where={"file_hash": file_hash}, limit=1)
    if existing["ids"]:
        logger.info(f"Skipping {file_path.name} — already ingested (hash match)")
        return {"status": "skipped", "file": file_path.name, "chunks": 0}

    # ── Load raw document ─────────────────────────────────────────
    logger.info(f"Loading {file_path.name}...")
    loader = _loader_for(file_path)
    raw_docs: list[Document] = loader.load()
    logger.info(f"  Loaded {len(raw_docs)} page(s) from {file_path.name}")

    # ── Chunk ─────────────────────────────────────────────────────
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )
    chunks: list[Document] = splitter.split_documents(raw_docs)

    # ── Enrich metadata ───────────────────────────────────────────
    base_meta = {
        "source_file": file_path.name,
        "file_hash": file_hash,
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
    }
    if metadata:
        base_meta.update(metadata)

    for i, chunk in enumerate(chunks):
        chunk.metadata.update({**base_meta, "chunk_index": i, "total_chunks": len(chunks)})

    # ── Embed + store ─────────────────────────────────────────────
    logger.info(f"  Embedding {len(chunks)} chunks...")
    vector_store.add_documents(chunks)
    logger.info(f"  ✓ Stored {len(chunks)} chunks for {file_path.name}")

    return {"status": "ingested", "file": file_path.name, "chunks": len(chunks)}


def ingest_directory(directory: Optional[Path] = None) -> list[dict]:
    """
    Ingest all supported documents in a directory.
    Skips files that are already in the vector store.
    """
    settings = get_settings()
    target_dir = directory or settings.docs_dir
    supported = {".pdf", ".docx", ".doc", ".txt", ".md", ".rst"}

    files = [f for f in target_dir.iterdir() if f.suffix.lower() in supported]
    if not files:
        logger.warning(f"No supported files found in {target_dir}")
        return []

    results = []
    for f in sorted(files):
        try:
            result = ingest_file(f)
            results.append(result)
        except Exception as e:
            logger.error(f"Failed to ingest {f.name}: {e}")
            results.append({"status": "error", "file": f.name, "error": str(e)})

    ingested = sum(1 for r in results if r["status"] == "ingested")
    skipped = sum(1 for r in results if r["status"] == "skipped")
    logger.info(f"Ingestion complete: {ingested} new, {skipped} skipped, {len(results) - ingested - skipped} errors")
    return results


def get_collection_stats() -> dict:
    """Return basic stats about what's in the vector store."""
    embeddings = get_embeddings()
    vector_store = _get_vector_store(embeddings)
    collection = vector_store._collection
    count = collection.count()
    return {
        "total_chunks": count,
        "collection_name": get_settings().chroma_collection_name,
        "persist_dir": str(get_settings().chroma_persist_dir),
    }


def clear_collection() -> dict:
    """Wipe the vector store. Useful for testing and re-ingestion."""
    settings = get_settings()
    embeddings = get_embeddings()
    vector_store = _get_vector_store(embeddings)
    vector_store._collection.delete(where={})  # delete all
    logger.warning("Vector store cleared.")
    return {"status": "cleared", "collection": settings.chroma_collection_name}
