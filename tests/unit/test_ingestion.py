"""
tests/unit/test_ingestion.py
Unit tests for the document ingestion pipeline.
Uses a mock vector store to avoid real ChromaDB dependency in CI.
"""
import json
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.core.ingestion import _file_hash, _loader_for, ingest_file


class TestFileHash:
    def test_hash_is_deterministic(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello world")
        h1 = _file_hash(f)
        h2 = _file_hash(f)
        assert h1 == h2

    def test_different_content_different_hash(self, tmp_path):
        f1 = tmp_path / "a.txt"
        f2 = tmp_path / "b.txt"
        f1.write_text("content a")
        f2.write_text("content b")
        assert _file_hash(f1) != _file_hash(f2)

    def test_hash_is_sha256_length(self, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("test")
        assert len(_file_hash(f)) == 64  # SHA-256 hex = 64 chars


class TestLoaderFor:
    def test_pdf_loader(self, tmp_path):
        from langchain_community.document_loaders import PyPDFLoader
        f = tmp_path / "doc.pdf"
        f.write_bytes(b"%PDF-1.4")  # minimal PDF header
        loader = _loader_for(f)
        assert isinstance(loader, PyPDFLoader)

    def test_txt_loader(self, tmp_path):
        from langchain_community.document_loaders import TextLoader
        f = tmp_path / "doc.txt"
        f.write_text("hello")
        loader = _loader_for(f)
        assert isinstance(loader, TextLoader)

    def test_unsupported_raises(self, tmp_path):
        f = tmp_path / "doc.xls"
        f.write_text("data")
        with pytest.raises(ValueError, match="Unsupported file type"):
            _loader_for(f)


class TestIngestFile:
    @patch("app.core.ingestion._get_vector_store")
    @patch("app.core.ingestion.get_embeddings")
    def test_skips_duplicate(self, mock_embeddings, mock_vs, tmp_path):
        """Files with matching hash should be skipped without re-embedding."""
        mock_store = MagicMock()
        mock_store.get.return_value = {"ids": ["existing-chunk-id"]}
        mock_vs.return_value = mock_store

        f = tmp_path / "doc.txt"
        f.write_text("Some content")

        result = ingest_file(f)
        assert result["status"] == "skipped"
        mock_store.add_documents.assert_not_called()

    @patch("app.core.ingestion._get_vector_store")
    @patch("app.core.ingestion.get_embeddings")
    def test_ingests_new_file(self, mock_embeddings, mock_vs, tmp_path):
        """New files should be chunked and added to vector store."""
        mock_store = MagicMock()
        mock_store.get.return_value = {"ids": []}  # no existing hash
        mock_vs.return_value = mock_store

        f = tmp_path / "doc.txt"
        f.write_text("This is a test document with enough content to chunk properly. " * 20)

        result = ingest_file(f)
        assert result["status"] == "ingested"
        assert result["chunks"] > 0
        mock_store.add_documents.assert_called_once()
