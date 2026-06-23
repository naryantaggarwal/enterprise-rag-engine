"""
tests/integration/test_api.py
Integration tests for the FastAPI routes.
Uses TestClient — no real HTTP, but real route logic with mocked LLM/DB calls.
"""
import json
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture
def client():
    return TestClient(app)


class TestHealth:
    def test_health_returns_200(self, client):
        with patch("app.api.routes.get_collection_stats", return_value={"total_chunks": 42}):
            resp = client.get("/api/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "healthy"
        assert "llm_provider" in data


class TestQuery:
    @patch("app.api.routes.query_rag")
    def test_query_returns_answer(self, mock_rag, client):
        mock_response = MagicMock()
        mock_response.answer = "The refund policy allows 30 days."
        mock_response.sources = [{"file": "policy.pdf", "chunk_index": 2}]
        mock_response.query = "What is the refund policy?"
        mock_response.retrieved_chunks = [MagicMock(), MagicMock()]
        mock_rag.return_value = mock_response

        resp = client.post(
            "/api/v1/query",
            json={"question": "What is the refund policy?"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "answer" in data
        assert data["chunk_count"] == 2

    def test_empty_question_rejected(self, client):
        resp = client.post("/api/v1/query", json={"question": ""})
        assert resp.status_code == 422  # Pydantic validation error

    def test_question_too_long_rejected(self, client):
        resp = client.post("/api/v1/query", json={"question": "x" * 2001})
        assert resp.status_code == 422


class TestCollectionStats:
    @patch("app.api.routes.get_collection_stats")
    def test_stats_endpoint(self, mock_stats, client):
        mock_stats.return_value = {"total_chunks": 100, "collection_name": "test"}
        resp = client.get("/api/v1/collection/stats")
        assert resp.status_code == 200
        assert resp.json()["total_chunks"] == 100


class TestEvalEndpoints:
    def test_latest_eval_404_when_missing(self, client):
        with patch("app.api.routes.Path.exists", return_value=False):
            resp = client.get("/api/v1/eval/latest")
        assert resp.status_code == 404

    def test_run_eval_returns_202_ish(self, client):
        resp = client.post("/api/v1/eval/run", json={})
        assert resp.status_code == 200  # Background task accepted
        assert "started" in resp.json()["status"]
