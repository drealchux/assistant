"""Unit and integration tests for retrieval components."""
from __future__ import annotations

import pickle
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from backend.chunking import DocumentChunker, RecursiveTextSplitter
from backend.config import Settings
from backend.models import DocumentMetadata
from backend.retrieval import BM25Index, reciprocal_rank_fusion


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        anthropic_api_key="test-key",
        qdrant_url="http://localhost:6333",
        bm25_index_path=str(tmp_path / "bm25.pkl"),
        chunks_path=str(tmp_path / "chunks.jsonl"),
        feedback_path=str(tmp_path / "feedback.jsonl"),
        query_log_path=str(tmp_path / "query_log.jsonl"),
    )


@pytest.fixture
def sample_metadata() -> DocumentMetadata:
    return DocumentMetadata(
        title="Employee Handbook",
        source_path="/docs/handbook.pdf",
        department="HR",
        country="Norway",
        document_type="pdf",
    )


@pytest.fixture
def sample_chunks() -> list[dict]:
    return [
        {
            "chunk_id": f"chunk_{i}",
            "text": f"Policy text about remote work in Norway. Employees may work from home {i} days.",
            "header": "Document: Employee Handbook | Section: Remote Work | Country: Norway",
            "document_id": "doc_1",
            "metadata": {"title": "Employee Handbook", "acl_groups": []},
        }
        for i in range(20)
    ]


# ── RecursiveTextSplitter ─────────────────────────────────────────────────────

class TestRecursiveTextSplitter:
    def test_short_text_not_split(self):
        splitter = RecursiveTextSplitter(chunk_size=500, chunk_overlap=50)
        text = "Short text under limit."
        chunks = splitter.split_text(text)
        assert len(chunks) == 1
        assert chunks[0] == text

    def test_long_text_split(self):
        splitter = RecursiveTextSplitter(chunk_size=100, chunk_overlap=20)
        text = " ".join(["word"] * 200)
        chunks = splitter.split_text(text)
        assert len(chunks) > 1
        for c in chunks:
            assert len(c) <= 120  # chunk_size + small slack

    def test_paragraph_boundary_preferred(self):
        splitter = RecursiveTextSplitter(chunk_size=50, chunk_overlap=0)
        text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
        chunks = splitter.split_text(text)
        assert any("Paragraph one" in c for c in chunks)

    def test_empty_text(self):
        splitter = RecursiveTextSplitter()
        assert splitter.split_text("") == []
        assert splitter.split_text("   ") == []


# ── DocumentChunker ───────────────────────────────────────────────────────────

class TestDocumentChunker:
    def test_creates_chunks_with_headers(self, sample_metadata: DocumentMetadata):
        chunker = DocumentChunker(chunk_size=200, chunk_overlap=20)
        text = "Remote work policy. " * 50
        chunks = chunker.chunk_document(text, sample_metadata)
        assert len(chunks) > 0
        for c in chunks:
            assert "Employee Handbook" in c.header
            assert c.document_id == sample_metadata.document_id

    def test_chunk_indices_sequential(self, sample_metadata: DocumentMetadata):
        chunker = DocumentChunker(chunk_size=300, chunk_overlap=30)
        text = "Some policy text. " * 100
        chunks = chunker.chunk_document(text, sample_metadata)
        for i, c in enumerate(chunks):
            assert c.chunk_index == i


# ── BM25Index ─────────────────────────────────────────────────────────────────

class TestBM25Index:
    def test_build_and_search(self, settings: Settings, sample_chunks: list[dict]):
        index = BM25Index(settings)
        index.build(sample_chunks)
        results = index.search("remote work Norway", top_k=5)
        assert len(results) > 0
        assert all("chunk_id" in r for r in results)
        assert all("score" in r for r in results)

    def test_search_empty_index(self, settings: Settings):
        index = BM25Index(settings)
        results = index.search("remote work", top_k=5)
        assert results == []

    def test_persist_and_reload(self, settings: Settings, sample_chunks: list[dict]):
        index = BM25Index(settings)
        index.build(sample_chunks)
        index2 = BM25Index(settings)
        results = index2.search("remote work", top_k=5)
        assert len(results) > 0

    def test_irrelevant_query_no_results(self, settings: Settings, sample_chunks: list[dict]):
        index = BM25Index(settings)
        index.build(sample_chunks)
        results = index.search("xyzxyzxyz123abc", top_k=5)
        assert results == []


# ── RRF Fusion ────────────────────────────────────────────────────────────────

class TestRRF:
    def test_fuses_two_lists(self):
        list1 = [{"chunk_id": "a", "text": "a", "score": 0.9},
                 {"chunk_id": "b", "text": "b", "score": 0.8}]
        list2 = [{"chunk_id": "b", "text": "b", "score": 0.7},
                 {"chunk_id": "c", "text": "c", "score": 0.6}]
        result = reciprocal_rank_fusion([list1, list2], k=60)
        ids = [r["chunk_id"] for r in result]
        # "b" appears in both lists, should be ranked highest
        assert ids[0] == "b"
        assert set(ids) == {"a", "b", "c"}

    def test_single_list_passthrough(self):
        lst = [{"chunk_id": "x", "text": "x", "score": 1.0}]
        result = reciprocal_rank_fusion([lst])
        assert result[0]["chunk_id"] == "x"

    def test_empty_lists(self):
        assert reciprocal_rank_fusion([]) == []
        assert reciprocal_rank_fusion([[]]) == []
