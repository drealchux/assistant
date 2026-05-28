"""Tests for generation pipeline, compression, ACL, and evaluation."""
from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from backend.acl import ACLFilter
from backend.config import Settings
from backend.evaluation import (
    EvaluationRunner,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from backend.models import CompressedChunk, EvalQuestion


# Fixtures 

@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    return Settings(
        anthropic_api_key="test-key",
        bm25_index_path=str(tmp_path / "bm25.pkl"),
        chunks_path=str(tmp_path / "chunks.jsonl"),
        feedback_path=str(tmp_path / "feedback.jsonl"),
        query_log_path=str(tmp_path / "query_log.jsonl"),
        eval_set_path=str(tmp_path / "eval.jsonl"),
    )


@pytest.fixture
def public_chunk() -> dict:
    return {
        "chunk_id": "chunk_pub",
        "text": "Remote work is allowed in Norway.",
        "header": "Document: Handbook | Section: Remote Work",
        "document_id": "doc_1",
        "metadata": {"title": "Handbook", "acl_groups": []},
        "score": 0.9,
    }


@pytest.fixture
def restricted_chunk() -> dict:
    return {
        "chunk_id": "chunk_hr",
        "text": "Salary bands are confidential.",
        "header": "Document: Comp Guide | Section: Salary",
        "document_id": "doc_2",
        "metadata": {"title": "Comp Guide", "acl_groups": ["hr-team", "managers"]},
        "score": 0.8,
    }


# ACL Filter

class TestACLFilter:
    def test_public_chunk_always_accessible(self, public_chunk: dict):
        f = ACLFilter()
        result = f.filter([public_chunk], user_groups=["engineering"])
        assert len(result) == 1

    def test_restricted_chunk_blocked_without_groups(self, restricted_chunk: dict):
        f = ACLFilter()
        result = f.filter([restricted_chunk], user_groups=[])
        assert len(result) == 0

    def test_restricted_chunk_accessible_with_correct_group(
        self, restricted_chunk: dict
    ):
        f = ACLFilter()
        result = f.filter([restricted_chunk], user_groups=["hr-team"])
        assert len(result) == 1

    def test_mixed_chunks_filtered_correctly(
        self, public_chunk: dict, restricted_chunk: dict
    ):
        f = ACLFilter()
        result = f.filter([public_chunk, restricted_chunk], user_groups=["engineering"])
        ids = [r["chunk_id"] for r in result]
        assert "chunk_pub" in ids
        assert "chunk_hr" not in ids

    def test_admin_group_can_access_restricted(self, restricted_chunk: dict):
        f = ACLFilter()
        result = f.filter([restricted_chunk], user_groups=["managers"])
        assert len(result) == 1


# Evaluation Metrics 

class TestRetrievalMetrics:
    def test_recall_perfect(self):
        assert recall_at_k(["a", "b", "c"], ["a", "b"], k=5) == 1.0

    def test_recall_zero(self):
        assert recall_at_k(["x", "y"], ["a", "b"], k=5) == 0.0

    def test_recall_partial(self):
        assert recall_at_k(["a", "x", "b"], ["a", "b", "c"], k=3) == pytest.approx(2 / 3)

    def test_recall_k_cutoff(self):
        # "b" is at position 3, outside k=2
        assert recall_at_k(["a", "x", "b"], ["a", "b"], k=2) == pytest.approx(0.5)

    def test_precision_at_k(self):
        assert precision_at_k(["a", "b", "c"], ["a", "b"], k=3) == pytest.approx(2 / 3)

    def test_precision_perfect(self):
        assert precision_at_k(["a", "b"], ["a", "b", "c"], k=2) == 1.0

    def test_ndcg_perfect(self):
        assert ndcg_at_k(["a", "b"], ["a", "b"], k=2) == pytest.approx(1.0)

    def test_ndcg_zero(self):
        assert ndcg_at_k(["x", "y"], ["a", "b"], k=2) == 0.0

    def test_recall_empty_relevant(self):
        assert recall_at_k(["a", "b"], [], k=5) == 1.0


# Evaluation Runner 

class TestEvaluationRunner:
    def test_summarize_empty(self, settings: Settings):
        runner = EvaluationRunner(settings)
        summary = runner.summarize([])
        assert summary.total_questions == 0

    def test_load_eval_set_missing_file(self, settings: Settings):
        runner = EvaluationRunner(settings)
        questions = runner.load_eval_set("nonexistent.jsonl")
        assert questions == []

    def test_load_eval_set(self, settings: Settings, tmp_path: Path):
        q = EvalQuestion(
            question="What is the remote work policy?",
            gold_answer="Employees can work remotely.",
            supporting_chunk_ids=["chunk_1"],
            difficulty="simple",
        )
        eval_path = tmp_path / "eval.jsonl"
        eval_path.write_text(q.model_dump_json() + "\n")
        runner = EvaluationRunner(settings)
        loaded = runner.load_eval_set(str(eval_path))
        assert len(loaded) == 1
        assert loaded[0].question == q.question
