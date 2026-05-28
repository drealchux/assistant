from __future__ import annotations

from datetime import datetime
from typing import Any, Optional
from uuid import uuid4

from pydantic import BaseModel, Field


# Document & Chunk

class DocumentMetadata(BaseModel):
    document_id: str = Field(default_factory=lambda: str(uuid4()))
    title: str
    source_path: str
    department: Optional[str] = None
    country: Optional[str] = None
    document_type: str  # pdf | docx | confluence | wiki
    author: Optional[str] = None
    last_updated: Optional[datetime] = None
    acl_groups: list[str] = Field(default_factory=list)  # allowed user groups


class Chunk(BaseModel):
    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    document_id: str
    text: str
    header: str  # enriched contextual header prepended at embed time
    chunk_index: int = 0
    metadata: DocumentMetadata
    score: float = 0.0  # retrieval score, populated after retrieval


class CompressedChunk(BaseModel):
    chunk_id: str
    header: str
    relevant_text: str
    original_text: str
    document_title: str
    section: str
    score: float = 0.0


# Query & Response

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    user_id: str
    user_groups: list[str] = Field(default_factory=list)
    top_k: int = Field(default=5, ge=1, le=20)
    stream: bool = False


class Citation(BaseModel):
    chunk_id: str
    document_title: str
    section: str
    relevant_text: str
    source_path: str
    score: float


class RetrievalPlan(BaseModel):
    mode: str = "hybrid"
    top_k: int
    use_reranking: bool
    use_compression: bool
    queries_used: list[str]


class LatencyMetrics(BaseModel):
    query_expansion_ms: int = 0
    retrieval_ms: int = 0
    reranking_ms: int = 0
    compression_ms: int = 0
    generation_ms: int = 0
    total_ms: int = 0


class QueryResponse(BaseModel):
    query_id: str = Field(default_factory=lambda: str(uuid4()))
    query: str
    answer: str
    citations: list[Citation]
    query_rewrites: list[str]
    retrieval_plan: RetrievalPlan
    latency_ms: LatencyMetrics
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Feedback

class FeedbackRating(str):
    HELPFUL = "helpful"
    INCORRECT = "incorrect"
    INCOMPLETE = "incomplete"
    MISSING_DOC = "missing_doc"


class FeedbackRequest(BaseModel):
    query_id: str
    user_id: str
    rating: str  # helpful | incorrect | incomplete | missing_doc
    comment: Optional[str] = None


class FeedbackRecord(BaseModel):
    feedback_id: str = Field(default_factory=lambda: str(uuid4()))
    query_id: str
    user_id: str
    rating: str
    comment: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)


# Ingestion

class IngestRequest(BaseModel):
    source_path: str
    document_type: str  # pdf | docx | confluence | wiki
    title: Optional[str] = None
    department: Optional[str] = None
    country: Optional[str] = None
    author: Optional[str] = None
    acl_groups: list[str] = Field(default_factory=list)


class IngestResponse(BaseModel):
    document_id: str
    title: str
    chunks_created: int
    status: str  # success | failed
    message: str = ""


# Documents

class DocumentInfo(BaseModel):
    document_id: str
    title: str
    source_path: str
    department: Optional[str] = None
    country: Optional[str] = None
    document_type: str
    author: Optional[str] = None
    last_updated: Optional[datetime] = None
    chunk_count: int = 0


# Evaluation

class EvalQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str
    gold_answer: str
    supporting_chunk_ids: list[str]
    difficulty: str  # simple | multi_section | ambiguous
    department: Optional[str] = None


class EvalResult(BaseModel):
    question_id: str
    question: str
    predicted_answer: str
    gold_answer: str
    retrieved_chunk_ids: list[str]
    recall_at_5: float
    recall_at_10: float
    precision_at_5: float
    faithfulness_score: float
    correctness_score: float
    latency_ms: int


class EvalSummary(BaseModel):
    total_questions: int
    avg_recall_at_5: float
    avg_recall_at_10: float
    avg_precision_at_5: float
    avg_faithfulness: float
    avg_correctness: float
    avg_latency_ms: float
    hallucination_rate: float


# Health

class HealthResponse(BaseModel):
    status: str  # healthy | degraded | unhealthy
    version: str
    components: dict[str, str]
    timestamp: datetime = Field(default_factory=datetime.utcnow)
