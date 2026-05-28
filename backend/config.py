from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Application 
    app_name: str = "Company Knowledge Copilot"
    app_version: str = "1.0.0"
    debug: bool = False
    log_level: str = "INFO"
    cors_origins: list[str] = ["http://localhost:3000", "http://localhost:8000"]

    # Anthropic / LLM
    anthropic_api_key: str = ""
    llm_model: str = "claude-sonnet-4-6"
    llm_max_tokens: int = 2048
    llm_temperature: float = 0.1

    # Embeddings
    embedding_model: str = "all-MiniLM-L6-v2"
    embedding_dim: int = 384
    embedding_batch_size: int = 64

    # Qdrant
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "company_knowledge"
    qdrant_api_key: Optional[str] = None

    # Retrieval
    top_k_semantic: int = 50
    top_k_keyword: int = 50
    top_k_rerank: int = 10
    top_k_final: int = 5
    rrf_k: int = 60  # reciprocal rank fusion constant

    # Chunking
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Reranker
    reranker_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    use_reranker: bool = True

    # Confluence (optional)
    confluence_url: Optional[str] = None
    confluence_username: Optional[str] = None
    confluence_api_token: Optional[str] = None
    confluence_space_key: Optional[str] = None

    # File paths 
    bm25_index_path: str = "data/bm25_index.pkl"
    feedback_path: str = "data/feedback.jsonl"
    query_log_path: str = "data/query_log.jsonl"
    chunks_path: str = "data/processed_chunks.jsonl"
    eval_set_path: str = "data/eval_set.jsonl"

    # Performance
    query_timeout_seconds: int = 30

    class Config:
        env_file = ".env"
        case_sensitive = False


@lru_cache()
def get_settings() -> Settings:
    return Settings()


settings = get_settings()
