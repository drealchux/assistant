from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Optional

from rank_bm25 import BM25Okapi

from .config import Settings
from .embeddings import EmbeddingModel, VectorStore
from .models import Chunk

logger = logging.getLogger(__name__)


# BM25 Keyword Index

class BM25Index:
    """BM25 keyword retrieval index backed by a persistent pickle file."""

    def __init__(self, settings: Settings):
        self.index_path = Path(settings.bm25_index_path)
        self._corpus: list[dict] = []   # list of chunk payloads
        self._tokenized: list[list[str]] = []
        self._bm25: Optional[BM25Okapi] = None
        self._load_if_exists()

    # Build / Update 

    def build(self, chunk_payloads: list[dict]) -> None:
        """Build the BM25 index from a list of chunk payloads."""
        self._corpus = chunk_payloads
        self._tokenized = [self._tokenize(p["text"]) for p in chunk_payloads]
        self._bm25 = BM25Okapi(self._tokenized)
        self._save()
        logger.info(f"BM25 index built with {len(self._corpus)} chunks")

    def add_chunks(self, chunk_payloads: list[dict]) -> None:
        """Incrementally add chunks to the index (requires full rebuild)."""
        self._corpus.extend(chunk_payloads)
        self._tokenized.extend([self._tokenize(p["text"]) for p in chunk_payloads])
        self._bm25 = BM25Okapi(self._tokenized)
        self._save()

    # Search

    def search(self, query: str, top_k: int = 50) -> list[dict]:
        if not self._bm25 or not self._corpus:
            return []
        tokens = self._tokenize(query)
        scores = self._bm25.get_scores(tokens)
        top_indices = sorted(
            range(len(scores)), key=lambda i: scores[i], reverse=True
        )[:top_k]
        return [
            {**self._corpus[i], "score": float(scores[i])}
            for i in top_indices
            if scores[i] > 0
        ]

    # Persistence

    def _save(self) -> None:
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump({"corpus": self._corpus, "tokenized": self._tokenized}, f)

    def _load_if_exists(self) -> None:
        if self.index_path.exists():
            with open(self.index_path, "rb") as f:
                data = pickle.load(f)
            self._corpus = data["corpus"]
            self._tokenized = data["tokenized"]
            self._bm25 = BM25Okapi(self._tokenized)
            logger.info(f"BM25 index loaded: {len(self._corpus)} chunks")

    @staticmethod
    def _tokenize(text: str) -> list[str]:
        return text.lower().split()


# Reciprocal Rank Fusion

def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    k: int = 60,
) -> list[dict]:
    """Fuse multiple ranked result lists into a single ranking using RRF."""
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for results in result_lists:
        for rank, item in enumerate(results):
            cid = item["chunk_id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            payloads[cid] = item

    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [{**payloads[cid], "score": score} for cid, score in fused]


# Hybrid Retriever

class HybridRetriever:
    """Combines semantic (Qdrant) + keyword (BM25) retrieval with RRF fusion."""

    def __init__(
        self,
        vector_store: VectorStore,
        bm25_index: BM25Index,
        embedder: EmbeddingModel,
        settings: Settings,
    ):
        self.vector_store = vector_store
        self.bm25_index = bm25_index
        self.embedder = embedder
        self.top_k_semantic = settings.top_k_semantic
        self.top_k_keyword = settings.top_k_keyword
        self.rrf_k = settings.rrf_k

    def retrieve(
        self,
        queries: list[str],
        top_k: int = 50,
        acl_groups: Optional[list[str]] = None,
    ) -> list[dict]:
        """
        Multi-query hybrid retrieval:
        1. For each query: semantic + keyword retrieval
        2. Fuse all result lists with RRF
        3. Return top_k deduplicated, ranked chunks
        """
        all_lists: list[list[dict]] = []

        for query in queries:
            # Semantic retrieval
            q_vec = self.embedder.embed_query(query)
            sem_results = self.vector_store.search(
                q_vec, top_k=self.top_k_semantic, acl_groups=acl_groups
            )
            if sem_results:
                all_lists.append(sem_results)

            # Keyword retrieval
            kw_results = self.bm25_index.search(query, top_k=self.top_k_keyword)
            if kw_results:
                all_lists.append(kw_results)

        if not all_lists:
            return []

        fused = reciprocal_rank_fusion(all_lists, k=self.rrf_k)
        return fused[:top_k]
