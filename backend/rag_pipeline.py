from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import AsyncIterator, Optional
from uuid import uuid4

import anthropic

from .acl import ACLFilter
from .compression import ContextualCompressor
from .config import Settings
from .embeddings import EmbeddingModel, VectorStore
from .models import (
    Citation,
    CompressedChunk,
    LatencyMetrics,
    QueryResponse,
    RetrievalPlan,
)
from .prompts import (
    ANSWER_GENERATION_SYSTEM,
    ANSWER_GENERATION_USER,
    QUERY_EXPANSION_SYSTEM,
    QUERY_EXPANSION_USER,
)
from .reranker import CrossEncoderReranker
from .retrieval import BM25Index, HybridRetriever

logger = logging.getLogger(__name__)


class RAGPipeline:
    """
    End-to-end RAG pipeline:
    Query → Expansion → Hybrid Retrieval → ACL Filter
        → Reranking → Contextual Compression → Answer Generation
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)

        # Component initialisation
        self.embedder = EmbeddingModel(settings)
        self.vector_store = VectorStore(settings)
        self.bm25_index = BM25Index(settings)
        self.retriever = HybridRetriever(
            self.vector_store, self.bm25_index, self.embedder, settings
        )
        self.reranker = CrossEncoderReranker(settings)
        self.compressor = ContextualCompressor(settings)
        self.acl_filter = ACLFilter()

    # Public API

    async def query(
        self,
        query: str,
        user_id: str,
        user_groups: list[str],
        top_k: int = 5,
    ) -> QueryResponse:
        t_total = time.time()
        query_id = str(uuid4())
        latency = LatencyMetrics()

        # 1. Query expansion
        t0 = time.time()
        rewrites = await self._expand_query(query)
        latency.query_expansion_ms = _ms(t0)

        # 2. Hybrid retrieval (multi-query: original + rewrites)
        t0 = time.time()
        all_queries = [query] + rewrites
        candidates = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.retriever.retrieve(
                all_queries,
                top_k=self.settings.top_k_semantic,
                acl_groups=user_groups or None,
            ),
        )
        latency.retrieval_ms = _ms(t0)
        logger.info(f"[{query_id}] Retrieved {len(candidates)} candidates")

        # 3. ACL filter (defence-in-depth, before reranking)
        candidates = self.acl_filter.filter(candidates, user_groups)

        # 4. Reranking
        if self.settings.use_reranker and candidates:
            t0 = time.time()
            candidates = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.reranker.rerank(
                    query, candidates, top_k=self.settings.top_k_rerank
                ),
            )
            latency.reranking_ms = _ms(t0)
            logger.info(f"[{query_id}] After reranking: {len(candidates)} chunks")

        candidates = candidates[:top_k]

        # 5. Contextual compression
        t0 = time.time()
        compressed = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.compressor.compress(query, candidates),
        )
        latency.compression_ms = _ms(t0)

        # 6. Answer generation
        t0 = time.time()
        answer, citations = await self._generate_answer(query, compressed)
        latency.generation_ms = _ms(t0)

        latency.total_ms = _ms(t_total)

        response = QueryResponse(
            query_id=query_id,
            query=query,
            answer=answer,
            citations=citations,
            query_rewrites=rewrites,
            retrieval_plan=RetrievalPlan(
                mode="hybrid",
                top_k=top_k,
                use_reranking=self.settings.use_reranker,
                use_compression=True,
                queries_used=all_queries,
            ),
            latency_ms=latency,
        )

        await self._log_query(user_id, response)
        return response

    async def stream_query(
        self,
        query: str,
        user_id: str,
        user_groups: list[str],
        top_k: int = 5,
    ) -> AsyncIterator[str]:
        """Yield SSE-formatted tokens as the answer streams."""
        t_total = time.time()
        query_id = str(uuid4())

        rewrites = await self._expand_query(query)
        all_queries = [query] + rewrites

        candidates = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.retriever.retrieve(all_queries, top_k=self.settings.top_k_semantic),
        )
        candidates = self.acl_filter.filter(candidates, user_groups)

        if self.settings.use_reranker and candidates:
            candidates = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self.reranker.rerank(query, candidates, top_k=self.settings.top_k_rerank),
            )

        candidates = candidates[:top_k]
        compressed = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self.compressor.compress(query, candidates),
        )

        context_str = self._build_context_string(compressed)

        # Stream the generation
        with self._client.messages.stream(
            model=self.settings.llm_model,
            max_tokens=self.settings.llm_max_tokens,
            system=[
                {
                    "type": "text",
                    "text": ANSWER_GENERATION_SYSTEM,
                    "cache_control": {"type": "ephemeral"},  # prompt caching
                }
            ],
            messages=[
                {
                    "role": "user",
                    "content": ANSWER_GENERATION_USER.format(
                        query=query, context=context_str
                    ),
                }
            ],
        ) as stream:
            for text in stream.text_stream:
                yield f"data: {json.dumps({'token': text})}\n\n"

        # Final event with citations and metadata
        citations = self._build_citations(compressed)
        yield f"data: {json.dumps({'done': True, 'citations': [c.model_dump() for c in citations], 'query_id': query_id, 'latency_ms': _ms(t_total)})}\n\n"

    # Internal steps

    async def _expand_query(self, query: str) -> list[str]:
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: self._client.messages.create(
                    model=self.settings.llm_model,
                    max_tokens=256,
                    system=QUERY_EXPANSION_SYSTEM,
                    messages=[
                        {
                            "role": "user",
                            "content": QUERY_EXPANSION_USER.format(query=query),
                        }
                    ],
                ),
            )
            raw = response.content[0].text.strip()
            rewrites = json.loads(raw)
            if isinstance(rewrites, list):
                return [str(r) for r in rewrites[:5]]
        except Exception as e:
            logger.warning(f"Query expansion failed: {e}")
        return []

    async def _generate_answer(
        self, query: str, compressed: list[CompressedChunk]
    ) -> tuple[str, list[Citation]]:
        context_str = self._build_context_string(compressed)

        response = await asyncio.get_event_loop().run_in_executor(
            None,
            lambda: self._client.messages.create(
                model=self.settings.llm_model,
                max_tokens=self.settings.llm_max_tokens,
                system=[
                    {
                        "type": "text",
                        "text": ANSWER_GENERATION_SYSTEM,
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                messages=[
                    {
                        "role": "user",
                        "content": ANSWER_GENERATION_USER.format(
                            query=query, context=context_str
                        ),
                    }
                ],
            ),
        )

        answer = response.content[0].text.strip()
        citations = self._build_citations(compressed)
        return answer, citations

    @staticmethod
    def _build_context_string(compressed: list[CompressedChunk]) -> str:
        parts = []
        for i, chunk in enumerate(compressed, 1):
            parts.append(
                f"[{i}] {chunk.header}\n{chunk.relevant_text}"
            )
        return "\n\n---\n\n".join(parts) if parts else "No relevant passages found."

    @staticmethod
    def _build_citations(compressed: list[CompressedChunk]) -> list[Citation]:
        return [
            Citation(
                chunk_id=c.chunk_id,
                document_title=c.document_title,
                section=c.section,
                relevant_text=c.relevant_text[:300],
                source_path="",
                score=c.score,
            )
            for c in compressed
        ]

    async def _log_query(self, user_id: str, response: QueryResponse) -> None:
        log_path = Path(self.settings.query_log_path)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "query_id": response.query_id,
            "user_id": user_id,
            "query": response.query,
            "answer_preview": response.answer[:200],
            "citation_count": len(response.citations),
            "latency_ms": response.latency_ms.model_dump(),
            "timestamp": response.timestamp.isoformat(),
        }
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def _ms(t_start: float) -> int:
    return int((time.time() - t_start) * 1000)
