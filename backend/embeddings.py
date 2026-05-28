from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models as qdrant_models
from sentence_transformers import SentenceTransformer

from .config import Settings
from .models import Chunk, DocumentMetadata

logger = logging.getLogger(__name__)


# Embedding Model

class EmbeddingModel:
    """Wraps sentence-transformers for batch text embedding."""

    def __init__(self, settings: Settings):
        self.model_name = settings.embedding_model
        self.batch_size = settings.embedding_batch_size
        self.dim = settings.embedding_dim
        logger.info(f"Loading embedding model: {self.model_name}")
        self._model = SentenceTransformer(self.model_name)

    def embed(self, texts: list[str]) -> np.ndarray:
        """Embed a list of texts, returns (N, dim) float32 array."""
        return self._model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
        )

    def embed_query(self, text: str) -> list[float]:
        return self._model.encode(
            text,
            normalize_embeddings=True,
        ).tolist()


# Vector Store (Qdrant) 

class VectorStore:
    """Qdrant-backed vector store with ACL filtering support."""

    def __init__(self, settings: Settings):
        self.collection = settings.qdrant_collection
        self.dim = settings.embedding_dim
        self._client = QdrantClient(
            url=settings.qdrant_url,
            api_key=settings.qdrant_api_key,
            timeout=30,
        )
        self._ensure_collection()

    # Collection lifecycle 

    def _ensure_collection(self) -> None:
        existing = [c.name for c in self._client.get_collections().collections]
        if self.collection not in existing:
            self._client.create_collection(
                collection_name=self.collection,
                vectors_config=qdrant_models.VectorParams(
                    size=self.dim,
                    distance=qdrant_models.Distance.COSINE,
                ),
            )
            logger.info(f"Created Qdrant collection: {self.collection}")

    def recreate_collection(self) -> None:
        self._client.recreate_collection(
            collection_name=self.collection,
            vectors_config=qdrant_models.VectorParams(
                size=self.dim,
                distance=qdrant_models.Distance.COSINE,
            ),
        )
        logger.info(f"Recreated Qdrant collection: {self.collection}")

    # Upsert 

    def upsert_chunks(self, chunks: list[Chunk], embeddings: np.ndarray) -> None:
        points = [
            qdrant_models.PointStruct(
                id=self._chunk_id_to_int(chunk.chunk_id),
                vector=embeddings[i].tolist(),
                payload=self._chunk_to_payload(chunk),
            )
            for i, chunk in enumerate(chunks)
        ]
        self._client.upsert(collection_name=self.collection, points=points)

    # Search

    def search(
        self,
        query_vector: list[float],
        top_k: int = 50,
        acl_groups: Optional[list[str]] = None,
    ) -> list[dict]:
        """Semantic search with optional ACL group filtering."""
        query_filter = self._build_acl_filter(acl_groups)

        results = self._client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )

        return [
            {
                "chunk_id": r.payload["chunk_id"],
                "text": r.payload["text"],
                "header": r.payload["header"],
                "document_id": r.payload["document_id"],
                "metadata": r.payload["metadata"],
                "score": r.score,
            }
            for r in results
        ]

    def get_all_chunks(self) -> list[dict]:
        """Scroll all chunks (for BM25 index rebuild)."""
        records, _ = self._client.scroll(
            collection_name=self.collection,
            limit=10_000,
            with_payload=True,
            with_vectors=False,
        )
        return [r.payload for r in records]

    def delete_document(self, document_id: str) -> None:
        self._client.delete(
            collection_name=self.collection,
            points_selector=qdrant_models.FilterSelector(
                filter=qdrant_models.Filter(
                    must=[
                        qdrant_models.FieldCondition(
                            key="document_id",
                            match=qdrant_models.MatchValue(value=document_id),
                        )
                    ]
                )
            ),
        )

    def count(self) -> int:
        return self._client.count(collection_name=self.collection).count

    # Helpers 

    def _build_acl_filter(
        self, acl_groups: Optional[list[str]]
    ) -> Optional[qdrant_models.Filter]:
        if not acl_groups:
            return None
        return qdrant_models.Filter(
            should=[
                qdrant_models.FieldCondition(
                    key="metadata.acl_groups",
                    match=qdrant_models.MatchAny(any=acl_groups),
                ),
                # Chunks with empty ACL are public within the system
                qdrant_models.IsEmptyCondition(
                    is_empty=qdrant_models.PayloadField(key="metadata.acl_groups")
                ),
            ]
        )

    @staticmethod
    def _chunk_id_to_int(chunk_id: str) -> int:
        return abs(hash(chunk_id)) % (10**15)

    @staticmethod
    def _chunk_to_payload(chunk: Chunk) -> dict:
        return {
            "chunk_id": chunk.chunk_id,
            "text": chunk.text,
            "header": chunk.header,
            "chunk_index": chunk.chunk_index,
            "document_id": chunk.document_id,
            "metadata": chunk.metadata.model_dump(mode="json"),
        }
