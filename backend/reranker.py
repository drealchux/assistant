from __future__ import annotations

import logging

from sentence_transformers import CrossEncoder

from .config import Settings

logger = logging.getLogger(__name__)


class CrossEncoderReranker:
    """
    Reranks retrieved chunks using a cross-encoder model.
    Cross-encoders consider the query and passage jointly, giving
    much higher precision than bi-encoder similarity alone.
    """

    def __init__(self, settings: Settings):
        self.model_name = settings.reranker_model
        logger.info(f"Loading reranker model: {self.model_name}")
        self._model = CrossEncoder(self.model_name, max_length=512)

    def rerank(
        self,
        query: str,
        candidates: list[dict],
        top_k: int = 10,
    ) -> list[dict]:
        """
        Score each (query, chunk_text) pair and return top_k chunks
        sorted by cross-encoder score descending.
        """
        if not candidates:
            return []

        pairs = [(query, c["text"]) for c in candidates]
        scores = self._model.predict(pairs, show_progress_bar=False)

        ranked = sorted(
            zip(candidates, scores),
            key=lambda x: float(x[1]),
            reverse=True,
        )

        return [
            {**chunk, "score": float(score)}
            for chunk, score in ranked[:top_k]
        ]
