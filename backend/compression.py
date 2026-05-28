from __future__ import annotations

import logging

import anthropic

from .config import Settings
from .models import CompressedChunk
from .prompts import COMPRESSION_SYSTEM, COMPRESSION_USER

logger = logging.getLogger(__name__)


class ContextualCompressor:
    """
    Extracts only the query-relevant sentences from each retrieved chunk.
    Reduces noise in the generation context while preserving grounding.
    """

    def __init__(self, settings: Settings):
        self._client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self._model = settings.llm_model

    def compress(
        self, query: str, chunks: list[dict]
    ) -> list[CompressedChunk]:
        compressed: list[CompressedChunk] = []

        for chunk in chunks:
            relevant_text = self._extract_relevant(query, chunk["text"])

            # If nothing is relevant, skip this chunk
            if not relevant_text.strip():
                continue

            meta = chunk.get("metadata", {})
            compressed.append(
                CompressedChunk(
                    chunk_id=chunk["chunk_id"],
                    header=chunk.get("header", ""),
                    relevant_text=relevant_text,
                    original_text=chunk["text"],
                    document_title=meta.get("title", "Unknown Document"),
                    section=self._extract_section(chunk.get("header", "")),
                    score=chunk.get("score", 0.0),
                )
            )

        return compressed

    def _extract_relevant(self, query: str, chunk_text: str) -> str:
        try:
            response = self._client.messages.create(
                model=self._model,
                max_tokens=512,
                system=COMPRESSION_SYSTEM,
                messages=[
                    {
                        "role": "user",
                        "content": COMPRESSION_USER.format(
                            query=query, chunk_text=chunk_text
                        ),
                    }
                ],
            )
            return response.content[0].text.strip()
        except Exception as e:
            logger.warning(f"Compression failed for chunk, using full text: {e}")
            # Fall back to full text on error
            return chunk_text

    @staticmethod
    def _extract_section(header: str) -> str:
        """Parse section name from contextual header string."""
        for part in header.split("|"):
            part = part.strip()
            if part.startswith("Section:"):
                return part.replace("Section:", "").strip()
        return "General"
