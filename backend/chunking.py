from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

from .models import Chunk, DocumentMetadata


# Recursive Text Splitter

@dataclass
class RecursiveTextSplitter:
    """Fixed-size chunking with overlap using recursive separator splitting."""

    chunk_size: int = 1000
    chunk_overlap: int = 200
    separators: list[str] = field(
        default_factory=lambda: ["\n\n", "\n", ". ", " ", ""]
    )

    def split_text(self, text: str) -> list[str]:
        return self._split_recursive(text, self.separators)

    def _split_recursive(self, text: str, separators: list[str]) -> list[str]:
        if len(text) <= self.chunk_size:
            return [text.strip()] if text.strip() else []

        separator = separators[0] if separators else ""
        for sep in separators:
            if sep in text:
                separator = sep
                break

        splits = text.split(separator) if separator else list(text)
        chunks: list[str] = []
        current = ""

        for split in splits:
            piece = (current + separator + split).strip() if current else split.strip()
            if len(piece) <= self.chunk_size:
                current = piece
            else:
                if current:
                    chunks.append(current)
                if len(split) > self.chunk_size:
                    sub = self._split_recursive(split, separators[1:] if len(separators) > 1 else [""])
                    if sub:
                        if chunks and self.chunk_overlap > 0:
                            overlap_text = chunks[-1][-self.chunk_overlap:]
                            sub[0] = (overlap_text + " " + sub[0]).strip()
                        chunks.extend(sub[:-1])
                        current = sub[-1]
                    else:
                        current = split
                else:
                    if chunks and self.chunk_overlap > 0:
                        overlap = chunks[-1][-self.chunk_overlap:]
                        current = (overlap + " " + split).strip()
                    else:
                        current = split

        if current:
            chunks.append(current)

        return [c for c in chunks if c.strip()]


# Contextual Header Builder

class ChunkEnricher:
    """Prepends a structured contextual header to each chunk for better retrieval."""

    def build_header(self, metadata: DocumentMetadata, section: Optional[str] = None) -> str:
        parts = [f"Document: {metadata.title}"]
        if section:
            parts.append(f"Section: {section}")
        if metadata.department:
            parts.append(f"Department: {metadata.department}")
        if metadata.country:
            parts.append(f"Country: {metadata.country}")
        if metadata.document_type:
            parts.append(f"Type: {metadata.document_type.upper()}")
        return " | ".join(parts)

    def enrich(self, chunk: Chunk, section: Optional[str] = None) -> Chunk:
        chunk.header = self.build_header(chunk.metadata, section)
        return chunk


# Section Detector 

class SectionDetector:
    """Heuristically detects section headers in plain text."""

    # Common heading patterns in policy docs
    _HEADING_RE = re.compile(
        r"^(\d+\.[\d.]*\s+.+|[A-Z][A-Z\s]{3,}|#{1,6}\s+.+)$",
        re.MULTILINE,
    )

    def detect_section(self, text: str) -> Optional[str]:
        matches = self._HEADING_RE.findall(text[:500])
        return matches[0].strip() if matches else None


# Main Document Chunker 

class DocumentChunker:
    """Splits a document into enriched Chunk objects."""

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.splitter = RecursiveTextSplitter(chunk_size, chunk_overlap)
        self.enricher = ChunkEnricher()
        self.section_detector = SectionDetector()

    def chunk_document(
        self,
        text: str,
        metadata: DocumentMetadata,
    ) -> list[Chunk]:
        raw_chunks = self.splitter.split_text(text)
        chunks: list[Chunk] = []

        for idx, raw in enumerate(raw_chunks):
            section = self.section_detector.detect_section(raw)
            header = self.enricher.build_header(metadata, section)

            # The text stored and embedded includes the header for richer retrieval
            enriched_text = f"{header}\n\n{raw}"

            chunk = Chunk(
                document_id=metadata.document_id,
                text=enriched_text,
                header=header,
                chunk_index=idx,
                metadata=metadata,
            )
            chunks.append(chunk)

        return chunks
