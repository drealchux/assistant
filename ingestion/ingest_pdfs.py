from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Optional

import numpy as np
from tqdm import tqdm

from backend.chunking import DocumentChunker
from backend.config import Settings
from backend.embeddings import EmbeddingModel, VectorStore
from backend.models import DocumentMetadata, IngestResponse
from backend.retrieval import BM25Index
from .loaders import get_loader
from .text_extraction import MetadataExtractor, TextCleaner

logger = logging.getLogger(__name__)


class IngestionPipeline:
    """
    End-to-end document ingestion:
    Load → Clean → Chunk → Embed → Index (Qdrant + BM25)
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder = EmbeddingModel(settings)
        self.vector_store = VectorStore(settings)
        self.bm25_index = BM25Index(settings)
        self.chunker = DocumentChunker(settings.chunk_size, settings.chunk_overlap)
        self.cleaner = TextCleaner()
        self.meta_extractor = MetadataExtractor()

    # Single file ingestion

    def ingest_file(
        self,
        source_path: str,
        document_type: str,
        title: Optional[str] = None,
        department: Optional[str] = None,
        country: Optional[str] = None,
        author: Optional[str] = None,
        acl_groups: Optional[list[str]] = None,
        **loader_kwargs,
    ) -> IngestResponse:
        logger.info(f"Ingesting: {source_path} [{document_type}]")

        # 1. Load raw pages
        loader = get_loader(document_type, **loader_kwargs)
        pages = loader.load(source_path)
        if not pages:
            return IngestResponse(
                document_id="",
                title=title or source_path,
                chunks_created=0,
                status="failed",
                message="No content extracted",
            )

        # 2. Clean and join all pages into one document string
        full_text = "\n\n".join(
            self.cleaner.clean(p["text"]) for p in pages if p.get("text")
        )
        if not full_text.strip():
            return IngestResponse(
                document_id="",
                title=title or source_path,
                chunks_created=0,
                status="failed",
                message="Empty document after cleaning",
            )

        # 3. Build metadata
        meta_dict = self.meta_extractor.extract(
            source_path=source_path,
            document_type=document_type,
            page_dicts=pages,
            title=title,
            department=department,
            country=country,
            author=author,
            acl_groups=acl_groups,
        )
        metadata = DocumentMetadata(**meta_dict)

        # 4. Chunk
        chunks = self.chunker.chunk_document(full_text, metadata)
        logger.info(f"Created {len(chunks)} chunks for '{metadata.title}'")

        # 5. Embed
        texts = [c.text for c in chunks]
        embeddings: np.ndarray = self.embedder.embed(texts)

        # 6. Upsert to Qdrant
        self.vector_store.upsert_chunks(chunks, embeddings)

        # 7. Update BM25 index
        chunk_payloads = [
            {
                "chunk_id": c.chunk_id,
                "text": c.text,
                "header": c.header,
                "document_id": c.document_id,
                "metadata": c.metadata.model_dump(mode="json"),
            }
            for c in chunks
        ]
        self.bm25_index.add_chunks(chunk_payloads)

        # 8. Persist processed chunks
        self._save_chunks(chunk_payloads)

        return IngestResponse(
            document_id=metadata.document_id,
            title=metadata.title,
            chunks_created=len(chunks),
            status="success",
        )

    # Batch directory ingestion

    def ingest_directory(
        self,
        directory: str,
        department: Optional[str] = None,
        country: Optional[str] = None,
        acl_groups: Optional[list[str]] = None,
    ) -> list[IngestResponse]:
        results = []
        supported_extensions = {".pdf", ".docx", ".html", ".htm"}
        ext_to_type = {".pdf": "pdf", ".docx": "docx", ".html": "wiki", ".htm": "wiki"}

        files = [
            f for f in Path(directory).rglob("*")
            if f.suffix.lower() in supported_extensions
        ]

        for fp in tqdm(files, desc="Ingesting documents"):
            doc_type = ext_to_type[fp.suffix.lower()]
            result = self.ingest_file(
                source_path=str(fp),
                document_type=doc_type,
                department=department,
                country=country,
                acl_groups=acl_groups,
            )
            results.append(result)

        logger.info(
            f"Batch ingestion complete: {sum(1 for r in results if r.status == 'success')} "
            f"/ {len(results)} succeeded"
        )
        return results

    # Rebuild BM25 from Qdrant

    def rebuild_bm25_index(self) -> None:
        logger.info("Rebuilding BM25 index from Qdrant...")
        payloads = self.vector_store.get_all_chunks()
        self.bm25_index.build(payloads)
        logger.info(f"BM25 index rebuilt: {len(payloads)} chunks")

    # Helpers

    def _save_chunks(self, chunk_payloads: list[dict]) -> None:
        path = Path(self.settings.chunks_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            for p in chunk_payloads:
                f.write(json.dumps(p) + "\n")
