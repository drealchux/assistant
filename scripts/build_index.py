#!/usr/bin/env python3
"""
Build or rebuild the vector index and BM25 index.

Usage:
    python -m scripts.build_index --source data/raw_docs --department HR --country Norway
    python -m scripts.build_index --rebuild-bm25
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from backend.config import get_settings
from ingestion.ingest_pdfs import IngestionPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the RAG vector index")
    parser.add_argument("--source", help="Directory containing documents to ingest")
    parser.add_argument("--file", help="Single file to ingest")
    parser.add_argument("--type", default="pdf", help="Document type (pdf|docx|wiki)")
    parser.add_argument("--department", help="Department tag for all documents")
    parser.add_argument("--country", help="Country tag for all documents")
    parser.add_argument("--acl-groups", nargs="*", default=[], help="ACL groups")
    parser.add_argument("--rebuild-bm25", action="store_true", help="Rebuild BM25 from Qdrant")
    parser.add_argument("--recreate-collection", action="store_true",
                        help="Drop and recreate the Qdrant collection (WARNING: destructive)")
    args = parser.parse_args()

    settings = get_settings()
    pipeline = IngestionPipeline(settings)

    if args.recreate_collection:
        confirm = input("This will DELETE all indexed data. Type 'yes' to confirm: ")
        if confirm.strip().lower() != "yes":
            print("Aborted.")
            return
        pipeline.vector_store.recreate_collection()
        logger.info("Collection recreated.")

    if args.rebuild_bm25:
        pipeline.rebuild_bm25_index()
        return

    if args.source:
        results = pipeline.ingest_directory(
            directory=args.source,
            department=args.department,
            country=args.country,
            acl_groups=args.acl_groups,
        )
        success = sum(1 for r in results if r.status == "success")
        total_chunks = sum(r.chunks_created for r in results)
        print(f"\nIngestion complete: {success}/{len(results)} documents, {total_chunks} chunks")
    elif args.file:
        result = pipeline.ingest_file(
            source_path=args.file,
            document_type=args.type,
            department=args.department,
            country=args.country,
            acl_groups=args.acl_groups,
        )
        print(f"\nResult: {result.status} — {result.chunks_created} chunks created")
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
