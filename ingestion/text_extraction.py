from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

logger = logging.getLogger(__name__)


class TextCleaner:
    """Normalises extracted text before chunking."""

    def clean(self, text: str) -> str:
        # Normalise whitespace
        text = re.sub(r"\r\n", "\n", text)
        text = re.sub(r"[ \t]+", " ", text)
        # Collapse 3+ blank lines to 2
        text = re.sub(r"\n{3,}", "\n\n", text)
        # Remove page break markers
        text = re.sub(r"\f", "\n\n", text)
        return text.strip()


class MetadataExtractor:
    """Extracts document metadata from file paths and page content."""

    def extract(
        self,
        source_path: str,
        document_type: str,
        page_dicts: list[dict],
        title: Optional[str] = None,
        department: Optional[str] = None,
        country: Optional[str] = None,
        author: Optional[str] = None,
        acl_groups: Optional[list[str]] = None,
    ) -> dict:
        p = Path(source_path)
        inferred_title = title or p.stem.replace("_", " ").replace("-", " ").title()
        inferred_author = author or self._extract_author(page_dicts)
        last_updated = self._extract_timestamp(page_dicts)

        return {
            "document_id": str(uuid4()),
            "title": inferred_title,
            "source_path": source_path,
            "department": department,
            "country": country,
            "document_type": document_type,
            "author": inferred_author,
            "last_updated": last_updated,
            "acl_groups": acl_groups or [],
        }

    @staticmethod
    def _extract_author(page_dicts: list[dict]) -> Optional[str]:
        for page in page_dicts[:1]:
            if "author" in page and page["author"]:
                return page["author"]
        return None

    @staticmethod
    def _extract_timestamp(page_dicts: list[dict]) -> Optional[datetime]:
        for page in page_dicts[:1]:
            raw = page.get("last_updated")
            if raw:
                try:
                    return datetime.fromisoformat(raw.replace("Z", "+00:00"))
                except (ValueError, AttributeError):
                    pass
        return None
