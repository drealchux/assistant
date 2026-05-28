from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class ACLFilter:
    """
    Enforces document-level access control.
    A chunk is accessible if:
      - its acl_groups list is empty (public within the system), OR
      - the user belongs to at least one of the chunk's acl_groups.

    Filtering happens BEFORE reranking and generation to ensure
    zero unauthorized document exposure.
    """

    def filter(
        self,
        chunks: list[dict],
        user_groups: list[str],
    ) -> list[dict]:
        if not user_groups:
            # No groups specified → only return public chunks
            return [c for c in chunks if not self._get_acl(c)]

        allowed = []
        for chunk in chunks:
            acl = self._get_acl(chunk)
            if not acl:
                # Public chunk
                allowed.append(chunk)
            elif any(g in acl for g in user_groups):
                allowed.append(chunk)
            else:
                logger.debug(
                    f"ACL blocked chunk {chunk.get('chunk_id')} "
                    f"(requires {acl}, user has {user_groups})"
                )

        return allowed

    @staticmethod
    def _get_acl(chunk: dict) -> list[str]:
        meta = chunk.get("metadata", {})
        if isinstance(meta, dict):
            return meta.get("acl_groups", [])
        return []
