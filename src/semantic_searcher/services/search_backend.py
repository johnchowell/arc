"""Pluggable search backend — local Qdrant or distributed fanout."""

import logging
import time
from typing import Protocol

import numpy as np

log = logging.getLogger(__name__)


class SearchBackend(Protocol):
    """Protocol for search data sources."""

    def text_search(self, query_vec: np.ndarray, query_text: str,
                    limit: int, qdrant_filter) -> list[dict]:
        """Hybrid search (dense + BM25). Returns [{page_id, score, chunk_text}]."""
        ...

    def dense_only_search(self, query_vec: np.ndarray,
                          limit: int, qdrant_filter) -> list[dict]:
        """Dense-only search (fast). Returns [{page_id, score, chunk_text}]."""
        ...

    def image_search(self, query_vec: np.ndarray,
                     limit: int, qdrant_filter) -> list[dict]:
        """Image search. Returns [{id, score, vector, image_url, ...}]."""
        ...


class LocalSearchBackend:
    """Wraps the existing Qdrant client for local search."""

    def __init__(self, qdrant_client, text_collection: str, images_collection: str):
        from qdrant_client.models import (
            Document, Fusion, FusionQuery, Prefetch, SearchParams,
        )
        self._qdrant = qdrant_client
        self._text_collection = text_collection
        self._images_collection = images_collection
        # Store imports for use in methods
        self._Document = Document
        self._Fusion = Fusion
        self._FusionQuery = FusionQuery
        self._Prefetch = Prefetch
        self._SearchParams = SearchParams

    def text_search(self, query_vec: np.ndarray, query_text: str,
                    limit: int = 2000, qdrant_filter=None) -> list[dict]:
        if self._qdrant is None:
            return []
        try:
            results = self._qdrant.query_points(
                collection_name=self._text_collection,
                prefetch=[
                    self._Prefetch(
                        query=query_vec.tolist(), using="dense",
                        limit=limit, filter=qdrant_filter,
                    ),
                    self._Prefetch(
                        query=self._Document(text=query_text, model="Qdrant/bm25"),
                        using="text-bm25",
                        limit=min(limit * 2, 4000), filter=qdrant_filter,
                    ),
                ],
                query=self._FusionQuery(fusion=self._Fusion.RRF),
                limit=limit,
                with_payload=["page_id", "chunk_text"],
            )
            return [
                {"page_id": p.payload["page_id"], "score": p.score, "chunk_text": p.payload["chunk_text"]}
                for p in results.points
            ]
        except Exception as e:
            log.warning("Qdrant hybrid search failed: %s", e)
            return []

    def dense_only_search(self, query_vec: np.ndarray,
                          limit: int = 2000, qdrant_filter=None) -> list[dict]:
        if self._qdrant is None:
            return []
        try:
            results = self._qdrant.query_points(
                collection_name=self._text_collection,
                query=query_vec.tolist(), using="dense",
                query_filter=qdrant_filter, limit=limit,
                with_payload=["page_id", "chunk_text"],
                search_params=self._SearchParams(exact=False, hnsw_ef=128),
            )
            return [
                {"page_id": p.payload["page_id"], "score": p.score, "chunk_text": p.payload["chunk_text"]}
                for p in results.points
            ]
        except Exception as e:
            log.warning("Qdrant dense search failed: %s", e)
            return []

    def image_search(self, query_vec: np.ndarray,
                     limit: int = 500, qdrant_filter=None) -> list[dict]:
        if self._qdrant is None:
            return []
        try:
            results = self._qdrant.query_points(
                collection_name=self._images_collection,
                query=query_vec.tolist(),
                query_filter=qdrant_filter, limit=limit,
                with_payload=True, with_vectors=True,
            )
            return [
                {
                    "id": p.id, "score": p.score,
                    "vector": np.array(p.vector, dtype=np.float32),
                    "image_url": p.payload["image_url"],
                    "alt_text": p.payload.get("alt_text", ""),
                    "nsfw_score": p.payload.get("nsfw_score", 0.0),
                    "icon_score": p.payload.get("icon_score", 0.0),
                    "page_ids": p.payload.get("page_ids", []),
                    "domain": p.payload.get("domain"),
                    "tld_group": p.payload.get("tld_group"),
                    "content_category": p.payload.get("content_category"),
                }
                for p in results.points
            ]
        except Exception as e:
            log.warning("Qdrant image search failed: %s", e)
            return []
