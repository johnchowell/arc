"""Handles incoming search requests from the hub on a worker node.

The worker performs local Qdrant queries and returns results to the hub.
"""

import logging
import time

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Document, FieldCondition, Filter, Fusion, FusionQuery,
    MatchAny, MatchValue, Prefetch, Range, SearchParams,
)

log = logging.getLogger(__name__)

TEXT_COLLECTION = "text_chunks_v2"
IMAGES_COLLECTION = "images"


class WorkerSearchHandler:
    """Handles search requests from the hub by querying local Qdrant."""

    def __init__(self, qdrant: QdrantClient):
        self._qdrant = qdrant

    def search(self, query_vector: list[float], query_text: str,
               clip_vector: list[float] | None = None,
               limit: int = 200, mode: str = "hybrid",
               filters: dict | None = None) -> dict:
        """Execute search on local Qdrant shard."""
        t0 = time.time()
        text_results = []
        image_results = []

        # Build filter
        qdrant_filter = self._build_filter(filters) if filters else None

        # Text search
        if mode in ("hybrid", "text", "dense"):
            query_vec = np.array(query_vector, dtype=np.float32)
            if mode == "dense":
                text_results = self._dense_search(query_vec, limit, qdrant_filter)
            else:
                text_results = self._hybrid_search(query_vec, query_text, limit, qdrant_filter)

        # Image search
        if mode == "image" and clip_vector:
            clip_vec = np.array(clip_vector, dtype=np.float32)
            image_results = self._image_search(clip_vec, limit, qdrant_filter)

        elapsed = round((time.time() - t0) * 1000, 2)
        return {
            "text_results": text_results,
            "image_results": image_results,
            "search_time_ms": elapsed,
        }

    def _hybrid_search(self, query_vec: np.ndarray, query_text: str,
                       limit: int, qdrant_filter) -> list[dict]:
        try:
            results = self._qdrant.query_points(
                collection_name=TEXT_COLLECTION,
                prefetch=[
                    Prefetch(query=query_vec.tolist(), using="dense",
                             limit=limit, filter=qdrant_filter),
                    Prefetch(query=Document(text=query_text, model="Qdrant/bm25"),
                             using="text-bm25",
                             limit=min(limit * 2, 4000), filter=qdrant_filter),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
                with_payload=["page_id", "chunk_text"],
            )
            return [
                {"page_id": p.payload["page_id"], "score": p.score, "chunk_text": p.payload["chunk_text"]}
                for p in results.points
            ]
        except Exception as e:
            log.warning("Local hybrid search failed: %s", e)
            return []

    def _dense_search(self, query_vec: np.ndarray, limit: int, qdrant_filter) -> list[dict]:
        try:
            results = self._qdrant.query_points(
                collection_name=TEXT_COLLECTION,
                query=query_vec.tolist(), using="dense",
                query_filter=qdrant_filter, limit=limit,
                with_payload=["page_id", "chunk_text"],
                search_params=SearchParams(exact=False, hnsw_ef=128),
            )
            return [
                {"page_id": p.payload["page_id"], "score": p.score, "chunk_text": p.payload["chunk_text"]}
                for p in results.points
            ]
        except Exception as e:
            log.warning("Local dense search failed: %s", e)
            return []

    def _image_search(self, clip_vec: np.ndarray, limit: int, qdrant_filter) -> list[dict]:
        try:
            results = self._qdrant.query_points(
                collection_name=IMAGES_COLLECTION,
                query=clip_vec.tolist(),
                query_filter=qdrant_filter, limit=limit,
                with_payload=True,
            )
            return [
                {
                    "image_url": p.payload["image_url"],
                    "alt_text": p.payload.get("alt_text", ""),
                    "score": p.score,
                    "page_ids": p.payload.get("page_ids", []),
                    "nsfw_score": p.payload.get("nsfw_score", 0.0),
                    "icon_score": p.payload.get("icon_score", 0.0),
                }
                for p in results.points
            ]
        except Exception as e:
            log.warning("Local image search failed: %s", e)
            return []

    @staticmethod
    def _build_filter(filters: dict) -> Filter | None:
        must = []
        must_not = []
        if filters.get("lang"):
            must.append(FieldCondition(key="language", match=MatchValue(value=filters["lang"])))
        if filters.get("domain"):
            must.append(FieldCondition(key="domain", match=MatchValue(value=filters["domain"])))
        if filters.get("safe_search", True):
            must.append(FieldCondition(key="nsfw_flag", match=MatchValue(value=False)))
        if not filters.get("include_stale", False):
            must_not.append(FieldCondition(key="is_stale", match=MatchValue(value=True)))
        if filters.get("tld_groups"):
            must.append(FieldCondition(key="tld_group", match=MatchAny(any=filters["tld_groups"])))
        if filters.get("categories"):
            must.append(FieldCondition(key="content_category", match=MatchAny(any=filters["categories"])))
        if must or must_not:
            return Filter(must=must or None, must_not=must_not or None)
        return None
