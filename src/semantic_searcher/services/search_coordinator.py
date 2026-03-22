"""Distributed search coordinator — fans out queries to worker shards and merges results."""

import asyncio
import logging
import time

import httpx
import numpy as np
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from semantic_searcher.config import settings
from semantic_searcher.database import async_session
from semantic_searcher.models.db import Worker, Shard

log = logging.getLogger(__name__)


class SearchCoordinator:
    """Fans out search queries to distributed workers and merges results."""

    def __init__(self):
        self._workers: dict[str, dict] = {}  # worker_id -> {endpoint_url, api_key_hash, shard_ids}
        self._client: httpx.AsyncClient | None = None
        self._timeout = settings.search_fanout_timeout_ms / 1000.0

    async def start(self):
        """Initialize the HTTP client pool and load worker registry."""
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(connect=2, read=self._timeout, write=5, pool=5),
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=50),
        )
        await self.refresh_workers()

    async def stop(self):
        if self._client:
            await self._client.aclose()

    async def refresh_workers(self):
        """Reload active workers from DB."""
        try:
            async with async_session() as session:
                result = await session.execute(
                    select(Worker).where(Worker.status == "active")
                )
                workers = result.scalars().all()
                self._workers = {
                    w.worker_id: {
                        "endpoint_url": w.endpoint_url,
                        "shard_ids": w.shard_ids or [],
                    }
                    for w in workers
                }
            log.info("SearchCoordinator: %d active workers", len(self._workers))
        except Exception as e:
            log.warning("Failed to refresh workers: %s", e)

    async def fanout_text_search(
        self, query_vec: np.ndarray, query_text: str,
        limit: int = 200, mode: str = "hybrid", filters: dict | None = None,
    ) -> list[dict]:
        """Fan out text search to all workers, merge results.

        Prefers WebSocket tunnel (works behind NAT), falls back to HTTP.
        Returns [{page_id, score, chunk_text}] sorted by score desc.
        """
        tunnel_hub = getattr(self, '_tunnel_hub', None)

        # If tunnel hub has connected workers, use WebSocket (no port forwarding needed)
        if tunnel_hub and tunnel_hub.connected_workers:
            return await tunnel_hub.fanout_search(
                query_vec.tolist(), query_text, limit=limit, mode=mode, filters=filters
            )

        # Fallback: HTTP fanout to workers with known endpoints
        if not self._workers:
            return []

        request_body = {
            "query_vector": query_vec.tolist(),
            "query_text": query_text,
            "limit": limit,
            "mode": mode,
            "filters": filters,
        }

        tasks = []
        worker_ids = []
        for wid, info in self._workers.items():
            url = f"{info['endpoint_url']}/api/worker/search"
            tasks.append(self._query_worker(url, request_body))
            worker_ids.append(wid)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results from all workers
        merged: dict[int, dict] = {}  # page_id -> best result
        for wid, result in zip(worker_ids, results):
            if isinstance(result, Exception):
                log.warning("Worker %s search failed: %s", wid, result)
                continue
            if not result:
                continue
            for item in result.get("text_results", []):
                pid = item["page_id"]
                if pid not in merged or item["score"] > merged[pid]["score"]:
                    merged[pid] = item

        sorted_results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        return sorted_results[:limit]

    async def fanout_image_search(
        self, clip_vec: np.ndarray, query_text: str,
        limit: int = 500, filters: dict | None = None,
    ) -> list[dict]:
        """Fan out image search to all workers, merge results."""
        if not self._workers:
            return []

        request_body = {
            "query_vector": [],  # not used for image search
            "query_text": query_text,
            "clip_vector": clip_vec.tolist(),
            "limit": limit,
            "mode": "image",
            "filters": filters,
        }

        tasks = []
        for wid, info in self._workers.items():
            url = f"{info['endpoint_url']}/api/worker/search"
            tasks.append(self._query_worker(url, request_body))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        merged = []
        for result in results:
            if isinstance(result, Exception) or not result:
                continue
            merged.extend(result.get("image_results", []))

        merged.sort(key=lambda x: x["score"], reverse=True)
        return merged[:limit]

    async def _query_worker(self, url: str, body: dict) -> dict | None:
        """Send search request to a single worker."""
        try:
            resp = await self._client.post(url, json=body)
            if resp.status_code == 200:
                return resp.json()
            log.warning("Worker %s returned %d", url, resp.status_code)
            return None
        except Exception as e:
            raise  # Let gather handle it


class DistributedSearchBackend:
    """SearchBackend implementation that delegates to SearchCoordinator."""

    def __init__(self, coordinator: SearchCoordinator):
        self._coordinator = coordinator

    def text_search(self, query_vec: np.ndarray, query_text: str,
                    limit: int = 2000, qdrant_filter=None) -> list[dict]:
        """Synchronous wrapper for async fanout (called from thread pool)."""
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self._coordinator.fanout_text_search(query_vec, query_text, limit, "hybrid")
            )
        finally:
            loop.close()

    def dense_only_search(self, query_vec: np.ndarray,
                          limit: int = 2000, qdrant_filter=None) -> list[dict]:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self._coordinator.fanout_text_search(query_vec, "", limit, "dense")
            )
        finally:
            loop.close()

    def image_search(self, query_vec: np.ndarray,
                     limit: int = 500, qdrant_filter=None) -> list[dict]:
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(
                self._coordinator.fanout_image_search(query_vec, "", limit)
            )
        finally:
            loop.close()
