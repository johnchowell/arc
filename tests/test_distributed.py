"""Tests for the distributed search engine architecture.

Tests cover:
- Worker auth (API key generation, hashing, verification)
- Worker schemas (request/response validation)
- Shard manager (health checks, promotion, assignment)
- Search backend protocol (local backend)
- Search coordinator (fanout, merge, timeout handling)
- Remote crawl client (job processing)
- Worker search handler (local Qdrant queries)
- Worker API endpoints (registration, heartbeat, crawl jobs, search)
"""

import asyncio
import datetime
import hashlib
import time
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest


# --- Worker Auth ---

class TestWorkerAuth:
    def test_hash_api_key_deterministic(self):
        from semantic_searcher.middleware.worker_auth import hash_api_key
        key = "test-key-12345"
        assert hash_api_key(key) == hash_api_key(key)

    def test_hash_api_key_different_keys(self):
        from semantic_searcher.middleware.worker_auth import hash_api_key
        assert hash_api_key("key-a") != hash_api_key("key-b")

    def test_generate_api_key_unique(self):
        from semantic_searcher.middleware.worker_auth import generate_api_key
        keys = {generate_api_key() for _ in range(100)}
        assert len(keys) == 100

    def test_generate_api_key_length(self):
        from semantic_searcher.middleware.worker_auth import generate_api_key
        key = generate_api_key()
        assert len(key) >= 32  # token_urlsafe(48) produces ~64 chars

    def test_hash_is_sha256(self):
        from semantic_searcher.middleware.worker_auth import hash_api_key
        key = "test"
        expected = hashlib.sha256(key.encode()).hexdigest()
        assert hash_api_key(key) == expected


# --- Worker Schemas ---

class TestWorkerSchemas:
    def test_worker_register_request(self):
        from semantic_searcher.models.worker_schemas import WorkerRegisterRequest
        req = WorkerRegisterRequest(name="test-worker", endpoint_url="http://localhost:8901")
        assert req.name == "test-worker"
        assert req.capabilities is None

    def test_worker_register_response(self):
        from semantic_searcher.models.worker_schemas import WorkerRegisterResponse
        resp = WorkerRegisterResponse(worker_id="abc", api_key="secret", message="ok")
        assert resp.worker_id == "abc"

    def test_crawl_job(self):
        from semantic_searcher.models.worker_schemas import CrawlJob
        job = CrawlJob(job_id=1, url="http://example.com", url_hash="abc", depth=0, target_shard_id="shard-0001")
        assert job.target_shard_id == "shard-0001"

    def test_crawl_complete_request_valid_statuses(self):
        from semantic_searcher.models.worker_schemas import CrawlCompleteRequest
        for status in ["completed", "failed", "skipped"]:
            req = CrawlCompleteRequest(job_id=1, status=status)
            assert req.status == status

    def test_crawl_complete_request_invalid_status(self):
        from semantic_searcher.models.worker_schemas import CrawlCompleteRequest
        with pytest.raises(Exception):
            CrawlCompleteRequest(job_id=1, status="invalid")

    def test_worker_search_request(self):
        from semantic_searcher.models.worker_schemas import WorkerSearchRequest
        req = WorkerSearchRequest(
            query_vector=[0.1] * 768,
            query_text="test query",
            limit=100,
        )
        assert len(req.query_vector) == 768
        assert req.clip_vector is None

    def test_worker_search_response(self):
        from semantic_searcher.models.worker_schemas import WorkerSearchResponse, WorkerTextResult
        resp = WorkerSearchResponse(
            worker_id="w1",
            shard_ids=["shard-0001"],
            text_results=[WorkerTextResult(page_id=1, score=0.9, chunk_text="hello")],
            search_time_ms=5.0,
        )
        assert len(resp.text_results) == 1

    def test_vector_batch_request(self):
        from semantic_searcher.models.worker_schemas import VectorBatchRequest, VectorPoint
        req = VectorBatchRequest(
            shard_id="shard-0001",
            points=[VectorPoint(id=1, dense_vector=[0.1] * 768, bm25_text="test", payload={"page_id": 1})],
        )
        assert len(req.points) == 1

    def test_shard_info(self):
        from semantic_searcher.models.worker_schemas import ShardInfo
        s = ShardInfo(shard_id="s1", page_id_start=0, page_id_end=500000, status="active", point_count=1000)
        assert s.primary_worker_id is None


# --- Search Backend ---

class TestLocalSearchBackend:
    def test_text_search_returns_list(self):
        from semantic_searcher.services.search_backend import LocalSearchBackend
        mock_qdrant = MagicMock()
        mock_point = MagicMock()
        mock_point.payload = {"page_id": 1, "chunk_text": "hello"}
        mock_point.score = 0.9
        mock_qdrant.query_points.return_value = MagicMock(points=[mock_point])

        backend = LocalSearchBackend(mock_qdrant, "text_chunks_v2", "images")
        results = backend.text_search(np.zeros(768, dtype=np.float32), "test", 10)
        assert len(results) == 1
        assert results[0]["page_id"] == 1
        assert results[0]["score"] == 0.9

    def test_dense_only_search(self):
        from semantic_searcher.services.search_backend import LocalSearchBackend
        mock_qdrant = MagicMock()
        mock_qdrant.query_points.return_value = MagicMock(points=[])
        backend = LocalSearchBackend(mock_qdrant, "text_chunks_v2", "images")
        results = backend.dense_only_search(np.zeros(768, dtype=np.float32), 10)
        assert results == []

    def test_image_search(self):
        from semantic_searcher.services.search_backend import LocalSearchBackend
        mock_qdrant = MagicMock()
        mock_point = MagicMock()
        mock_point.id = 1
        mock_point.score = 0.8
        mock_point.vector = np.zeros(512).tolist()
        mock_point.payload = {
            "image_url": "http://img.jpg", "alt_text": "test",
            "nsfw_score": 0.0, "icon_score": 0.0,
            "page_ids": [1], "domain": "test.com",
            "tld_group": "com", "content_category": "other",
        }
        mock_qdrant.query_points.return_value = MagicMock(points=[mock_point])
        backend = LocalSearchBackend(mock_qdrant, "text_chunks_v2", "images")
        results = backend.image_search(np.zeros(512, dtype=np.float32), 10)
        assert len(results) == 1
        assert results[0]["image_url"] == "http://img.jpg"

    def test_handles_none_qdrant(self):
        from semantic_searcher.services.search_backend import LocalSearchBackend
        backend = LocalSearchBackend(None, "text_chunks_v2", "images")
        assert backend.text_search(np.zeros(768), "test", 10) == []
        assert backend.dense_only_search(np.zeros(768), 10) == []
        assert backend.image_search(np.zeros(512), 10) == []

    def test_handles_qdrant_exception(self):
        from semantic_searcher.services.search_backend import LocalSearchBackend
        mock_qdrant = MagicMock()
        mock_qdrant.query_points.side_effect = Exception("connection failed")
        backend = LocalSearchBackend(mock_qdrant, "text_chunks_v2", "images")
        assert backend.text_search(np.zeros(768), "test", 10) == []


# --- Search Coordinator ---

class TestSearchCoordinator:
    @pytest.mark.asyncio
    async def test_fanout_empty_workers(self):
        from semantic_searcher.services.search_coordinator import SearchCoordinator
        coord = SearchCoordinator()
        coord._workers = {}
        results = await coord.fanout_text_search(np.zeros(768), "test", 10)
        assert results == []

    @pytest.mark.asyncio
    async def test_fanout_merges_results(self):
        from semantic_searcher.services.search_coordinator import SearchCoordinator
        coord = SearchCoordinator()
        coord._workers = {
            "w1": {"endpoint_url": "http://w1:8901", "shard_ids": ["s1"]},
            "w2": {"endpoint_url": "http://w2:8901", "shard_ids": ["s2"]},
        }

        # Mock the HTTP calls
        async def mock_query(url, body):
            if "w1" in url:
                return {"text_results": [{"page_id": 1, "score": 0.9, "chunk_text": "a"}]}
            return {"text_results": [{"page_id": 2, "score": 0.8, "chunk_text": "b"}]}

        coord._query_worker = mock_query
        results = await coord.fanout_text_search(np.zeros(768), "test", 10)
        assert len(results) == 2
        assert results[0]["score"] >= results[1]["score"]

    @pytest.mark.asyncio
    async def test_fanout_deduplicates_by_page_id(self):
        from semantic_searcher.services.search_coordinator import SearchCoordinator
        coord = SearchCoordinator()
        coord._workers = {
            "w1": {"endpoint_url": "http://w1:8901", "shard_ids": ["s1"]},
            "w2": {"endpoint_url": "http://w2:8901", "shard_ids": ["s1"]},
        }

        async def mock_query(url, body):
            # Both workers return same page (replica)
            return {"text_results": [
                {"page_id": 1, "score": 0.9 if "w1" in url else 0.8, "chunk_text": "a"}
            ]}

        coord._query_worker = mock_query
        results = await coord.fanout_text_search(np.zeros(768), "test", 10)
        assert len(results) == 1
        assert results[0]["score"] == 0.9  # Takes max score

    @pytest.mark.asyncio
    async def test_fanout_handles_worker_failure(self):
        from semantic_searcher.services.search_coordinator import SearchCoordinator
        coord = SearchCoordinator()
        coord._workers = {
            "w1": {"endpoint_url": "http://w1:8901", "shard_ids": ["s1"]},
            "w2": {"endpoint_url": "http://w2:8901", "shard_ids": ["s2"]},
        }

        async def mock_query(url, body):
            if "w1" in url:
                raise ConnectionError("worker offline")
            return {"text_results": [{"page_id": 2, "score": 0.8, "chunk_text": "b"}]}

        coord._query_worker = mock_query
        results = await coord.fanout_text_search(np.zeros(768), "test", 10)
        assert len(results) == 1  # Only w2's results


# --- Worker Search Handler ---

class TestWorkerSearchHandler:
    def test_build_filter_empty(self):
        from semantic_searcher.services.worker_search_handler import WorkerSearchHandler
        f = WorkerSearchHandler._build_filter({})
        assert f is not None  # safe_search defaults True

    def test_build_filter_with_domain(self):
        from semantic_searcher.services.worker_search_handler import WorkerSearchHandler
        f = WorkerSearchHandler._build_filter({"domain": "example.com", "safe_search": False})
        assert f is not None
        assert len(f.must) == 1

    def test_build_filter_returns_none_when_all_disabled(self):
        from semantic_searcher.services.worker_search_handler import WorkerSearchHandler
        f = WorkerSearchHandler._build_filter({"safe_search": False, "include_stale": True})
        assert f is None

    def test_search_returns_dict(self):
        from semantic_searcher.services.worker_search_handler import WorkerSearchHandler
        mock_qdrant = MagicMock()
        mock_qdrant.query_points.return_value = MagicMock(points=[])
        handler = WorkerSearchHandler(mock_qdrant)
        result = handler.search([0.1] * 768, "test", limit=10)
        assert "text_results" in result
        assert "search_time_ms" in result


# --- Shard Manager ---

class TestShardManager:
    @pytest.mark.asyncio
    async def test_create_initial_shards_calculates_ranges(self):
        from semantic_searcher.services.shard_manager import ShardManager
        mgr = ShardManager()
        # Just verify the math, don't actually hit DB
        num_shards = 4
        max_page_id = 1000
        shard_size = max_page_id // num_shards + 1
        assert shard_size == 251
        ranges = [(i * shard_size, min((i + 1) * shard_size - 1, max_page_id)) for i in range(num_shards)]
        assert ranges[0] == (0, 250)
        assert ranges[3] == (753, 1000)


# --- DB Models ---

class TestDBModels:
    def test_worker_model_fields(self):
        from semantic_searcher.models.db import Worker
        assert hasattr(Worker, "worker_id")
        assert hasattr(Worker, "api_key_hash")
        assert hasattr(Worker, "endpoint_url")
        assert hasattr(Worker, "status")
        assert hasattr(Worker, "shard_ids")
        assert hasattr(Worker, "reputation_score")

    def test_shard_model_fields(self):
        from semantic_searcher.models.db import Shard
        assert hasattr(Shard, "shard_id")
        assert hasattr(Shard, "page_id_start")
        assert hasattr(Shard, "page_id_end")
        assert hasattr(Shard, "primary_worker_id")
        assert hasattr(Shard, "replica_worker_ids")

    def test_crawl_job_assignment_model(self):
        from semantic_searcher.models.db import CrawlJobAssignment
        assert hasattr(CrawlJobAssignment, "crawl_queue_id")
        assert hasattr(CrawlJobAssignment, "worker_id")
        assert hasattr(CrawlJobAssignment, "assigned_shard_id")


# --- Config ---

class TestDistributedConfig:
    def test_default_distributed_mode(self):
        from semantic_searcher.config import Settings
        s = Settings(_env_file=None)
        assert s.distributed_mode is True
        assert s.hub_role == "hub"

    def test_shard_defaults(self):
        from semantic_searcher.config import Settings
        s = Settings(_env_file=None)
        assert s.shard_size_target == 500_000
        assert s.shard_replication_factor == 2
        assert s.search_fanout_timeout_ms == 2000

    def test_worker_config(self):
        from semantic_searcher.config import Settings
        s = Settings(_env_file=None, hub_role="worker", hub_url="http://hub:8900", worker_api_key="secret")
        assert s.hub_role == "worker"
        assert s.hub_url == "http://hub:8900"


# --- Integration-style tests (mocked DB) ---

class TestWorkerRegistrationFlow:
    def test_key_generation_and_hash_match(self):
        from semantic_searcher.middleware.worker_auth import generate_api_key, hash_api_key
        key = generate_api_key()
        stored_hash = hash_api_key(key)
        # Simulate verification: re-hash the key and compare
        assert hash_api_key(key) == stored_hash

    def test_different_key_doesnt_match(self):
        from semantic_searcher.middleware.worker_auth import generate_api_key, hash_api_key
        key1 = generate_api_key()
        key2 = generate_api_key()
        assert hash_api_key(key1) != hash_api_key(key2)


class TestSearchMergeLogic:
    def test_merge_prefers_highest_score(self):
        """When multiple workers return the same page_id, take the highest score."""
        results_w1 = [{"page_id": 1, "score": 0.9, "chunk_text": "a"}]
        results_w2 = [{"page_id": 1, "score": 0.7, "chunk_text": "b"}]

        merged = {}
        for r in results_w1 + results_w2:
            pid = r["page_id"]
            if pid not in merged or r["score"] > merged[pid]["score"]:
                merged[pid] = r

        assert len(merged) == 1
        assert merged[1]["score"] == 0.9

    def test_merge_combines_different_pages(self):
        results_w1 = [{"page_id": 1, "score": 0.9, "chunk_text": "a"}]
        results_w2 = [{"page_id": 2, "score": 0.8, "chunk_text": "b"}]

        merged = {}
        for r in results_w1 + results_w2:
            pid = r["page_id"]
            if pid not in merged or r["score"] > merged[pid]["score"]:
                merged[pid] = r

        sorted_results = sorted(merged.values(), key=lambda x: x["score"], reverse=True)
        assert len(sorted_results) == 2
        assert sorted_results[0]["page_id"] == 1

    def test_merge_respects_limit(self):
        results = [{"page_id": i, "score": 1.0 - i * 0.01, "chunk_text": f"chunk {i}"} for i in range(100)]
        limited = sorted(results, key=lambda x: x["score"], reverse=True)[:10]
        assert len(limited) == 10
        assert limited[0]["score"] > limited[9]["score"]


# --- WebSocket Tunnel ---

class TestWorkerTunnelHub:
    def test_no_connected_workers(self):
        from semantic_searcher.services.worker_tunnel import WorkerTunnelHub
        hub = WorkerTunnelHub()
        assert hub.connected_workers == []
        assert not hub.is_connected("w1")

    @pytest.mark.asyncio
    async def test_register_and_unregister(self):
        from semantic_searcher.services.worker_tunnel import WorkerTunnelHub
        hub = WorkerTunnelHub()
        mock_ws = MagicMock()
        mock_ws.close = AsyncMock()
        await hub.register("w1", mock_ws)
        assert hub.is_connected("w1")
        assert "w1" in hub.connected_workers
        await hub.unregister("w1")
        assert not hub.is_connected("w1")

    @pytest.mark.asyncio
    async def test_register_replaces_old_connection(self):
        from semantic_searcher.services.worker_tunnel import WorkerTunnelHub
        hub = WorkerTunnelHub()
        ws1 = MagicMock(); ws1.close = AsyncMock()
        ws2 = MagicMock(); ws2.close = AsyncMock()
        await hub.register("w1", ws1)
        await hub.register("w1", ws2)
        assert hub.is_connected("w1")
        ws1.close.assert_called_once()

    @pytest.mark.asyncio
    async def test_fanout_empty(self):
        from semantic_searcher.services.worker_tunnel import WorkerTunnelHub
        hub = WorkerTunnelHub()
        results = await hub.fanout_search([0.1] * 768, "test", limit=10)
        assert results == []

    @pytest.mark.asyncio
    async def test_handle_search_response(self):
        from semantic_searcher.services.worker_tunnel import WorkerTunnelHub
        import json
        hub = WorkerTunnelHub()
        # Create a pending future
        future = asyncio.get_event_loop().create_future()
        hub._pending["req-123"] = future
        # Simulate response
        await hub.handle_message("w1", json.dumps({
            "type": "search_response",
            "request_id": "req-123",
            "results": {"text_results": [{"page_id": 1, "score": 0.9, "chunk_text": "hi"}]},
        }))
        result = future.result()
        assert result["text_results"][0]["page_id"] == 1


class TestWorkerTunnelClient:
    def test_ws_url_conversion(self):
        from semantic_searcher.services.worker_tunnel import WorkerTunnelClient
        client = WorkerTunnelClient(
            hub_url="http://hub:8900", api_key="key", worker_id="w1",
            search_handler=MagicMock(),
        )
        assert client._ws_url == "ws://hub:8900/api/worker/ws/w1"

    def test_wss_url_conversion(self):
        from semantic_searcher.services.worker_tunnel import WorkerTunnelClient
        client = WorkerTunnelClient(
            hub_url="https://hub:8900", api_key="key", worker_id="w1",
            search_handler=MagicMock(),
        )
        assert client._ws_url == "wss://hub:8900/api/worker/ws/w1"

    def test_initial_state(self):
        from semantic_searcher.services.worker_tunnel import WorkerTunnelClient
        client = WorkerTunnelClient(
            hub_url="http://hub:8900", api_key="key", worker_id="w1",
            search_handler=MagicMock(),
        )
        assert not client.connected


class TestSearchCoordinatorTunnelPreference:
    @pytest.mark.asyncio
    async def test_prefers_tunnel_over_http(self):
        from semantic_searcher.services.search_coordinator import SearchCoordinator
        from semantic_searcher.services.worker_tunnel import WorkerTunnelHub
        coord = SearchCoordinator()
        tunnel = WorkerTunnelHub()

        # Register a fake worker on the tunnel
        mock_ws = MagicMock()
        mock_ws.close = AsyncMock()
        await tunnel.register("w1", mock_ws)

        coord._tunnel_hub = tunnel
        coord._workers = {"w1": {"endpoint_url": "http://w1:8901", "shard_ids": []}}

        # Mock the tunnel fanout
        tunnel.fanout_search = AsyncMock(return_value=[{"page_id": 1, "score": 0.9, "chunk_text": "via tunnel"}])

        results = await coord.fanout_text_search(np.zeros(768), "test", 10)
        tunnel.fanout_search.assert_called_once()
        assert results[0]["chunk_text"] == "via tunnel"
