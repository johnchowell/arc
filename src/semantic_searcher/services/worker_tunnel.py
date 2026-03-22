"""WebSocket tunnel for worker↔hub communication.

Workers behind NAT connect to the hub via WebSocket. The hub sends search
requests through the tunnel and receives results — no port forwarding needed.

Hub side: WorkerTunnelHub manages connected workers and dispatches search requests.
Worker side: WorkerTunnelClient maintains the persistent connection and handles requests.
"""

import asyncio
import json
import logging
import time
import uuid
from typing import Any

log = logging.getLogger(__name__)

# How long to wait for a worker to respond to a search request
_SEARCH_TIMEOUT = 3.0


class WorkerTunnelHub:
    """Hub-side WebSocket tunnel manager.

    Tracks connected workers and dispatches search/heartbeat requests
    through their WebSocket connections.
    """

    def __init__(self):
        # worker_id -> websocket connection
        self._connections: dict[str, Any] = {}
        # request_id -> Future (for awaiting search responses)
        self._pending: dict[str, asyncio.Future] = {}

    @property
    def connected_workers(self) -> list[str]:
        return list(self._connections.keys())

    def is_connected(self, worker_id: str) -> bool:
        return worker_id in self._connections

    async def register(self, worker_id: str, websocket):
        """Register a worker's WebSocket connection."""
        old = self._connections.pop(worker_id, None)
        if old:
            try:
                await old.close()
            except Exception:
                pass
        self._connections[worker_id] = websocket
        log.info("Worker %s connected via WebSocket", worker_id)

    async def unregister(self, worker_id: str):
        """Remove a worker's connection."""
        self._connections.pop(worker_id, None)
        log.info("Worker %s disconnected from WebSocket", worker_id)

    async def send_search_request(self, worker_id: str,
                                   query_vector: list[float],
                                   query_text: str,
                                   clip_vector: list[float] | None = None,
                                   limit: int = 200,
                                   mode: str = "hybrid",
                                   filters: dict | None = None) -> dict | None:
        """Send a search request to a worker via WebSocket and await response."""
        ws = self._connections.get(worker_id)
        if ws is None:
            return None

        request_id = str(uuid.uuid4())
        future = asyncio.get_event_loop().create_future()
        self._pending[request_id] = future

        try:
            msg = json.dumps({
                "type": "search",
                "request_id": request_id,
                "query_vector": query_vector,
                "query_text": query_text,
                "clip_vector": clip_vector,
                "limit": limit,
                "mode": mode,
                "filters": filters,
            })
            await ws.send_text(msg)

            # Wait for response with timeout
            result = await asyncio.wait_for(future, timeout=_SEARCH_TIMEOUT)
            return result
        except asyncio.TimeoutError:
            log.warning("Search timeout for worker %s (req %s)", worker_id, request_id[:8])
            return None
        except Exception as e:
            log.warning("Search error for worker %s: %s", worker_id, e)
            return None
        finally:
            self._pending.pop(request_id, None)

    async def handle_message(self, worker_id: str, data: str):
        """Process an incoming message from a worker."""
        try:
            msg = json.loads(data)
        except json.JSONDecodeError:
            return

        msg_type = msg.get("type")

        if msg_type == "search_response":
            request_id = msg.get("request_id")
            future = self._pending.get(request_id)
            if future and not future.done():
                future.set_result(msg.get("results", {}))

        elif msg_type == "heartbeat":
            # Worker sent a heartbeat through the tunnel
            pass

    async def fanout_search(self, query_vector: list[float], query_text: str,
                            clip_vector: list[float] | None = None,
                            limit: int = 200, mode: str = "hybrid",
                            filters: dict | None = None) -> list[dict]:
        """Fan out search to all connected workers via WebSocket tunnels."""
        if not self._connections:
            return []

        tasks = []
        worker_ids = []
        for wid in list(self._connections.keys()):
            tasks.append(self.send_search_request(
                wid, query_vector, query_text, clip_vector, limit, mode, filters
            ))
            worker_ids.append(wid)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Merge results (same logic as SearchCoordinator)
        merged: dict[int, dict] = {}
        for wid, result in zip(worker_ids, results):
            if isinstance(result, Exception) or result is None:
                continue
            for item in result.get("text_results", []):
                pid = item["page_id"]
                if pid not in merged or item["score"] > merged[pid]["score"]:
                    merged[pid] = item

        return sorted(merged.values(), key=lambda x: x["score"], reverse=True)[:limit]


class WorkerTunnelClient:
    """Worker-side WebSocket tunnel client.

    Maintains a persistent connection to the hub, receives search requests,
    executes them locally, and sends results back.
    """

    def __init__(self, hub_url: str, api_key: str, worker_id: str, search_handler):
        # Convert http(s) to ws(s)
        ws_url = hub_url.replace("http://", "ws://").replace("https://", "wss://")
        self._ws_url = f"{ws_url}/api/worker/ws/{worker_id}"
        self._api_key = api_key
        self._worker_id = worker_id
        self._search_handler = search_handler
        self._running = False
        self._task = None
        self._connected = False

    @property
    def connected(self) -> bool:
        return self._connected

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._run())
        log.info("WebSocket tunnel client started (hub=%s)", self._ws_url)

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self):
        """Maintain persistent WebSocket connection with auto-reconnect."""
        import websockets

        while self._running:
            try:
                headers = {
                    "Authorization": f"Bearer {self._api_key}",
                    "X-Worker-ID": self._worker_id,
                }
                async with websockets.connect(
                    self._ws_url,
                    additional_headers=headers,
                    ping_interval=20,
                    ping_timeout=10,
                    max_size=10 * 1024 * 1024,  # 10MB max message
                ) as ws:
                    self._connected = True
                    log.info("WebSocket tunnel connected to hub")

                    # Send initial heartbeat
                    await ws.send(json.dumps({"type": "heartbeat", "worker_id": self._worker_id}))

                    async for message in ws:
                        if not self._running:
                            break
                        await self._handle_message(ws, message)

            except asyncio.CancelledError:
                break
            except Exception as e:
                self._connected = False
                log.warning("WebSocket tunnel disconnected: %s — reconnecting in 5s", e)
                await asyncio.sleep(5)

        self._connected = False

    async def _handle_message(self, ws, data: str):
        """Handle incoming message from hub."""
        try:
            msg = json.loads(data)
        except json.JSONDecodeError:
            return

        msg_type = msg.get("type")

        if msg_type == "search":
            # Execute search locally and send results back
            request_id = msg.get("request_id", "")
            try:
                result = self._search_handler.search(
                    query_vector=msg.get("query_vector", []),
                    query_text=msg.get("query_text", ""),
                    clip_vector=msg.get("clip_vector"),
                    limit=msg.get("limit", 200),
                    mode=msg.get("mode", "hybrid"),
                    filters=msg.get("filters"),
                )
                await ws.send(json.dumps({
                    "type": "search_response",
                    "request_id": request_id,
                    "results": result,
                }))
            except Exception as e:
                log.error("Search request failed: %s", e)
                await ws.send(json.dumps({
                    "type": "search_response",
                    "request_id": request_id,
                    "results": {"text_results": [], "image_results": [], "search_time_ms": 0},
                }))

        elif msg_type == "ping":
            await ws.send(json.dumps({"type": "pong"}))
