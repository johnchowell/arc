"""Shard management — health monitoring, ping latency, activity logging, replication.

Runs as a background service on the hub.
"""

import asyncio
import datetime
import logging
import subprocess
import time
import uuid
from urllib.parse import urlparse

import httpx
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from semantic_searcher.config import settings
from semantic_searcher.database import async_session
from semantic_searcher.models.db import Shard, Worker, WorkerActivityLog

log = logging.getLogger(__name__)

_HEARTBEAT_TIMEOUT = 90  # seconds before marking worker offline
_RECHECK_INTERVAL = 30   # seconds between health checks
_ACTIVITY_LOG_INTERVAL = 60  # seconds between activity log entries


def _ping_host(host: str, timeout: float = 2.0) -> float | None:
    """Measure ICMP ping latency to a host. Returns ms or None on failure."""
    try:
        result = subprocess.run(
            ["ping", "-c", "1", "-W", str(int(timeout)), host],
            capture_output=True, text=True, timeout=timeout + 1,
        )
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                if "time=" in line:
                    ms = float(line.split("time=")[1].split()[0])
                    return ms
    except Exception:
        pass
    return None


def _extract_host(endpoint_url: str) -> str:
    """Extract hostname/IP from endpoint URL."""
    parsed = urlparse(endpoint_url)
    return parsed.hostname or parsed.netloc.split(":")[0]


class ShardManager:
    """Background service that monitors worker health and manages shard replication."""

    def __init__(self):
        self._running = False
        self._task = None
        self._last_activity_log = 0.0

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._run())
        log.info("Shard manager started")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self):
        while self._running:
            try:
                await self._check_worker_health()
                await self._check_shard_health()

                # Log activity periodically
                now = time.monotonic()
                if now - self._last_activity_log >= _ACTIVITY_LOG_INTERVAL:
                    await self._log_worker_activity()
                    self._last_activity_log = now
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.warning("Shard manager error: %s", e)
            await asyncio.sleep(_RECHECK_INTERVAL)

    async def _check_worker_health(self):
        """Check all active workers — ping, update status, measure latency."""
        async with async_session() as session:
            result = await session.execute(select(Worker).where(Worker.status.in_(["active", "offline"])))
            workers = result.scalars().all()

            cutoff = datetime.datetime.utcnow() - datetime.timedelta(seconds=_HEARTBEAT_TIMEOUT)

            for w in workers:
                host = _extract_host(w.endpoint_url)
                ip = w.ip_address or host

                # Measure ping latency in a thread (blocking call)
                loop = asyncio.get_event_loop()
                latency = await loop.run_in_executor(None, _ping_host, ip)

                updates = {"ping_latency_ms": latency}

                if w.status == "active" and w.last_heartbeat and w.last_heartbeat < cutoff:
                    updates["status"] = "offline"
                    log.warning("Worker %s (%s) missed heartbeat — marking offline (ping: %s)",
                                w.worker_id, w.name, f"{latency:.1f}ms" if latency else "unreachable")
                elif w.status == "offline" and latency is not None:
                    # Worker is reachable again via ping — check if heartbeat resumes
                    pass

                if ip != w.ip_address:
                    updates["ip_address"] = ip

                await session.execute(
                    update(Worker).where(Worker.id == w.id).values(**updates)
                )

            await session.commit()

    async def _log_worker_activity(self):
        """Write a snapshot of all worker activity to the log table."""
        async with async_session() as session:
            result = await session.execute(select(Worker))
            workers = result.scalars().all()

            for w in workers:
                log_entry = WorkerActivityLog(
                    worker_id=w.worker_id,
                    ip_address=w.ip_address,
                    status=w.status,
                    ping_latency_ms=w.ping_latency_ms,
                    jobs_completed=w.jobs_completed,
                    jobs_failed=w.jobs_failed,
                    pages_indexed=w.pages_indexed,
                )
                session.add(log_entry)

            await session.commit()

    async def _check_shard_health(self):
        """Check for orphaned shards and trigger re-replication if needed."""
        async with async_session() as session:
            result = await session.execute(select(Shard).where(Shard.status == "active"))
            shards = result.scalars().all()

            workers_result = await session.execute(select(Worker).where(Worker.status == "active"))
            active_workers = {w.worker_id: w for w in workers_result.scalars().all()}

            for shard in shards:
                if shard.primary_worker_id and shard.primary_worker_id not in active_workers:
                    replicas = shard.replica_worker_ids or []
                    promoted = None
                    for rid in replicas:
                        if rid in active_workers:
                            promoted = rid
                            break

                    if promoted:
                        new_replicas = [r for r in replicas if r != promoted]
                        await session.execute(
                            update(Shard).where(Shard.id == shard.id).values(
                                primary_worker_id=promoted,
                                replica_worker_ids=new_replicas,
                            )
                        )
                        log.info("Promoted replica %s to primary for shard %s", promoted, shard.shard_id)
                    else:
                        await session.execute(
                            update(Shard).where(Shard.id == shard.id).values(status="orphaned")
                        )
                        log.warning("Shard %s is orphaned — no active replicas", shard.shard_id)

            await session.commit()

    async def create_initial_shards(self, num_shards: int, max_page_id: int):
        """Create initial shard definitions for the page ID space."""
        async with async_session() as session:
            existing = await session.execute(select(Shard))
            if existing.scalars().first():
                log.info("Shards already exist, skipping creation")
                return

            shard_size = max_page_id // num_shards + 1
            for i in range(num_shards):
                start = i * shard_size
                end = min((i + 1) * shard_size - 1, max_page_id)
                shard = Shard(
                    shard_id=f"shard-{i:04d}",
                    page_id_start=start,
                    page_id_end=end,
                    status="active",
                )
                session.add(shard)

            await session.commit()
            log.info("Created %d shards (shard_size=%d, max_page_id=%d)", num_shards, shard_size, max_page_id)

    async def assign_shards_to_worker(self, worker_id: str, count: int = 4):
        """Assign unassigned shards to a worker."""
        async with async_session() as session:
            result = await session.execute(
                select(Shard).where(Shard.primary_worker_id.is_(None)).limit(count)
            )
            available = result.scalars().all()

            assigned = []
            for shard in available:
                await session.execute(
                    update(Shard).where(Shard.id == shard.id).values(primary_worker_id=worker_id)
                )
                assigned.append(shard.shard_id)

            if assigned:
                worker_result = await session.execute(
                    select(Worker).where(Worker.worker_id == worker_id)
                )
                worker = worker_result.scalar_one_or_none()
                if worker:
                    existing = worker.shard_ids or []
                    await session.execute(
                        update(Worker).where(Worker.id == worker.id).values(
                            shard_ids=existing + assigned
                        )
                    )

                await session.commit()
                log.info("Assigned %d shards to worker %s: %s", len(assigned), worker_id, assigned)

            return assigned

    async def add_replica(self, shard_id: str, worker_id: str):
        """Add a replica worker for a shard."""
        async with async_session() as session:
            result = await session.execute(
                select(Shard).where(Shard.shard_id == shard_id)
            )
            shard = result.scalar_one_or_none()
            if not shard:
                return

            replicas = shard.replica_worker_ids or []
            if worker_id not in replicas:
                replicas.append(worker_id)
                await session.execute(
                    update(Shard).where(Shard.id == shard.id).values(replica_worker_ids=replicas)
                )
                await session.commit()
                log.info("Added replica %s for shard %s", worker_id, shard_id)
