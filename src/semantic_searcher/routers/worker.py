"""Distributed worker management API.

Hub endpoints for:
- Worker registration/deregistration (admin)
- Heartbeat (workers)
- Crawl job distribution (workers pull)
- Crawl completion reporting (workers push metadata)
- Shard management (admin)
"""

import datetime
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, WebSocket, WebSocketDisconnect
from sqlalchemy import select, update, text
from sqlalchemy.ext.asyncio import AsyncSession

from semantic_searcher.database import get_session
from semantic_searcher.middleware.worker_auth import (
    generate_api_key,
    hash_api_key,
    verify_worker,
)
from semantic_searcher.models.db import (
    CrawlJobAssignment,
    CrawlQueue,
    Page,
    Shard,
    Worker,
)
from semantic_searcher.models.worker_schemas import (
    CrawlCompleteRequest,
    CrawlJob,
    CrawlJobsResponse,
    HeartbeatRequest,
    HeartbeatResponse,
    ShardInfo,
    ShardListResponse,
    WorkerInfo,
    WorkerListResponse,
    WorkerRegisterRequest,
    WorkerRegisterResponse,
)

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api/worker", tags=["worker"])

_CRAWL_JOB_LEASE_SECONDS = 300


# ── Admin endpoints (no worker auth — protected by hub admin separately) ──

@router.post("/register", response_model=WorkerRegisterResponse)
async def register_worker(
    body: WorkerRegisterRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    """Register a new worker. Returns the API key (shown once)."""
    worker_id = str(uuid.uuid4())
    api_key = generate_api_key()
    key_hash = hash_api_key(api_key)

    worker = Worker(
        worker_id=worker_id,
        name=body.name,
        api_key_hash=key_hash,
        endpoint_url=body.endpoint_url,
        status="active",
        capabilities=body.capabilities,
        last_heartbeat=datetime.datetime.utcnow(),
    )
    session.add(worker)
    await session.commit()

    log.info("Registered worker %s (%s) at %s", worker_id, body.name, body.endpoint_url)
    return WorkerRegisterResponse(
        worker_id=worker_id,
        api_key=api_key,
        message=f"Worker '{body.name}' registered. Save this API key — it won't be shown again.",
    )


@router.delete("/{worker_id}")
async def deregister_worker(
    worker_id: str,
    session: AsyncSession = Depends(get_session),
):
    """Deregister a worker and orphan its shards."""
    result = await session.execute(select(Worker).where(Worker.worker_id == worker_id))
    worker = result.scalar_one_or_none()
    if not worker:
        raise HTTPException(status_code=404, detail="Worker not found")

    # Orphan shards where this worker is primary
    await session.execute(
        update(Shard)
        .where(Shard.primary_worker_id == worker_id)
        .values(primary_worker_id=None, status="orphaned")
    )
    await session.delete(worker)
    await session.commit()
    log.info("Deregistered worker %s", worker_id)
    return {"message": "Worker deregistered", "worker_id": worker_id}


@router.get("/list", response_model=WorkerListResponse)
async def list_workers(session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(Worker).order_by(Worker.created_at))
    workers = result.scalars().all()
    return WorkerListResponse(workers=[
        WorkerInfo(
            worker_id=w.worker_id,
            name=w.name,
            endpoint_url=w.endpoint_url,
            status=w.status,
            last_heartbeat=w.last_heartbeat.isoformat() if w.last_heartbeat else None,
            shard_ids=w.shard_ids,
            reputation_score=w.reputation_score,
            capabilities=w.capabilities,
        )
        for w in workers
    ])


@router.get("/activity")
async def worker_activity(
    worker_id: str | None = None,
    limit: int = 50,
    session: AsyncSession = Depends(get_session),
):
    """View worker activity dashboard — status, IP, ping, job stats."""
    from semantic_searcher.models.db import WorkerActivityLog

    # Current status of all workers
    workers_result = await session.execute(select(Worker).order_by(Worker.name))
    workers = workers_result.scalars().all()

    dashboard = []
    for w in workers:
        entry = {
            "worker_id": w.worker_id,
            "name": w.name,
            "ip_address": w.ip_address,
            "status": w.status,
            "ping_latency_ms": w.ping_latency_ms,
            "last_heartbeat": w.last_heartbeat.isoformat() if w.last_heartbeat else None,
            "jobs_completed": w.jobs_completed,
            "jobs_failed": w.jobs_failed,
            "pages_indexed": w.pages_indexed,
            "endpoint_url": w.endpoint_url,
            "shard_ids": w.shard_ids,
        }
        dashboard.append(entry)

    # Recent activity log
    query = select(WorkerActivityLog).order_by(WorkerActivityLog.logged_at.desc()).limit(limit)
    if worker_id:
        query = query.where(WorkerActivityLog.worker_id == worker_id)
    log_result = await session.execute(query)
    activity_log = [
        {
            "worker_id": l.worker_id,
            "ip_address": l.ip_address,
            "status": l.status,
            "ping_latency_ms": l.ping_latency_ms,
            "jobs_completed": l.jobs_completed,
            "pages_indexed": l.pages_indexed,
            "logged_at": l.logged_at.isoformat() if l.logged_at else None,
        }
        for l in log_result.scalars().all()
    ]

    return {"workers": dashboard, "activity_log": activity_log}


# ── Shard management ──

@router.get("/shards", response_model=ShardListResponse)
async def list_shards(session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(Shard).order_by(Shard.page_id_start))
    shards = result.scalars().all()
    return ShardListResponse(shards=[
        ShardInfo(
            shard_id=s.shard_id,
            page_id_start=s.page_id_start,
            page_id_end=s.page_id_end,
            primary_worker_id=s.primary_worker_id,
            replica_worker_ids=s.replica_worker_ids,
            status=s.status,
            point_count=s.point_count,
        )
        for s in shards
    ])


# ── Worker-authenticated endpoints ──

@router.post("/heartbeat", response_model=HeartbeatResponse)
async def heartbeat(
    body: HeartbeatRequest,
    request: Request,
    worker: Worker = Depends(verify_worker),
    session: AsyncSession = Depends(get_session),
):
    """Worker heartbeat — updates last_heartbeat, IP, and reports stats."""
    # Extract real IP from request
    client_ip = request.headers.get("X-Real-IP") or request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
    if not client_ip and request.client:
        client_ip = request.client.host

    updates = {
        "last_heartbeat": datetime.datetime.utcnow(),
        "status": "active",
    }
    if client_ip:
        updates["ip_address"] = client_ip
    if body.system_stats:
        stats = body.system_stats
        if "jobs_completed" in stats:
            updates["jobs_completed"] = stats["jobs_completed"]
        if "jobs_failed" in stats:
            updates["jobs_failed"] = stats["jobs_failed"]
        if "pages_indexed" in stats:
            updates["pages_indexed"] = stats["pages_indexed"]

    await session.execute(
        update(Worker).where(Worker.worker_id == worker.worker_id).values(**updates)
    )
    await session.commit()

    return HeartbeatResponse(
        status="ok",
        shard_assignments=worker.shard_ids,
    )


@router.get("/crawl-jobs", response_model=CrawlJobsResponse)
async def get_crawl_jobs(
    batch_size: int = 50,
    worker: Worker = Depends(verify_worker),
    session: AsyncSession = Depends(get_session),
):
    """Worker pulls a batch of crawl jobs from the queue."""
    from semantic_searcher.config import settings
    import random

    batch_size = min(batch_size, 200)

    # Get ID range for random sampling
    id_range = await session.execute(text(
        "SELECT MIN(id), MAX(id) FROM crawl_queue WHERE status = 'queued'"
    ))
    id_min, id_max = id_range.one()
    if id_min is None:
        return CrawlJobsResponse(jobs=[], lease_duration_seconds=_CRAWL_JOB_LEASE_SECONDS)

    # Random probe for diverse URLs
    n_probes = min(30, id_max - id_min + 1)
    probes = sorted(random.sample(range(id_min, id_max + 1), n_probes))

    rows = []
    for probe_id in probes:
        r = await session.execute(text(
            "SELECT id, url, url_hash, depth FROM crawl_queue "
            "WHERE status = 'queued' AND id >= :probe ORDER BY id LIMIT :lim"
        ), {"probe": probe_id, "lim": batch_size * 2})
        rows.extend(r.fetchall())
        if len(rows) >= batch_size * 5:
            break

    # Dedup and select
    seen_ids = set()
    selected = []
    for row in rows:
        if row[0] not in seen_ids and len(selected) < batch_size:
            seen_ids.add(row[0])
            selected.append(row)

    if not selected:
        return CrawlJobsResponse(jobs=[], lease_duration_seconds=_CRAWL_JOB_LEASE_SECONDS)

    # Determine shard assignment for each URL (hash-based)
    shard_result = await session.execute(select(Shard).where(Shard.status == "active"))
    active_shards = shard_result.scalars().all()

    # Mark as crawling and create job assignments
    jobs = []
    for row in selected:
        queue_id, url, url_hash, depth = row[0], row[1], row[2], row[3]

        # Assign shard (consistent hash on URL hash)
        if active_shards:
            shard_idx = hash(url_hash) % len(active_shards)
            target_shard = active_shards[shard_idx].shard_id
        else:
            target_shard = "default"

        await session.execute(
            update(CrawlQueue).where(CrawlQueue.id == queue_id).values(status="crawling")
        )
        assignment = CrawlJobAssignment(
            crawl_queue_id=queue_id,
            worker_id=worker.worker_id,
            assigned_shard_id=target_shard,
        )
        session.add(assignment)
        jobs.append(CrawlJob(
            job_id=queue_id,
            url=url,
            url_hash=url_hash,
            depth=depth,
            target_shard_id=target_shard,
        ))

    await session.commit()
    log.info("Assigned %d crawl jobs to worker %s", len(jobs), worker.worker_id)
    return CrawlJobsResponse(jobs=jobs, lease_duration_seconds=_CRAWL_JOB_LEASE_SECONDS)


@router.post("/crawl-complete")
async def crawl_complete(
    body: CrawlCompleteRequest,
    worker: Worker = Depends(verify_worker),
    session: AsyncSession = Depends(get_session),
):
    """Worker reports crawl completion with page metadata."""
    # Update crawl queue status
    new_status = {"completed": "done", "failed": "failed", "skipped": "skipped"}.get(body.status, "failed")
    await session.execute(
        update(CrawlQueue)
        .where(CrawlQueue.id == body.job_id)
        .values(status=new_status, crawled_at=datetime.datetime.utcnow())
    )

    # Update job assignment
    await session.execute(
        update(CrawlJobAssignment)
        .where(CrawlJobAssignment.crawl_queue_id == body.job_id, CrawlJobAssignment.worker_id == worker.worker_id)
        .values(status=body.status, completed_at=datetime.datetime.utcnow())
    )

    # If completed with metadata, store page metadata in hub DB
    if body.status == "completed" and body.metadata:
        meta = body.metadata
        url_hash = meta.get("url_hash", "")
        url = meta.get("url", "")

        # Get or create page row
        existing = await session.execute(
            select(Page.id).where(Page.url_hash == url_hash)
        )
        page_row = existing.scalar_one_or_none()

        now = datetime.datetime.utcnow()
        if page_row:
            await session.execute(
                update(Page).where(Page.id == page_row).values(
                    title=meta.get("title"),
                    meta_description=meta.get("meta_description"),
                    domain=meta.get("domain"),
                    tld_group=meta.get("tld_group"),
                    content_category=meta.get("content_category"),
                    language=meta.get("language"),
                    nsfw_flag=meta.get("nsfw_flag", False),
                    status="indexed",
                    indexed_at=now,
                )
            )
        else:
            from semantic_searcher.utils.url_utils import normalize_url, url_hash as compute_hash
            page = Page(
                url=url,
                url_hash=url_hash or compute_hash(url),
                title=meta.get("title"),
                meta_description=meta.get("meta_description"),
                domain=meta.get("domain"),
                tld_group=meta.get("tld_group"),
                content_category=meta.get("content_category"),
                language=meta.get("language"),
                nsfw_flag=meta.get("nsfw_flag", False),
                status="indexed",
                indexed_at=now,
            )
            session.add(page)

    await session.commit()
    return {"status": "ok", "job_id": body.job_id}


@router.get("/shard-assignment")
async def get_shard_assignment(
    worker: Worker = Depends(verify_worker),
    session: AsyncSession = Depends(get_session),
):
    """Worker gets its current shard assignments."""
    shard_ids = worker.shard_ids or []
    shards = []
    if shard_ids:
        result = await session.execute(
            select(Shard).where(Shard.shard_id.in_(shard_ids))
        )
        for s in result.scalars().all():
            shards.append(ShardInfo(
                shard_id=s.shard_id,
                page_id_start=s.page_id_start,
                page_id_end=s.page_id_end,
                primary_worker_id=s.primary_worker_id,
                replica_worker_ids=s.replica_worker_ids,
                status=s.status,
                point_count=s.point_count,
            ))
    return {"worker_id": worker.worker_id, "shards": shards}


# ── Worker-side endpoints (called by hub or other workers) ──

@router.post("/search")
async def worker_search(request: Request, body: dict):
    """Hub sends search request — worker queries local Qdrant and returns results."""
    handler = getattr(request.app.state, "worker_search_handler", None)
    if handler is None:
        raise HTTPException(status_code=501, detail="Not a worker node")

    result = handler.search(
        query_vector=body.get("query_vector", []),
        query_text=body.get("query_text", ""),
        clip_vector=body.get("clip_vector"),
        limit=body.get("limit", 200),
        mode=body.get("mode", "hybrid"),
        filters=body.get("filters"),
    )

    from semantic_searcher.config import settings
    result["worker_id"] = settings.worker_id
    result["shard_ids"] = []  # TODO: populate from shard assignments
    return result


@router.post("/vector-batch")
async def receive_vector_batch(request: Request, body: dict):
    """Receive a batch of vectors for replication (from hub migration or primary worker)."""
    from qdrant_client.models import Document, PointStruct

    qdrant = getattr(request.app.state, "_qdrant", None)
    if qdrant is None:
        # Try to get from indexer
        indexer = getattr(request.app.state, "indexer_service", None)
        if indexer:
            qdrant = indexer._qdrant

    if qdrant is None:
        raise HTTPException(status_code=501, detail="No Qdrant client available")

    collection = body.get("collection", "text_chunks_v2")
    shard_id = body.get("shard_id", "")
    points_data = body.get("points", [])

    points = []
    for p in points_data:
        bm25_text = p.get("bm25_text", "")
        points.append(PointStruct(
            id=p["id"],
            vector={
                "dense": p["dense_vector"],
                "text-bm25": Document(text=bm25_text, model="Qdrant/bm25"),
            },
            payload=p.get("payload", {}),
        ))

    if points:
        qdrant.upsert(collection, points=points, wait=True)

    return {"accepted": len(points), "rejected": 0, "message": "ok"}


# ── WebSocket tunnel (workers behind NAT connect here) ──

@router.websocket("/ws/{worker_id}")
async def worker_websocket(websocket: WebSocket, worker_id: str):
    """Persistent WebSocket tunnel for worker↔hub communication.

    Workers connect here on startup. The hub sends search requests
    through the tunnel — no port forwarding needed on the worker side.
    """
    # Validate API key from headers
    auth = websocket.headers.get("authorization", "")
    if not auth.startswith("Bearer "):
        await websocket.close(code=4001, reason="Missing auth")
        return

    from semantic_searcher.middleware.worker_auth import hash_api_key
    from semantic_searcher.database import async_session as get_async_session
    key_hash = hash_api_key(auth.replace("Bearer ", ""))

    async with get_async_session() as session:
        result = await session.execute(select(Worker).where(Worker.worker_id == worker_id))
        worker = result.scalar_one_or_none()

    if worker is None or worker.api_key_hash != key_hash:
        await websocket.close(code=4003, reason="Invalid credentials")
        return

    if worker.status == "banned":
        await websocket.close(code=4003, reason="Worker banned")
        return

    await websocket.accept()

    # Register with the tunnel hub
    tunnel_hub = getattr(websocket.app.state, "worker_tunnel_hub", None)
    if tunnel_hub is None:
        await websocket.close(code=4500, reason="Tunnel not available")
        return

    await tunnel_hub.register(worker_id, websocket)

    # Update worker status
    async with get_async_session() as session:
        await session.execute(
            update(Worker).where(Worker.worker_id == worker_id).values(
                status="active",
                last_heartbeat=datetime.datetime.utcnow(),
                ip_address=websocket.client.host if websocket.client else None,
            )
        )
        await session.commit()

    try:
        while True:
            data = await websocket.receive_text()
            await tunnel_hub.handle_message(worker_id, data)
    except WebSocketDisconnect:
        pass
    except Exception as e:
        log.warning("WebSocket error for worker %s: %s", worker_id, e)
    finally:
        await tunnel_hub.unregister(worker_id)
