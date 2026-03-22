import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI

from semantic_searcher.config import settings
from semantic_searcher.database import async_session
from semantic_searcher.services.clip_service import CLIPService
from semantic_searcher.services.crawler import CrawlerService
from semantic_searcher.services.ct_watcher import CTWatcherService
from semantic_searcher.services.link_harvester import LinkHarvesterService
from semantic_searcher.services.subdomain_enum import SubdomainEnumeratorService
from semantic_searcher.services.indexer import IndexerService
from semantic_searcher.services.renderer import RendererService
from semantic_searcher.services.search_queue import SearchQueue
from semantic_searcher.services.searcher import SearchService
from semantic_searcher.services.tokenizer import BPETokenizerService

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def _worker_lifespan_ctx(app: FastAPI):
    """Minimal startup for distributed worker — no MySQL needed."""
    import asyncio as _asyncio
    import torch

    log.info("Starting in WORKER mode (no MySQL)")

    # Load CLIP + MPNet
    n_gpus = torch.cuda.device_count()
    clip = CLIPService(device=f"cuda:0" if n_gpus > 0 else "cpu")
    await clip.start()
    app.state.clip_services = [clip]

    from semantic_searcher.services.text_encoder import TextEncoderService
    text_encoder = TextEncoderService(device="cpu")
    await text_encoder.start()
    app.state.text_encoder = text_encoder

    # Local Qdrant
    from qdrant_client import QdrantClient
    from semantic_searcher.services.qdrant_collections import ensure_collections
    qdrant_client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=60)
    ensure_collections(qdrant_client)
    log.info("Local Qdrant connected at %s:%d", settings.qdrant_host, settings.qdrant_port)

    # Worker search handler (responds to hub search requests)
    from semantic_searcher.services.worker_search_handler import WorkerSearchHandler
    app.state.worker_search_handler = WorkerSearchHandler(qdrant_client)

    # Load API key securely (wipes from environment after reading)
    from semantic_searcher.services.secret_loader import load_api_key
    api_key = load_api_key()

    # Remote crawl client (pulls jobs from hub API)
    from semantic_searcher.services.remote_crawl_client import RemoteCrawlClient
    remote_client = RemoteCrawlClient(
        text_encoder=text_encoder, clip=clip, qdrant_client=qdrant_client,
        hub_url=settings.hub_url, api_key=api_key, worker_id=settings.worker_id,
    )
    app.state.remote_crawl_client = remote_client
    await remote_client.start()

    # WebSocket tunnel to hub (enables search without port forwarding)
    from semantic_searcher.services.worker_tunnel import WorkerTunnelClient
    tunnel_client = WorkerTunnelClient(
        hub_url=settings.hub_url, api_key=api_key,
        worker_id=settings.worker_id, search_handler=app.state.worker_search_handler,
    )
    app.state.tunnel_client = tunnel_client
    await tunnel_client.start()

    # Heartbeat loop (fallback — tunnel also sends heartbeats)
    async def _heartbeat():
        while True:
            await remote_client.send_heartbeat()
            await _asyncio.sleep(settings.worker_heartbeat_interval)
    _asyncio.create_task(_heartbeat())

    # Stubs for routes that check these
    app.state.search_service = None
    app.state.search_queue = None
    app.state.indexer_service = None
    app.state.search_coordinator = None
    app.state.shard_manager = None
    app.state.worker_tunnel_hub = None

    log.info("Worker startup complete (tunnel=%s)", "connected" if tunnel_client.connected else "connecting")
    yield

    # Shutdown
    await remote_client.stop()
    log.info("Worker shutdown complete")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    log.info("Starting Semantic Searcher...")

    # Worker-only mode: skip MySQL, minimal startup
    if settings.distributed_mode and settings.hub_role == "worker":
        async with _worker_lifespan_ctx(app):
            yield
        return

    import torch
    n_gpus = torch.cuda.device_count()
    if n_gpus >= 2:
        clips = []
        for i in range(n_gpus):
            c = CLIPService(device=f"cuda:{i}")
            await c.start()
            clips.append(c)
        log.info("Loaded CLIP on %d GPUs", n_gpus)
    else:
        c = CLIPService()
        await c.start()
        clips = [c]
    app.state.clip_services = clips

    renderer = RendererService(max_concurrent=settings.renderer_concurrency)
    app.state.renderer_service = renderer

    # Initialize Qdrant
    from qdrant_client import QdrantClient
    from semantic_searcher.services.qdrant_collections import ensure_collections
    qdrant_client = QdrantClient(
        host=settings.qdrant_host, port=settings.qdrant_port, timeout=60,
    )
    ensure_collections(qdrant_client)
    # Ensure is_stale index exists (migration for existing collections)
    from qdrant_client.models import PayloadSchemaType
    try:
        qdrant_client.create_payload_index(
            "text_chunks", "is_stale", field_schema=PayloadSchemaType.BOOL
        )
        log.info("Created is_stale payload index on text_chunks")
    except Exception:
        pass  # Already exists
    log.info("Qdrant connected at %s:%d", settings.qdrant_host, settings.qdrant_port)

    # Initialize MPNet text encoder for v2 collection
    from semantic_searcher.services.text_encoder import TextEncoderService
    text_encoder = TextEncoderService(device="cpu")
    await text_encoder.start()
    app.state.text_encoder = text_encoder

    indexer = IndexerService(clips, renderer=renderer, qdrant=qdrant_client, text_encoder=text_encoder)
    app.state.indexer_service = indexer

    search_queue = None
    if not settings.crawl_only:
        search_svc = SearchService(clips[0], qdrant=qdrant_client, text_encoder=text_encoder)
        async with async_session() as session:
            await search_svc.load_index(session)
        search_svc.load_cross_encoder()
        search_svc.init_spell_checker()
        app.state.search_service = search_svc

        search_queue = SearchQueue(
            max_workers=settings.search_max_workers,
            max_queued=settings.search_queue_max,
        )
        app.state.search_queue = search_queue
        log.info("Search queue: max %d parallel, %d queued", settings.search_max_workers, settings.search_queue_max)
    else:
        app.state.search_service = None
        app.state.search_queue = None
        log.info("Crawl-only mode: search index and queue disabled")

    crawler = CrawlerService(indexer)
    app.state.crawler_service = crawler

    ct_watcher = CTWatcherService()
    app.state.ct_watcher = ct_watcher

    link_harvester = LinkHarvesterService()
    app.state.link_harvester = link_harvester

    subdomain_enum = SubdomainEnumeratorService()
    app.state.subdomain_enum = subdomain_enum

    tokenizer = BPETokenizerService()
    await tokenizer.start()
    app.state.tokenizer_service = tokenizer

    # Ensure crawl_seen dedup table exists and archive completed rows
    async with async_session() as session:
        from sqlalchemy import text as sql_text

        await session.execute(sql_text(
            "CREATE TABLE IF NOT EXISTS crawl_seen ("
            "  url_hash VARCHAR(64) PRIMARY KEY"
            ") ENGINE=InnoDB"
        ))
        await session.execute(sql_text(
            "CREATE TABLE IF NOT EXISTS search_logs ("
            "  id BIGINT AUTO_INCREMENT PRIMARY KEY,"
            "  query VARCHAR(2048) NOT NULL,"
            "  searched_at DATETIME DEFAULT CURRENT_TIMESTAMP"
            ") ENGINE=InnoDB"
        ))
        await session.commit()

    import asyncio as _asyncio

    # Migrate completed url_hashes to crawl_seen in background (don't block startup)
    async def _background_archival():
        try:
            total_archived = 0
            async with async_session() as arch_session:
                from sqlalchemy import text as arch_text
                while True:
                    r = await arch_session.execute(arch_text(
                        "INSERT IGNORE INTO crawl_seen (url_hash) "
                        "SELECT url_hash FROM crawl_queue "
                        "WHERE status IN ('done', 'failed', 'skipped') "
                        "LIMIT 10000"
                    ))
                    r2 = await arch_session.execute(arch_text(
                        "DELETE FROM crawl_queue "
                        "WHERE status IN ('done', 'failed', 'skipped') "
                        "LIMIT 10000"
                    ))
                    await arch_session.commit()
                    if r2.rowcount == 0:
                        break
                    total_archived += r2.rowcount
                    if total_archived % 100000 == 0:
                        log.info("Archived %d completed crawl_queue rows...", total_archived)
                    # Yield to event loop between batches
                    await _asyncio.sleep(0.1)
            if total_archived:
                log.info("Archived %d completed crawl_queue rows to crawl_seen", total_archived)
        except Exception as e:
            log.warning("crawl_queue archival error: %s", e)

    _asyncio.create_task(_background_archival())

    # Sync remaining crawl_queue url_hashes to crawl_seen in background
    async def _background_crawl_seen_sync():
        try:
            last_id = 0
            total = 0
            async with async_session() as bg_session:
                from sqlalchemy import text as bg_text
                while True:
                    await bg_session.execute(bg_text(
                        "INSERT IGNORE INTO crawl_seen (url_hash) "
                        "SELECT url_hash FROM crawl_queue "
                        "WHERE id > :last_id ORDER BY id LIMIT 10000"
                    ), {"last_id": last_id})
                    r2 = await bg_session.execute(bg_text(
                        "SELECT MAX(id) FROM (SELECT id FROM crawl_queue "
                        "WHERE id > :last_id ORDER BY id LIMIT 10000) t"
                    ), {"last_id": last_id})
                    max_id = r2.scalar()
                    await bg_session.commit()
                    if max_id is None:
                        break
                    last_id = max_id
                    total += 10000
                    if total % 100000 == 0:
                        log.info("crawl_seen background sync: %d rows processed...", total)
            log.info("crawl_seen background sync complete")
        except Exception as e:
            log.warning("crawl_seen background sync error: %s", e)

    _asyncio.create_task(_background_crawl_seen_sync())

    # Reset any stuck entries from interrupted runs
    async with async_session() as session:
        from sqlalchemy import text as sql_text
        try:
            result = await session.execute(sql_text(
                "UPDATE crawl_queue SET status = 'queued' WHERE status = 'crawling'"
            ))
            if result.rowcount:
                log.info("Reset %d stuck crawling entries to queued", result.rowcount)
            result2 = await session.execute(sql_text(
                "UPDATE crawl_queue cq "
                "JOIN pages p ON cq.url_hash = p.url_hash "
                "SET cq.status = 'queued' "
                "WHERE p.status = 'indexing'"
            ))
            if result2.rowcount:
                log.info("Re-queued %d stuck indexing pages for reprocessing", result2.rowcount)
        except Exception as e:
            log.warning("Stuck entries reset skipped (will retry next restart): %s", e)
            await session.rollback()
        await session.commit()

    # --- Distributed mode initialization ---
    app.state.search_coordinator = None
    app.state.shard_manager = None
    app.state.remote_crawl_client = None
    app.state.worker_tunnel_hub = None

    if settings.distributed_mode and settings.hub_role == "hub":
        from semantic_searcher.services.search_coordinator import SearchCoordinator
        from semantic_searcher.services.shard_manager import ShardManager
        from semantic_searcher.services.worker_tunnel import WorkerTunnelHub

        # WebSocket tunnel hub (workers connect here — no port forwarding needed)
        tunnel_hub = WorkerTunnelHub()
        app.state.worker_tunnel_hub = tunnel_hub

        coordinator = SearchCoordinator()
        coordinator._tunnel_hub = tunnel_hub  # Use tunnels for search fanout
        await coordinator.start()
        app.state.search_coordinator = coordinator

        shard_mgr = ShardManager()
        await shard_mgr.start()
        app.state.shard_manager = shard_mgr

        # Create distributed tables
        async with async_session() as session:
            from sqlalchemy import text as sql_text
            for ddl in [
                "CREATE TABLE IF NOT EXISTS workers ("
                "  id INT AUTO_INCREMENT PRIMARY KEY,"
                "  worker_id VARCHAR(64) UNIQUE NOT NULL,"
                "  name VARCHAR(255) NOT NULL,"
                "  api_key_hash VARCHAR(128) NOT NULL,"
                "  endpoint_url VARCHAR(512) NOT NULL,"
                "  status ENUM('active','offline','draining','banned') DEFAULT 'active',"
                "  last_heartbeat DATETIME,"
                "  shard_ids JSON,"
                "  capabilities JSON,"
                "  reputation_score FLOAT DEFAULT 0.5,"
                "  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,"
                "  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP"
                ") ENGINE=InnoDB",
                "CREATE TABLE IF NOT EXISTS shards ("
                "  id INT AUTO_INCREMENT PRIMARY KEY,"
                "  shard_id VARCHAR(64) UNIQUE NOT NULL,"
                "  page_id_start BIGINT NOT NULL,"
                "  page_id_end BIGINT NOT NULL,"
                "  primary_worker_id VARCHAR(64),"
                "  replica_worker_ids JSON,"
                "  status ENUM('active','rebalancing','orphaned') DEFAULT 'active',"
                "  point_count INT DEFAULT 0,"
                "  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,"
                "  updated_at DATETIME DEFAULT CURRENT_TIMESTAMP"
                ") ENGINE=InnoDB",
                "CREATE TABLE IF NOT EXISTS crawl_job_assignments ("
                "  id BIGINT AUTO_INCREMENT PRIMARY KEY,"
                "  crawl_queue_id BIGINT NOT NULL,"
                "  worker_id VARCHAR(64) NOT NULL,"
                "  assigned_shard_id VARCHAR(64) NOT NULL,"
                "  status ENUM('assigned','in_progress','completed','failed') DEFAULT 'assigned',"
                "  assigned_at DATETIME DEFAULT CURRENT_TIMESTAMP,"
                "  completed_at DATETIME,"
                "  INDEX idx_cja_worker (worker_id),"
                "  INDEX idx_cja_queue (crawl_queue_id)"
                ") ENGINE=InnoDB",
                "CREATE TABLE IF NOT EXISTS worker_activity_log ("
                "  id BIGINT AUTO_INCREMENT PRIMARY KEY,"
                "  worker_id VARCHAR(64) NOT NULL,"
                "  ip_address VARCHAR(45),"
                "  status VARCHAR(20) NOT NULL,"
                "  ping_latency_ms FLOAT,"
                "  jobs_completed INT DEFAULT 0,"
                "  jobs_failed INT DEFAULT 0,"
                "  pages_indexed INT DEFAULT 0,"
                "  system_stats JSON,"
                "  logged_at DATETIME DEFAULT CURRENT_TIMESTAMP,"
                "  INDEX idx_wal_worker (worker_id),"
                "  INDEX idx_wal_time (logged_at)"
                ") ENGINE=InnoDB",
            ]:
                await session.execute(sql_text(ddl))
            await session.commit()
        log.info("Hub distributed mode: coordinator + shard manager started")

    elif settings.distributed_mode and settings.hub_role == "worker":
        from semantic_searcher.services.remote_crawl_client import RemoteCrawlClient
        from semantic_searcher.services.worker_search_handler import WorkerSearchHandler

        remote_client = RemoteCrawlClient(
            text_encoder=text_encoder,
            clip=clips[0],
            qdrant_client=qdrant_client,
            hub_url=settings.hub_url,
            api_key=settings.worker_api_key,
            worker_id=settings.worker_id,
        )
        app.state.remote_crawl_client = remote_client
        app.state.worker_search_handler = WorkerSearchHandler(qdrant_client)
        log.info("Worker distributed mode: remote crawl client + search handler initialized")

    # Auto-start all background services
    if settings.hub_role != "worker":
        ct_watcher.start()
        link_harvester.start()
        subdomain_enum.start()
        await crawler.start(seed_urls=[], max_depth=settings.crawler_max_depth, max_pages=0)
        log.info("All background services started (crawler, CT watcher, harvester, subdomain enum)")
    else:
        # Worker mode: start remote crawl client instead of local crawler
        if app.state.remote_crawl_client:
            await app.state.remote_crawl_client.start()
            # Start heartbeat loop
            async def _heartbeat_loop():
                while True:
                    await app.state.remote_crawl_client.send_heartbeat()
                    await _asyncio.sleep(settings.worker_heartbeat_interval)
            _asyncio.create_task(_heartbeat_loop())
        log.info("Worker mode: remote crawl client started")

    if not settings.crawl_only:
        # Pre-warm search cache with popular queries (background, non-blocking)
        async def _warmup_search_cache():
            try:
                async with async_session() as session:
                    from sqlalchemy import text as sql_text
                    result = await session.execute(sql_text(
                        "SELECT query, COUNT(*) as cnt FROM search_logs "
                        "GROUP BY query ORDER BY cnt DESC LIMIT 100"
                    ))
                    top_queries = [row[0] for row in result.fetchall()]
                if top_queries:
                    import concurrent.futures
                    loop = _asyncio.get_event_loop()
                    cached = await loop.run_in_executor(
                        None, search_svc.warmup_cache, top_queries
                    )
                    log.info("Cache warmup: pre-cached %d/%d popular queries", cached, len(top_queries))
            except Exception as e:
                log.warning("Cache warmup failed: %s", e)

        import asyncio
        asyncio.create_task(_warmup_search_cache())

        # Periodic page metadata refresh (every 2 minutes)
        async def _periodic_page_refresh():
            await _asyncio.sleep(120)
            while True:
                try:
                    async with async_session() as session:
                        new_pages, _, _ = await search_svc.incremental_update(session)
                except Exception as e:
                    log.warning("Page metadata refresh failed: %s", e)
                await _asyncio.sleep(120)

        _asyncio.create_task(_periodic_page_refresh())

    # Mark stale pages in Qdrant (background, runs once at startup then daily)
    # Uses the searcher's stale page set (computed from MySQL indexed_at dates)
    # to batch-update is_stale payload on Qdrant points by page_id.
    async def _mark_stale_pages():
        from qdrant_client.models import (
            FieldCondition, Filter, MatchValue, MatchAny,
        )
        from semantic_searcher.services.qdrant_collections import STALE_THRESHOLD_DAYS
        while True:
            try:
                stale_pids = search_svc._stale_pages
                if not stale_pids:
                    log.info("No stale pages to mark")
                    await _asyncio.sleep(86400)
                    continue

                # Process in batches of page_ids
                stale_list = list(stale_pids)
                total_marked = 0
                batch_size = 100  # page_ids per batch
                for i in range(0, len(stale_list), batch_size):
                    batch_pids = stale_list[i:i + batch_size]
                    # Scroll for points matching these page_ids that aren't already stale
                    scroll_offset = None
                    while True:
                        results, scroll_offset = qdrant_client.scroll(
                            collection_name="text_chunks",
                            scroll_filter=Filter(
                                must=[FieldCondition(key="page_id", match=MatchAny(any=batch_pids))],
                                must_not=[FieldCondition(key="is_stale", match=MatchValue(value=True))],
                            ),
                            limit=1000,
                            offset=scroll_offset,
                            with_payload=["page_id"],
                            with_vectors=False,
                        )
                        if not results:
                            break
                        qdrant_client.set_payload(
                            collection_name="text_chunks",
                            payload={"is_stale": True},
                            points=[p.id for p in results],
                            wait=False,
                        )
                        total_marked += len(results)
                        if scroll_offset is None:
                            break
                    # Yield to event loop every batch
                    if i % 1000 == 0 and i > 0:
                        await _asyncio.sleep(0.1)
                if total_marked:
                    log.info("Marked %d Qdrant points as stale (%d pages >%d days old)",
                             total_marked, len(stale_pids), STALE_THRESHOLD_DAYS)

                # Also set is_stale=False for fresh pages that don't have it yet
                all_pids = set(search_svc._pages.keys())
                fresh_pids = list(all_pids - stale_pids)
                total_fresh = 0
                for i in range(0, len(fresh_pids), batch_size):
                    batch_pids = fresh_pids[i:i + batch_size]
                    scroll_offset = None
                    while True:
                        from qdrant_client.models import IsNullCondition, PayloadField
                        results, scroll_offset = qdrant_client.scroll(
                            collection_name="text_chunks",
                            scroll_filter=Filter(must=[
                                FieldCondition(key="page_id", match=MatchAny(any=batch_pids)),
                                IsNullCondition(is_null=PayloadField(key="is_stale")),
                            ]),
                            limit=1000,
                            offset=scroll_offset,
                            with_payload=["page_id"],
                            with_vectors=False,
                        )
                        if not results:
                            break
                        qdrant_client.set_payload(
                            collection_name="text_chunks",
                            payload={"is_stale": False},
                            points=[p.id for p in results],
                            wait=False,
                        )
                        total_fresh += len(results)
                        if scroll_offset is None:
                            break
                    if i % 1000 == 0 and i > 0:
                        await _asyncio.sleep(0.1)
                if total_fresh:
                    log.info("Set is_stale=False on %d Qdrant points (fresh pages)", total_fresh)
            except Exception as e:
                log.warning("Stale page marking failed: %s", e)
            await _asyncio.sleep(86400)  # Re-check daily

    _asyncio.create_task(_mark_stale_pages())

    log.info("Semantic Searcher ready on port %d", settings.fastapi_port)
    yield

    # Shutdown
    if subdomain_enum.is_running:
        await subdomain_enum.stop()
    if link_harvester.is_running:
        await link_harvester.stop()
    if ct_watcher.is_running:
        await ct_watcher.stop()
    if crawler.is_running:
        await crawler.stop()
    if renderer.is_started:
        await renderer.stop()
    if search_queue is not None:
        search_queue.shutdown()
    qdrant_client.close()
    log.info("Semantic Searcher stopped")


app = FastAPI(title="Semantic Searcher", version="0.1.0", lifespan=lifespan)

from semantic_searcher.middleware.rate_limit import RateLimitMiddleware
app.add_middleware(RateLimitMiddleware, exclude_paths={"/", "/robots.txt", "/sitemap.xml", "/opensearch.xml"})

from pathlib import Path
from fastapi import Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, HTMLResponse
from semantic_searcher.routers import search, index, crawl, health, worker

app.include_router(search.router)
app.include_router(index.router)
app.include_router(crawl.router)
app.include_router(health.router)
if settings.distributed_mode:
    app.include_router(worker.router)

_static_dir = Path(__file__).resolve().parent.parent.parent / "static"
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


_index_html_cache: dict = {"html": "", "mtime": 0.0}


@app.get("/")
async def root(request: Request):
    host = request.headers.get("host", "")
    if host.startswith("edu."):
        return FileResponse(str(_static_dir / "edu_index.html"))
    # Cache the template in memory; reload only if file changes
    index_path = _static_dir / "index.html"
    mtime = index_path.stat().st_mtime
    if mtime != _index_html_cache["mtime"]:
        _index_html_cache["html"] = index_path.read_text()
        _index_html_cache["mtime"] = mtime
    # Use in-memory count only (no DB call); client polls /api/stats for accuracy
    search_svc = request.app.state.search_service
    page_count = len(search_svc._pages) if search_svc else 0
    from semantic_searcher.routers.health import _stats_cache
    if _stats_cache["count"] > page_count:
        page_count = _stats_cache["count"]
    html = _index_html_cache["html"].replace('/*__PAGE_COUNT__*/', f'var __PAGE_COUNT__={page_count};', 1)
    return HTMLResponse(html)


@app.get("/robots.txt")
async def robots_txt():
    return FileResponse(str(_static_dir / "robots.txt"), media_type="text/plain")


@app.get("/sitemap.xml")
async def sitemap_xml():
    return FileResponse(str(_static_dir / "sitemap.xml"), media_type="application/xml")


@app.get("/opensearch.xml")
async def opensearch_xml():
    return FileResponse(str(_static_dir / "opensearch.xml"), media_type="application/opensearchdescription+xml")


if __name__ == "__main__":
    import uvicorn
    config = uvicorn.Config(app, host="127.0.0.1", port=settings.fastapi_port)
    server = uvicorn.Server(config)
    server.run()
