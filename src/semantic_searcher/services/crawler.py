import asyncio
import datetime
import logging
import re
import time
from urllib.parse import urlparse

import httpx
from sqlalchemy import select, func, update, text
from semantic_searcher.config import settings
from semantic_searcher.database import async_session
from semantic_searcher.models.db import CrawlQueue, CrawlState
from semantic_searcher.services.indexer import IndexerService
from semantic_searcher.utils.robots import RobotsCache, USER_AGENT
from semantic_searcher.utils.url_utils import normalize_url, url_hash

log = logging.getLogger(__name__)


_DOMAIN_FAIL_THRESHOLD = 3      # failures before blacklisting a domain
_DOMAIN_BLACKLIST_SECS = 7200   # blacklist duration (2 hours)

# URLs matching these patterns are junk — skip without fetching
_JUNK_URL_RE = re.compile(
    r'\.pdf$|\.docx?$|\.xlsx?$|\.pptx?$|\.zip$|\.rar$|\.7z$|\.tar|\.gz$'
    r'|\.mp[34]$|\.avi$|\.mov$|\.wmv$|\.flv$|\.webm$|\.mkv$'
    r'|\.jpe?g$|\.png$|\.gif$|\.svg$|\.ico$|\.bmp$|\.tiff?$|\.webp$'
    r'|\.css$|\.js$|\.woff2?$|\.ttf$|\.eot$'
    r'|[?&]action=edit|[?&]action=history|[?&]oldid=|[?&]diff='
    r'|/Special:|/Spezial:|/Speciaal:|/Especial:|/Spécial:|/特別:'
    r'|/w/index\.php\?.*(?:action|printable|redlink)'
    r'|/submit\?|/share\?|/login|/signup|/register'
    r'|/feed/?$|/rss/?$|/atom/?$',
    re.IGNORECASE,
)

_BROWSER_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Sec-Fetch-User": "?1",
}


class CrawlerService:
    def __init__(self, indexer: IndexerService):
        self.indexer = indexer
        self._running = False
        self._task: asyncio.Task | None = None
        self._index_tasks: list[asyncio.Task] = []
        self._index_queue: asyncio.Queue | None = None
        self._robots = RobotsCache()
        self._domain_last_request: dict[str, float] = {}
        self._indexing_urls: set[str] = set()  # guard against duplicate indexing
        # Domain failure tracking: domain -> (fail_count, blacklisted_until)
        self._domain_failures: dict[str, list] = {}  # domain -> [fail_count, blacklisted_until]

    @property
    def is_running(self) -> bool:
        return self._running

    async def start(self, seed_urls: list[str], max_depth: int = 3, max_pages: int = 10000):
        if self._running:
            log.warning("Crawler already running")
            return
        self._running = True

        # Persist crawl state
        async with async_session() as session:
            state = CrawlState(
                is_running=True,
                seed_urls=seed_urls,
                max_depth=max_depth,
                max_pages=max_pages,
                pages_crawled=0,
                started_at=datetime.datetime.utcnow(),
            )
            session.add(state)
            await session.flush()
            state_id = state.id

            # Enqueue seeds
            for url in seed_urls:
                norm = normalize_url(url)
                uhash = url_hash(url)
                existing = await session.execute(
                    select(CrawlQueue).where(CrawlQueue.url_hash == uhash)
                )
                if existing.scalar_one_or_none() is None:
                    session.add(CrawlQueue(url=norm, url_hash=uhash, depth=0, priority=0.0, status="queued"))
            await session.commit()

        self._max_depth = max_depth
        self._index_queue = asyncio.Queue(maxsize=200)
        # One index worker per GPU for true parallel encoding
        self._index_tasks = []
        for i, clip in enumerate(self.indexer.clips):
            task = asyncio.create_task(self._index_worker(clip, worker_id=i))
            self._index_tasks.append(task)
        log.info("Started %d index workers", len(self._index_tasks))
        self._task = asyncio.create_task(
            self._run(state_id, max_depth, max_pages)
        )

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        # Send one sentinel per worker
        if self._index_queue:
            for _ in self._index_tasks:
                await self._index_queue.put(None)
        for task in self._index_tasks:
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._index_tasks = []
        # Update crawl state
        async with async_session() as session:
            result = await session.execute(
                select(CrawlState).where(CrawlState.is_running == True).order_by(CrawlState.id.desc()).limit(1)
            )
            state = result.scalar_one_or_none()
            if state:
                state.is_running = False
                state.stopped_at = datetime.datetime.utcnow()
                await session.commit()

    async def status(self) -> dict:
        async with async_session() as session:
            # Get latest crawl state
            result = await session.execute(
                select(CrawlState).order_by(CrawlState.id.desc()).limit(1)
            )
            state = result.scalar_one_or_none()

            # Queue stats
            queue_result = await session.execute(
                select(CrawlQueue.status, func.count()).group_by(CrawlQueue.status)
            )
            queue_stats = dict(queue_result.all())

            return {
                "is_running": self._running,
                "pages_crawled": state.pages_crawled if state else 0,
                "queue_size": queue_stats.get("queued", 0),
                "crawling": queue_stats.get("crawling", 0),
                "done": queue_stats.get("done", 0),
                "failed": queue_stats.get("failed", 0),
                "max_pages": state.max_pages if state else 0,
                "max_depth": state.max_depth if state else 0,
                "started_at": state.started_at.isoformat() if state and state.started_at else None,
            }

    async def _run(self, state_id: int, max_depth: int, max_pages: int):
        pages_crawled = 0
        sem = asyncio.Semaphore(settings.crawler_workers)

        try:
            pool_limits = httpx.Limits(
                max_connections=settings.crawler_workers,
                max_keepalive_connections=settings.crawler_workers // 2,
                keepalive_expiry=30,
            )
            async with httpx.AsyncClient(
                follow_redirects=True,
                headers={"User-Agent": USER_AGENT, **_BROWSER_HEADERS},
                timeout=httpx.Timeout(connect=5, read=30, write=10, pool=10),
                limits=pool_limits,
            ) as client:
                while self._running:
                    try:
                        # Fetch batch from queue — domain-diverse sampling
                        async with async_session() as session:
                            import random
                            batch_size = settings.crawler_workers * 2
                            # Oversample heavily to compensate for blacklisted domains
                            n_blacklisted = len([d for d in self._domain_failures if self._is_domain_blacklisted(d)])
                            oversample = max(10, min(50, n_blacklisted // 10 + 10))
                            pool_size = batch_size * oversample

                            # Get ID range for random probing
                            id_range = await session.execute(text(
                                "SELECT MIN(id), MAX(id) FROM crawl_queue "
                                "WHERE status = 'queued'"
                            ))
                            id_min, id_max = id_range.one()

                            # Probe random offsets within the ID range to get diverse samples
                            rand_rows: list[tuple[int, str]] = []
                            if id_min is not None and id_max is not None:
                                n_probes = min(50, id_max - id_min + 1)
                                probes = sorted(random.sample(
                                    range(id_min, id_max + 1),
                                    n_probes,
                                ))
                                for probe_id in probes:
                                    r = await session.execute(text(
                                        "SELECT id, url FROM crawl_queue "
                                        "WHERE status = 'queued' AND id >= :probe "
                                        "ORDER BY id LIMIT :lim"
                                    ), {"probe": probe_id, "lim": pool_size // 10})
                                    rand_rows.extend(r.fetchall())

                            # Also get some top-priority rows to avoid starving high-priority items
                            top_result = await session.execute(text(
                                "SELECT id, url FROM crawl_queue "
                                "WHERE status = 'queued' "
                                "ORDER BY priority DESC, id "
                                "LIMIT :lim"
                            ), {"lim": pool_size // 2})
                            top_rows = top_result.fetchall()

                            # Merge and dedup
                            seen_ids: set[int] = set()
                            merged: list[tuple[int, str]] = []
                            # Interleave: alternate rand and top for balanced mix
                            ri, ti = 0, 0
                            while ri < len(rand_rows) or ti < len(top_rows):
                                if ri < len(rand_rows):
                                    row = rand_rows[ri]; ri += 1
                                    if row[0] not in seen_ids:
                                        seen_ids.add(row[0])
                                        merged.append(row)
                                if ti < len(top_rows):
                                    row = top_rows[ti]; ti += 1
                                    if row[0] not in seen_ids:
                                        seen_ids.add(row[0])
                                        merged.append(row)

                            # Domain-diverse selection: skip blacklisted + junk URLs, max 3 per domain
                            seen_domains: dict[str, int] = {}
                            diverse_ids: list[int] = []
                            skipped_blacklisted = 0
                            skipped_junk = 0
                            max_per_domain = 3
                            for row_id, row_url in merged:
                                # Skip junk URLs (PDFs, edit pages, media files, etc.)
                                if _JUNK_URL_RE.search(row_url):
                                    skipped_junk += 1
                                    continue
                                try:
                                    domain = urlparse(row_url).netloc
                                except Exception:
                                    domain = ""
                                # Skip blacklisted domains at selection time
                                if self._is_domain_blacklisted(domain):
                                    skipped_blacklisted += 1
                                    continue
                                count = seen_domains.get(domain, 0)
                                if count < max_per_domain:
                                    diverse_ids.append(row_id)
                                    seen_domains[domain] = count + 1
                                    if len(diverse_ids) >= batch_size:
                                        break
                            if skipped_blacklisted or skipped_junk:
                                log.info("Batch selection: skipped %d blacklisted + %d junk URLs",
                                         skipped_blacklisted, skipped_junk)

                            if diverse_ids:
                                items_result = await session.execute(
                                    select(CrawlQueue).where(CrawlQueue.id.in_(diverse_ids))
                                )
                                items = list(items_result.scalars().all())
                            else:
                                items = []
                            if not items:
                                log.info("Crawl queue empty, waiting 30s...")
                                await asyncio.sleep(30)
                                continue
                            log.info("Crawler batch: %d items, index_queue=%d/%d",
                                     len(items), self._index_queue.qsize(), self._index_queue.maxsize)
                            # Mark as crawling
                            for item in items:
                                item.status = "crawling"
                            await session.commit()

                        # Process batch concurrently with per-task and batch-level timeouts
                        PER_TASK_TIMEOUT = 120  # seconds per URL
                        BATCH_TIMEOUT = 300     # seconds for entire batch

                        async def _timed_crawl(item):
                            try:
                                return await asyncio.wait_for(
                                    self._crawl_one(client, item, max_depth, sem),
                                    timeout=PER_TASK_TIMEOUT,
                                )
                            except asyncio.TimeoutError:
                                log.warning("Crawl timeout (%ds): %s", PER_TASK_TIMEOUT, item.url[:120])
                                try:
                                    async with async_session() as s:
                                        result = await s.execute(
                                            select(CrawlQueue).where(CrawlQueue.id == item.id)
                                        )
                                        q = result.scalar_one_or_none()
                                        if q and q.status == "crawling":
                                            q.status = "failed"
                                            await s.commit()
                                except Exception:
                                    pass
                                return False

                        tasks = []
                        for item in items:
                            if not self._running:
                                break
                            tasks.append(_timed_crawl(item))

                        try:
                            results = await asyncio.wait_for(
                                asyncio.gather(*tasks, return_exceptions=True),
                                timeout=BATCH_TIMEOUT,
                            )
                        except asyncio.TimeoutError:
                            log.error("Batch timeout (%ds) — resetting stuck entries", BATCH_TIMEOUT)
                            # Reset any still-crawling items back to queued
                            batch_ids = [item.id for item in items]
                            try:
                                async with async_session() as s:
                                    await s.execute(
                                        update(CrawlQueue)
                                        .where(CrawlQueue.id.in_(batch_ids), CrawlQueue.status == "crawling")
                                        .values(status="queued")
                                    )
                                    await s.commit()
                            except Exception:
                                pass
                            results = []

                        queued_count = 0
                        err_count = 0
                        for r in results:
                            if isinstance(r, Exception):
                                log.error("Crawl error: %s", r)
                                err_count += 1
                            elif r:
                                pages_crawled += 1
                                queued_count += 1
                        log.info("Crawler batch done: %d queued for indexing, %d errors, %d total crawled",
                                 queued_count, err_count, pages_crawled)

                        # Update state
                        async with async_session() as session:
                            await session.execute(
                                update(CrawlState)
                                .where(CrawlState.id == state_id)
                                .values(pages_crawled=pages_crawled)
                            )
                            await session.commit()

                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        log.error("Crawler batch error (will retry in 10s): %s", e, exc_info=True)
                        await asyncio.sleep(10)

        except asyncio.CancelledError:
            pass
        except Exception as e:
            log.error("Crawler _run died unexpectedly: %s", e, exc_info=True)

        # Mark stopped
        self._running = False
        try:
            async with async_session() as session:
                await session.execute(
                    update(CrawlState)
                    .where(CrawlState.id == state_id)
                    .values(is_running=False, stopped_at=datetime.datetime.utcnow())
                )
                await session.commit()
        except Exception:
            pass
        log.info("Crawler stopped: %d pages crawled", pages_crawled)

    async def _crawl_one(
        self, client: httpx.AsyncClient, item: CrawlQueue, max_depth: int, sem: asyncio.Semaphore
    ) -> bool:
        async with sem:
            url = item.url
            depth = item.depth

            try:
                # Skip junk URLs
                if _JUNK_URL_RE.search(url):
                    async with async_session() as session:
                        result = await session.execute(
                            select(CrawlQueue).where(CrawlQueue.id == item.id)
                        )
                        q = result.scalar_one()
                        q.status = "skipped"
                        await session.commit()
                    return False

                # Rate limit per domain
                domain = urlparse(url).netloc

                # Skip blacklisted domains
                if self._is_domain_blacklisted(domain):
                    async with async_session() as session:
                        result = await session.execute(
                            select(CrawlQueue).where(CrawlQueue.id == item.id)
                        )
                        q = result.scalar_one()
                        q.status = "skipped"
                        await session.commit()
                    return False

                await self._rate_limit(domain)

                # Check robots.txt
                if not await self._robots.can_fetch(url, client):
                    async with async_session() as session:
                        result = await session.execute(
                            select(CrawlQueue).where(CrawlQueue.id == item.id)
                        )
                        q = result.scalar_one()
                        q.status = "skipped"
                        await session.commit()
                    return False

                # Fetch (uses client-level timeout: connect=5s, read=30s)
                resp = await client.get(url)
                content_type = resp.headers.get("content-type", "")
                if "text/html" not in content_type or resp.status_code != 200:
                    async with async_session() as session:
                        result = await session.execute(
                            select(CrawlQueue).where(CrawlQueue.id == item.id)
                        )
                        q = result.scalar_one()
                        q.status = "skipped"
                        await session.commit()
                    return False

                html = resp.text
                if len(html) < 100:
                    async with async_session() as session:
                        result = await session.execute(
                            select(CrawlQueue).where(CrawlQueue.id == item.id)
                        )
                        q = result.scalar_one()
                        q.status = "skipped"
                        await session.commit()
                    return False

                # Dedup: skip if another worker is already indexing this URL
                if url in self._indexing_urls:
                    async with async_session() as session:
                        result = await session.execute(
                            select(CrawlQueue).where(CrawlQueue.id == item.id)
                        )
                        q = result.scalar_one()
                        q.status = "skipped"
                        await session.commit()
                    return False
                self._indexing_urls.add(url)

                # Queue for async indexing (non-blocking)
                await self._index_queue.put((url, html, depth, item.id))
                self._record_domain_success(domain)
                return True

            except Exception as e:
                domain = urlparse(url).netloc
                self._record_domain_failure(domain)
                log.error("Error crawling %s: %s", url, e)
                try:
                    async with async_session() as session:
                        result = await session.execute(
                            select(CrawlQueue).where(CrawlQueue.id == item.id)
                        )
                        q = result.scalar_one_or_none()
                        if q:
                            q.status = "failed"
                            await session.commit()
                except Exception:
                    pass
                return False

    async def _index_worker(self, clip: "CLIPService", worker_id: int = 0):
        """Background worker that batches pages for GPU-efficient CLIP encoding.

        Each worker owns a dedicated CLIP instance (GPU) and pulls from the shared queue.
        """
        BATCH_SIZE = settings.index_batch_size
        BATCH_TIMEOUT = settings.index_batch_timeout
        tag = f"[worker-{worker_id}/{clip.device}]"

        while True:
            batch = []
            try:
                # Collect a batch
                item = await self._index_queue.get()
                if item is None:
                    break
                batch.append(item)

                # Try to fill the batch within timeout
                deadline = asyncio.get_event_loop().time() + BATCH_TIMEOUT
                while len(batch) < BATCH_SIZE:
                    remaining = deadline - asyncio.get_event_loop().time()
                    if remaining <= 0:
                        break
                    try:
                        item = await asyncio.wait_for(self._index_queue.get(), timeout=remaining)
                        if item is None:
                            # Put sentinel back and process what we have
                            await self._index_queue.put(None)
                            break
                        batch.append(item)
                    except asyncio.TimeoutError:
                        break

                if not batch:
                    continue

                # Batch index using this worker's dedicated CLIP instance
                items_for_indexer = [(url, html) for url, html, depth, queue_id in batch]
                try:
                    await self.indexer.index_batch(items_for_indexer, clip=clip)

                    # Mark queue entries as done in a separate transaction
                    # (link extraction is handled by the link harvester service)
                    async with async_session() as session:
                        for url, html, depth, queue_id in batch:
                            try:
                                result = await session.execute(
                                    select(CrawlQueue).where(CrawlQueue.id == queue_id)
                                )
                                q = result.scalar_one_or_none()
                                if q:
                                    q.status = "done"
                                    q.crawled_at = datetime.datetime.utcnow()
                            except Exception as e:
                                log.error("%s Post-index error for %s: %s", tag, url, e)
                        await session.commit()

                except Exception as e:
                    log.error("%s Batch index error: %s", tag, e)
                    async with async_session() as session:
                        for url, html, depth, queue_id in batch:
                            try:
                                result = await session.execute(
                                    select(CrawlQueue).where(CrawlQueue.id == queue_id)
                                )
                                q = result.scalar_one_or_none()
                                if q:
                                    q.status = "failed"
                            except Exception:
                                pass
                        await session.commit()

                log.info("%s Batch indexed %d pages", tag, len(batch))

                # Clean up dedup set
                for url, html, depth, queue_id in batch:
                    self._indexing_urls.discard(url)

            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error("%s Index worker error: %s", tag, e)
                for url, html, depth, queue_id in batch:
                    self._indexing_urls.discard(url)

    def _is_domain_blacklisted(self, domain: str) -> bool:
        """Check if a domain is temporarily blacklisted due to repeated failures."""
        entry = self._domain_failures.get(domain)
        if entry is None:
            return False
        fail_count, blacklisted_until = entry
        if blacklisted_until and time.time() < blacklisted_until:
            return True
        if blacklisted_until and time.time() >= blacklisted_until:
            # Blacklist expired, reset
            self._domain_failures[domain] = [0, 0]
            return False
        return False

    def _record_domain_failure(self, domain: str):
        """Record a failure for a domain and blacklist if threshold exceeded."""
        entry = self._domain_failures.get(domain)
        if entry is None:
            self._domain_failures[domain] = [1, 0]
        else:
            entry[0] += 1
            if entry[0] >= _DOMAIN_FAIL_THRESHOLD and not entry[1]:
                entry[1] = time.time() + _DOMAIN_BLACKLIST_SECS
                log.info("Domain blacklisted for %ds: %s (%d failures)",
                         _DOMAIN_BLACKLIST_SECS, domain, entry[0])

    def _record_domain_success(self, domain: str):
        """Reset failure count on success."""
        if domain in self._domain_failures:
            self._domain_failures[domain] = [0, 0]

    async def _rate_limit(self, domain: str):
        now = time.time()
        last = self._domain_last_request.get(domain, 0)
        wait = settings.crawler_rate_limit - (now - last)
        if wait > 0:
            await asyncio.sleep(wait)
        self._domain_last_request[domain] = time.time()
