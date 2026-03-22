import asyncio
import logging
from datetime import datetime, timezone
from pathlib import Path

from sqlalchemy import select, text

from semantic_searcher.config import settings
from semantic_searcher.database import async_session
from semantic_searcher.models.db import CrawlQueue, Page
from semantic_searcher.utils.link_extractor import extract_all_links
from semantic_searcher.utils.url_utils import normalize_url, url_hash

log = logging.getLogger(__name__)


class LinkHarvesterService:
    def __init__(self):
        self._running = False
        self._task: asyncio.Task | None = None
        self._pages_harvested = 0
        self._links_queued = 0

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> dict:
        return {
            "is_running": self._running,
            "pages_harvested": self._pages_harvested,
            "links_queued": self._links_queued,
        }

    def start(self):
        if self._running:
            log.warning("Link harvester already running")
            return
        self._running = True
        self._task = asyncio.create_task(self._run())
        log.info("Link harvester started")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        log.info("Link harvester stopped")

    async def _run(self):
        while self._running:
            try:
                count = await self._harvest_batch()
                if count > 0:
                    await asyncio.sleep(settings.harvester_batch_interval)
                else:
                    await asyncio.sleep(settings.harvester_idle_interval)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error("Link harvester error: %s", e)
                await asyncio.sleep(settings.harvester_idle_interval)

    async def _harvest_batch(self) -> int:
        """Fetch a batch of unharvested pages and process them. Returns count processed."""
        async with async_session() as session:
            result = await session.execute(
                select(Page.id, Page.url, Page.url_hash)
                .where(Page.status == "indexed")
                .where(Page.links_harvested_at.is_(None))
                .order_by(Page.id)
                .limit(settings.harvester_batch_size)
            )
            rows = result.all()

        if not rows:
            return 0

        count = 0
        for page_id, page_url, page_url_hash in rows:
            try:
                queued = await self._harvest_page(page_id, page_url, page_url_hash)
                self._links_queued += queued
                self._pages_harvested += 1
                count += 1
            except Exception as e:
                log.warning("Error harvesting page %d (%s): %s", page_id, page_url, e)
            finally:
                await self._mark_harvested(page_id)

        return count

    async def _harvest_page(self, page_id: int, page_url: str, page_url_hash: str) -> int:
        """Read cached HTML, extract links, queue new URLs. Returns count queued."""
        cache_path = Path(settings.html_cache_dir) / page_url_hash[:2] / f"{page_url_hash}.html"
        if not cache_path.exists():
            log.debug("No cached HTML for page %d, skipping link extraction", page_id)
            return 0

        html = cache_path.read_text(encoding="utf-8", errors="replace")
        links = extract_all_links(html, page_url)

        if not links:
            return 0

        # Deduplicate and normalize
        candidates = []
        seen: set[str] = set()
        for link in links:
            try:
                norm = normalize_url(link)
                uhash = url_hash(link)
                if uhash not in seen:
                    seen.add(uhash)
                    candidates.append((norm, uhash))
            except Exception:
                continue

        if not candidates:
            return 0

        async with async_session() as session:
            # Batch-check which hashes already exist in pages or crawl_seen
            all_hashes = [c[1] for c in candidates]
            existing: set[str] = set()
            for i in range(0, len(all_hashes), 500):
                batch_hashes = all_hashes[i:i + 500]
                result = await session.execute(
                    select(Page.url_hash).where(Page.url_hash.in_(batch_hashes))
                )
                existing.update(r[0] for r in result.all())
                # Also check crawl_seen (archived queue entries)
                seen_result = await session.execute(
                    text(
                        "SELECT url_hash FROM crawl_seen WHERE url_hash IN ("
                        + ", ".join([f":h{j}" for j in range(len(batch_hashes))])
                        + ")"
                    ),
                    {f"h{j}": h for j, h in enumerate(batch_hashes)},
                )
                existing.update(r[0] for r in seen_result.fetchall())

            new_links = [c for c in candidates if c[1] not in existing]
            if not new_links:
                return 0

            # Register in crawl_seen + crawl_queue
            conn = await session.connection()
            for i in range(0, len(new_links), 100):
                batch = new_links[i:i + 100]
                await conn.execute(
                    text(
                        "INSERT IGNORE INTO crawl_seen (url_hash) VALUES "
                        + ", ".join([f"(:h{j})" for j in range(len(batch))])
                    ),
                    {f"h{j}": uhash for j, (_, uhash) in enumerate(batch)},
                )
                await conn.execute(
                    text(
                        "INSERT IGNORE INTO crawl_queue (url, url_hash, depth, priority, source_page_id, status) VALUES "
                        + ", ".join(["(:u{0}, :h{0}, 0, 5.0, :s{0}, 'queued')".format(j) for j in range(len(batch))])
                    ),
                    {
                        k: v
                        for j, (norm, uhash) in enumerate(batch)
                        for k, v in [(f"u{j}", norm), (f"h{j}", uhash), (f"s{j}", page_id)]
                    },
                )
            await session.commit()

            queued = len(new_links)
            if queued:
                log.info(
                    "Harvested %d new links from page %d (%s)",
                    queued, page_id, page_url,
                )

        return queued

    async def _mark_harvested(self, page_id: int):
        """Mark page as harvested regardless of success/failure."""
        try:
            async with async_session() as session:
                await session.execute(
                    text(
                        "UPDATE pages SET links_harvested_at = :now WHERE id = :pid"
                    ),
                    {"now": datetime.now(timezone.utc), "pid": page_id},
                )
                await session.commit()
        except Exception as e:
            log.error("Failed to mark page %d as harvested: %s", page_id, e)
