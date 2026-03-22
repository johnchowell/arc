import asyncio
import json
import logging
import re
import time

import websockets

from semantic_searcher.config import settings
from semantic_searcher.database import async_session
from semantic_searcher.models.db import CrawlQueue
from semantic_searcher.utils.url_utils import normalize_url, url_hash

log = logging.getLogger(__name__)

_IP_RE = re.compile(r"^\d{1,3}(\.\d{1,3}){3}$")
_PRIVATE_TLDS = {".local", ".internal", ".test", ".localhost", ".invalid", ".example", ".onion"}


class CTWatcherService:
    def __init__(self, certstream_url: str | None = None):
        self.certstream_url = certstream_url or settings.certstream_url
        self._running = False
        self._connected = False
        self._task: asyncio.Task | None = None
        self._domains_discovered = 0
        self._domains_queued = 0
        self._pending: set[str] = set()
        self._last_flush = 0.0

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> dict:
        return {
            "is_running": self._running,
            "domains_discovered": self._domains_discovered,
            "domains_queued": self._domains_queued,
            "connected": self._connected,
        }

    def start(self):
        if self._running:
            log.warning("CT watcher already running")
            return
        self._running = True
        self._last_flush = time.monotonic()
        self._task = asyncio.create_task(self._run())
        log.info("CT watcher started, connecting to %s", self.certstream_url)

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        # Flush remaining domains
        if self._pending:
            await self._flush_to_db(self._pending)
            self._pending.clear()
        self._connected = False
        log.info("CT watcher stopped")

    async def _run(self):
        while self._running:
            try:
                async with websockets.connect(
                    self.certstream_url,
                    ping_interval=30,
                    ping_timeout=10,
                    close_timeout=5,
                ) as ws:
                    self._connected = True
                    log.info("CT watcher connected to certstream")
                    async for raw in ws:
                        if not self._running:
                            break
                        try:
                            msg = json.loads(raw)
                        except json.JSONDecodeError:
                            continue
                        domains = self._process_cert(msg)
                        if domains:
                            self._domains_discovered += len(domains)
                            self._pending.update(domains)
                        # Flush on batch size or time interval
                        elapsed = time.monotonic() - self._last_flush
                        if (
                            len(self._pending) >= settings.ct_flush_batch_size
                            or elapsed >= settings.ct_flush_interval
                        ):
                            if self._pending:
                                await self._flush_to_db(self._pending.copy())
                                self._pending.clear()
                                self._last_flush = time.monotonic()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                self._connected = False
                if self._running:
                    log.warning("CT watcher disconnected: %s — reconnecting in 5s", e)
                    await asyncio.sleep(5)

    def _process_cert(self, msg: dict) -> list[str]:
        """Extract domains from a certstream certificate_update message."""
        if msg.get("message_type") != "certificate_update":
            return []
        try:
            all_domains = msg["data"]["leaf_cert"]["all_domains"]
        except (KeyError, TypeError):
            return []
        result = []
        for raw in all_domains:
            cleaned = self._filter_domain(raw)
            if cleaned:
                result.append(cleaned)
        return result

    @staticmethod
    def _filter_domain(domain: str) -> str | None:
        """Clean and validate a domain from a certificate."""
        if not domain:
            return None
        # Strip wildcard prefix
        if domain.startswith("*."):
            domain = domain[2:]
        domain = domain.lower().strip(".")
        # Reject if no dot (e.g. "localhost")
        if "." not in domain:
            return None
        # Reject IP addresses
        if _IP_RE.match(domain):
            return None
        # Reject private/internal TLDs
        for tld in _PRIVATE_TLDS:
            if domain.endswith(tld):
                return None
        return domain

    async def _flush_to_db(self, domains: set[str]):
        """Insert discovered domains into crawl_queue, skipping duplicates."""
        if not domains:
            return
        try:
            candidates = []
            seen: set[str] = set()
            for domain in domains:
                url = f"https://{domain}/"
                norm = normalize_url(url)
                uhash = url_hash(url)
                if uhash not in seen:
                    seen.add(uhash)
                    candidates.append((norm, uhash))

            if not candidates:
                return

            from sqlalchemy import text
            queued = 0
            async with async_session() as session:
                conn = await session.connection()
                for i in range(0, len(candidates), 100):
                    batch = candidates[i:i + 100]
                    params = {
                        k: v
                        for j, (norm, uhash) in enumerate(batch)
                        for k, v in [(f"u{j}", norm), (f"h{j}", uhash)]
                    }
                    # Check crawl_seen to skip already-processed URLs
                    seen_result = await conn.execute(
                        text(
                            "SELECT url_hash FROM crawl_seen WHERE url_hash IN ("
                            + ", ".join([f":h{j}" for j in range(len(batch))])
                            + ")"
                        ),
                        {f"h{j}": uhash for j, (_, uhash) in enumerate(batch)},
                    )
                    already_seen = {r[0] for r in seen_result.fetchall()}
                    new_batch = [(n, h) for n, h in batch if h not in already_seen]
                    if not new_batch:
                        continue
                    new_params = {
                        k: v
                        for j, (norm, uhash) in enumerate(new_batch)
                        for k, v in [(f"u{j}", norm), (f"h{j}", uhash)]
                    }
                    await conn.execute(
                        text(
                            "INSERT IGNORE INTO crawl_seen (url_hash) VALUES "
                            + ", ".join([f"(:h{j})" for j in range(len(new_batch))])
                        ),
                        {f"h{j}": uhash for j, (_, uhash) in enumerate(new_batch)},
                    )
                    await conn.execute(
                        text(
                            "INSERT IGNORE INTO crawl_queue (url, url_hash, depth, priority, status) VALUES "
                            + ", ".join(["(:u{0}, :h{0}, 0, 10.0, 'queued')".format(j) for j in range(len(new_batch))])
                        ),
                        new_params,
                    )
                    queued += len(new_batch)
                await session.commit()
            self._domains_queued += queued
            if queued:
                log.info("CT watcher flushed %d domains to crawl queue", queued)
        except Exception as e:
            log.error("CT watcher flush error: %s", e)
