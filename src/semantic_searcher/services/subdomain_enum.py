import asyncio
import logging

import dns.asyncresolver
import dns.exception
import dns.name
import dns.rdatatype
import dns.resolver
from sqlalchemy import select, union

from semantic_searcher.config import settings
from semantic_searcher.database import async_session
from semantic_searcher.models.db import CrawlQueue, Page
from semantic_searcher.utils.subdomain_wordlist import SUBDOMAIN_PREFIXES
from semantic_searcher.utils.url_utils import normalize_url, url_hash

log = logging.getLogger(__name__)


def _extract_root_domain(hostname: str) -> str:
    """Extract root domain (last two labels) from a hostname."""
    parts = hostname.rstrip(".").split(".")
    if len(parts) >= 2:
        return ".".join(parts[-2:])
    return hostname


class SubdomainEnumeratorService:
    def __init__(self):
        self._running = False
        self._task: asyncio.Task | None = None
        self._seen_domains: set[str] = set()
        self._domains_enumerated = 0
        self._subdomains_queued = 0
        self._last_crawl_queue_id = 0
        self._last_page_id = 0

    @property
    def is_running(self) -> bool:
        return self._running

    @property
    def stats(self) -> dict:
        return {
            "is_running": self._running,
            "domains_enumerated": self._domains_enumerated,
            "subdomains_queued": self._subdomains_queued,
        }

    def start(self):
        if self._running:
            log.warning("Subdomain enumerator already running")
            return
        self._running = True
        self._task = asyncio.create_task(self._run())
        log.info("Subdomain enumerator started")

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None
        log.info("Subdomain enumerator stopped")

    async def _run(self):
        while self._running:
            try:
                domains = await self._get_new_domains()
                if domains:
                    for domain in domains:
                        if not self._running:
                            break
                        subs = await self._enumerate_domain(domain)
                        if subs:
                            queued = await self._queue_subdomains(subs)
                            self._subdomains_queued += queued
                        self._seen_domains.add(domain)
                        self._domains_enumerated += 1
                        log.info(
                            "Enumerated %s: %d subdomains found",
                            domain, len(subs),
                        )
                    await asyncio.sleep(1.0)
                else:
                    await asyncio.sleep(settings.enum_idle_interval)
            except asyncio.CancelledError:
                raise
            except Exception as e:
                log.error("Subdomain enumerator error: %s", e)
                await asyncio.sleep(settings.enum_idle_interval)

    async def _get_new_domains(self) -> list[str]:
        """Find root domains from recently added crawl_queue and pages entries."""
        from sqlalchemy import text

        async with async_session() as session:
            # Only scan rows added since last check (incremental, not full table scan)
            result = await session.execute(text(
                "SELECT DISTINCT SUBSTRING_INDEX(SUBSTRING_INDEX("
                "SUBSTRING_INDEX(url, '/', 3), '/', -1), ':', 1) AS host "
                "FROM crawl_queue WHERE id > :last_id "
                "UNION "
                "SELECT DISTINCT SUBSTRING_INDEX(SUBSTRING_INDEX("
                "SUBSTRING_INDEX(url, '/', 3), '/', -1), ':', 1) AS host "
                "FROM pages WHERE id > :last_page_id"
            ), {"last_id": self._last_crawl_queue_id, "last_page_id": self._last_page_id})
            hostnames = [row[0] for row in result.all()]

            # Update high-water marks
            max_cq = await session.execute(text("SELECT MAX(id) FROM crawl_queue"))
            max_pg = await session.execute(text("SELECT MAX(id) FROM pages"))
            cq_max = max_cq.scalar() or 0
            pg_max = max_pg.scalar() or 0
            self._last_crawl_queue_id = cq_max
            self._last_page_id = pg_max

        root_domains: set[str] = set()
        for hostname in hostnames:
            try:
                if hostname:
                    root = _extract_root_domain(hostname)
                    if root and root not in self._seen_domains:
                        root_domains.add(root)
            except Exception:
                continue

        return list(root_domains)[:settings.enum_batch_size]

    async def _enumerate_domain(self, domain: str) -> set[str]:
        """Probe DNS for live subdomains of the given domain."""
        discovered: set[str] = set()
        sem = asyncio.Semaphore(settings.enum_dns_concurrency)
        resolver = dns.asyncresolver.Resolver()
        resolver.lifetime = settings.enum_dns_timeout

        async def probe(prefix: str):
            fqdn = f"{prefix}.{domain}"
            async with sem:
                try:
                    await resolver.resolve(fqdn, "A")
                    discovered.add(fqdn)
                except (
                    dns.resolver.NXDOMAIN,
                    dns.resolver.NoAnswer,
                    dns.resolver.NoNameservers,
                    dns.exception.Timeout,
                    dns.name.EmptyLabel,
                    Exception,
                ):
                    pass

        # Brute-force subdomain prefixes
        tasks = [probe(prefix) for prefix in SUBDOMAIN_PREFIXES]
        await asyncio.gather(*tasks)

        # Record mining: MX and NS
        for rdtype in ("MX", "NS"):
            try:
                answers = await resolver.resolve(domain, rdtype)
                for rdata in answers:
                    hostname = str(rdata.exchange if rdtype == "MX" else rdata.target).rstrip(".")
                    if hostname.endswith(f".{domain}"):
                        discovered.add(hostname)
            except Exception:
                pass

        return discovered

    async def _queue_subdomains(self, subdomains: set[str]) -> int:
        """Queue discovered subdomains for crawling. Returns count queued."""
        # Deduplicate and normalize
        candidates = []
        seen: set[str] = set()
        for sub in subdomains:
            try:
                url = normalize_url(f"https://{sub}/")
                uhash = url_hash(f"https://{sub}/")
                if uhash not in seen:
                    seen.add(uhash)
                    candidates.append((url, uhash))
            except Exception:
                continue

        if not candidates:
            return 0

        async with async_session() as session:
            # Batch-check which hashes already exist in pages or crawl_seen
            all_hashes = [c[1] for c in candidates]
            existing: set[str] = set()
            from sqlalchemy import text
            for i in range(0, len(all_hashes), 500):
                batch_hashes = all_hashes[i:i + 500]
                result = await session.execute(
                    select(Page.url_hash).where(Page.url_hash.in_(batch_hashes))
                )
                existing.update(r[0] for r in result.all())
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
                        "INSERT IGNORE INTO crawl_queue (url, url_hash, depth, priority, status) VALUES "
                        + ", ".join(["(:u{0}, :h{0}, 0, 3.0, 'queued')".format(j) for j in range(len(batch))])
                    ),
                    {
                        k: v
                        for j, (url, uhash) in enumerate(batch)
                        for k, v in [(f"u{j}", url), (f"h{j}", uhash)]
                    },
                )
            await session.commit()

        return len(new_links)
