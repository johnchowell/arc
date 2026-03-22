"""Remote crawl client for distributed workers.

Pulls crawl jobs from the hub API, crawls locally, stores vectors in local Qdrant,
and reports metadata back to the hub.
"""

import asyncio
import logging
import time
from urllib.parse import urlparse

import httpx
from qdrant_client.models import Document, PointStruct

from semantic_searcher.config import settings
from semantic_searcher.services.clip_service import CLIPService
from semantic_searcher.services.text_encoder import TextEncoderService
from semantic_searcher.utils.content_extractor import extract_content, chunk_text
from semantic_searcher.utils.url_utils import normalize_url, url_hash, extract_domain, extract_tld_group
from semantic_searcher.utils.robots import RobotsCache, USER_AGENT

log = logging.getLogger(__name__)

_BROWSER_HEADERS = {
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "DNT": "1",
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
}


class RemoteCrawlClient:
    """Worker-side client that pulls jobs from hub and crawls locally."""

    def __init__(
        self,
        text_encoder: TextEncoderService,
        clip: CLIPService,
        qdrant_client,
        hub_url: str,
        api_key: str,
        worker_id: str,
    ):
        self._text_encoder = text_encoder
        self._clip = clip
        self._qdrant = qdrant_client
        self._hub_url = hub_url.rstrip("/")
        self._api_key = api_key
        self._worker_id = worker_id
        self._running = False
        self._task = None
        self._robots = RobotsCache()
        self._domain_last_request: dict[str, float] = {}

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._api_key}",
            "X-Worker-ID": self._worker_id,
        }

    async def start(self):
        self._running = True
        self._task = asyncio.create_task(self._run())
        log.info("Remote crawl client started (hub=%s, worker=%s)", self._hub_url, self._worker_id)

    async def stop(self):
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self):
        """Main loop: pull jobs → crawl → index locally → report back."""
        async with httpx.AsyncClient(
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT, **_BROWSER_HEADERS},
            timeout=httpx.Timeout(connect=5, read=30, write=10, pool=10),
        ) as crawl_client:
            hub_client = httpx.AsyncClient(timeout=30)
            try:
                while self._running:
                    try:
                        # Pull jobs from hub
                        resp = await hub_client.get(
                            f"{self._hub_url}/api/worker/crawl-jobs?batch_size=50",
                            headers=self._headers(),
                        )
                        if resp.status_code != 200:
                            log.warning("Hub returned %d for crawl-jobs", resp.status_code)
                            await asyncio.sleep(10)
                            continue

                        data = resp.json()
                        jobs = data.get("jobs", [])
                        if not jobs:
                            await asyncio.sleep(5)
                            continue

                        # Process jobs concurrently
                        sem = asyncio.Semaphore(20)
                        tasks = [
                            self._process_job(crawl_client, hub_client, job, sem)
                            for job in jobs
                        ]
                        await asyncio.gather(*tasks, return_exceptions=True)

                    except asyncio.CancelledError:
                        raise
                    except Exception as e:
                        log.error("Crawl client error: %s", e)
                        await asyncio.sleep(10)
            finally:
                await hub_client.aclose()

    async def _process_job(self, crawl_client: httpx.AsyncClient,
                           hub_client: httpx.AsyncClient, job: dict,
                           sem: asyncio.Semaphore):
        """Crawl a single URL, index locally, report to hub."""
        async with sem:
            url = job["url"]
            job_id = job["job_id"]
            shard_id = job["target_shard_id"]

            try:
                # Rate limit per domain
                domain = urlparse(url).netloc
                now = time.time()
                last = self._domain_last_request.get(domain, 0)
                wait = settings.crawler_rate_limit - (now - last)
                if wait > 0:
                    await asyncio.sleep(wait)
                self._domain_last_request[domain] = time.time()

                # Fetch
                resp = await crawl_client.get(url)
                content_type = resp.headers.get("content-type", "")
                if "text/html" not in content_type or resp.status_code != 200:
                    await self._report_complete(hub_client, job_id, "skipped")
                    return

                html = resp.text
                if len(html) < 100:
                    await self._report_complete(hub_client, job_id, "skipped")
                    return

                # Extract content
                content = extract_content(html, url)
                chunks = chunk_text(content.text)
                if not chunks:
                    await self._report_complete(hub_client, job_id, "skipped")
                    return

                # Encode with MPNet
                text_vectors = await self._text_encoder.encode_texts_async(chunks)

                # Store in local Qdrant
                title_str = content.title or ""
                points = []
                for i, (chunk, vec) in enumerate(zip(chunks, text_vectors)):
                    bm25_text = f"{title_str} {chunk}".strip()
                    point_id = hash(f"{url}:{i}") & 0x7FFFFFFFFFFFFFFF  # positive int64
                    points.append(PointStruct(
                        id=point_id,
                        vector={
                            "dense": vec.tolist(),
                            "text-bm25": Document(text=bm25_text, model="Qdrant/bm25"),
                        },
                        payload={
                            "page_id": job_id,  # Use job_id as page_id for now
                            "chunk_index": i,
                            "chunk_text": chunk,
                            "url": url,
                            "title": title_str,
                            "domain": extract_domain(url),
                            "tld_group": extract_tld_group(url),
                            "shard_id": shard_id,
                        },
                    ))

                if points:
                    self._qdrant.upsert("text_chunks_v2", points=points, wait=False)

                # Report metadata to hub
                domain = extract_domain(url)
                await self._report_complete(hub_client, job_id, "completed", metadata={
                    "url": url,
                    "url_hash": job["url_hash"],
                    "title": content.title,
                    "meta_description": content.meta_description,
                    "domain": domain,
                    "tld_group": extract_tld_group(url),
                    "language": None,
                    "nsfw_flag": False,
                    "chunk_count": len(chunks),
                    "image_count": len(content.image_urls) if content.image_urls else 0,
                })

            except Exception as e:
                log.error("Job %d (%s) failed: %s", job_id, url[:80], e)
                await self._report_complete(hub_client, job_id, "failed")

    async def _report_complete(self, client: httpx.AsyncClient, job_id: int,
                               status: str, metadata: dict | None = None):
        """Report job completion to hub."""
        try:
            await client.post(
                f"{self._hub_url}/api/worker/crawl-complete",
                json={"job_id": job_id, "status": status, "metadata": metadata},
                headers=self._headers(),
            )
        except Exception as e:
            log.warning("Failed to report job %d: %s", job_id, e)

    async def send_heartbeat(self):
        """Send heartbeat to hub."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                await client.post(
                    f"{self._hub_url}/api/worker/heartbeat",
                    json={"system_stats": {}},
                    headers=self._headers(),
                )
        except Exception:
            pass
