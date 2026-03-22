import gzip
import logging
import random
import re
from pathlib import Path

import httpx

from semantic_searcher.config import settings

log = logging.getLogger(__name__)

CC_BASE = "https://data.commoncrawl.org/projects/hyperlinkgraph/"


class CommonCrawlSeeder:
    def __init__(self, cache_dir: Path | None = None):
        self.cache_dir = cache_dir or Path(settings.webindex_dir) / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._domains: list[str] | None = None

    def _get_latest_vertex_url(self) -> str:
        """Fetch CC hyperlink graph directory to find the latest release."""
        resp = httpx.get(CC_BASE, follow_redirects=True, timeout=30)
        resp.raise_for_status()
        # Parse directory listing for release links like "cc-main-2024-feb-apr-may/"
        releases = re.findall(r'href="(cc-main-[^"]+/)"', resp.text)
        if not releases:
            raise RuntimeError("No Common Crawl hyperlink graph releases found")
        latest = sorted(releases)[-1]
        return f"{CC_BASE}{latest}domain/vertices/part-00000.gz"

    def download(self, force: bool = False) -> Path:
        """Download and parse one CC vertex partition into a domain list."""
        cache_file = self.cache_dir / "cc_domains.txt"
        if cache_file.exists() and not force:
            log.info("CC domain cache already exists at %s", cache_file)
            return cache_file

        url = self._get_latest_vertex_url()
        log.info("Downloading CC vertex partition from %s", url)

        gz_path = self.cache_dir / "vertices-part-00000.gz"
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(gz_path, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=65536):
                    f.write(chunk)

        # Parse: each line is "id\treversed.domain" e.g. "42\tcom.google"
        domains = []
        with gzip.open(gz_path, "rt", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                parts = line.split("\t", 1)
                if len(parts) < 2:
                    continue
                surt = parts[1]
                domain = self._reverse_surt(surt)
                if domain:
                    domains.append(domain)

        # Write parsed domains
        with open(cache_file, "w") as f:
            for d in domains:
                f.write(d + "\n")

        log.info("Parsed %d domains from CC vertex partition", len(domains))
        gz_path.unlink(missing_ok=True)
        return cache_file

    @staticmethod
    def _reverse_surt(surt: str) -> str | None:
        """Reverse SURT hostname: 'com.google' -> 'google.com'."""
        segments = surt.strip().split(".")
        if len(segments) < 2:
            return None
        return ".".join(reversed(segments))

    def load(self) -> list[str]:
        """Load domains from cache, downloading if needed."""
        if self._domains is not None:
            return self._domains
        cache_file = self.cache_dir / "cc_domains.txt"
        if not cache_file.exists():
            self.download()
        self._domains = [
            line.strip() for line in cache_file.read_text().splitlines() if line.strip()
        ]
        return self._domains

    def sample(self, n: int = 500) -> list[str]:
        """Return a random sample of n domains."""
        domains = self.load()
        n = min(n, len(domains))
        return random.sample(domains, n)

    @staticmethod
    def domains_to_urls(domains: list[str]) -> list[str]:
        return [f"https://{d}/" for d in domains]
