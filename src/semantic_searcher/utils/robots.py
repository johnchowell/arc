import logging
import time
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

log = logging.getLogger(__name__)

ROBOTS_AGENT = "Googlebot/2.1"
USER_AGENT = "Mozilla/5.0 (X11; Linux x86_64; rv:137.0) Gecko/20100101 Firefox/137.0"


class RobotsCache:
    def __init__(self, ttl: int = 3600):
        self._cache: dict[str, tuple[RobotFileParser, float]] = {}
        self._ttl = ttl

    async def can_fetch(self, url: str, client: httpx.AsyncClient) -> bool:
        return True
