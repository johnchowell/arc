"""IP-based rate limiting middleware.

A single user search triggers multiple parallel requests (search stream,
autocomplete, related, filters).  We count by "search actions" — only
heavyweight endpoints (/api/search, /api/images/search) count toward the
burst limit.  Lightweight supporting endpoints are free.

Rules:
- More than 3 search actions in 2 seconds → throttled to 1 per 5 seconds for 1 hour
- Continually hitting the throttle limit → blocked for 24 hours
"""

import time
import logging
from collections import defaultdict
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

log = logging.getLogger(__name__)

BURST_WINDOW = 2.0        # seconds
BURST_MAX = 6             # max search actions in burst window (text+image per query)
THROTTLE_INTERVAL = 5.0   # seconds between search actions when throttled
THROTTLE_DURATION = 300   # 5 minutes
BLOCK_DURATION = 86400    # 24 hours
BLOCK_THRESHOLD = 10      # violations during throttle before 24h ban

# Paths that count toward rate limiting (heavyweight search endpoints)
_RATED_PREFIXES = (
    "/api/search",
    "/api/images/search",
)

# Paths that are always free (lightweight supporting requests)
_FREE_PREFIXES = (
    "/api/autocomplete",
    "/api/suggest",
    "/api/related",
    "/api/filters",
    "/api/search/queue",
    "/api/health",
    "/api/click",
    "/api/bounce",
    "/api/image/sources",
    "/api/feedback",
)


class _IPState:
    __slots__ = ("timestamps", "throttled_at", "last_allowed", "blocked_until", "violations")

    def __init__(self):
        self.timestamps: list[float] = []
        self.throttled_at: float = 0.0
        self.last_allowed: float = 0.0
        self.blocked_until: float = 0.0
        self.violations: int = 0


def _is_rated(path: str) -> bool:
    """Return True if this path should count toward the rate limit."""
    for prefix in _FREE_PREFIXES:
        if path.startswith(prefix):
            return False
    for prefix in _RATED_PREFIXES:
        if path.startswith(prefix):
            return True
    return False


class RateLimitMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, exclude_paths: set[str] | None = None):
        super().__init__(app)
        self._ips: dict[str, _IPState] = defaultdict(_IPState)
        self._exclude = exclude_paths or set()
        self._last_cleanup = time.monotonic()

    def _get_ip(self, request: Request) -> str:
        forwarded = request.headers.get("x-forwarded-for")
        if forwarded:
            return forwarded.split(",")[0].strip()
        real_ip = request.headers.get("x-real-ip")
        if real_ip:
            return real_ip.strip()
        return request.client.host if request.client else "unknown"

    def _cleanup(self, now: float):
        """Purge expired entries every 5 minutes to prevent memory growth."""
        if now - self._last_cleanup < 300:
            return
        self._last_cleanup = now
        expired = []
        for ip, state in self._ips.items():
            if state.blocked_until and now < state.blocked_until:
                continue
            if state.throttled_at and now - state.throttled_at < THROTTLE_DURATION:
                continue
            if not state.timestamps or now - state.timestamps[-1] > BURST_WINDOW * 10:
                expired.append(ip)
        for ip in expired:
            del self._ips[ip]

    async def dispatch(self, request: Request, call_next):
        path = request.url.path

        # Static files and excluded paths always pass through
        if path in self._exclude or path.startswith("/static"):
            return await call_next(request)

        # Only rate-limit heavyweight search endpoints
        if not _is_rated(path):
            return await call_next(request)

        now = time.monotonic()
        self._cleanup(now)

        ip = self._get_ip(request)
        state = self._ips[ip]

        # Blocked (24hr ban)
        if state.blocked_until and now < state.blocked_until:
            remaining = int(state.blocked_until - now)
            return JSONResponse(
                status_code=429,
                content={"error": "Too many requests. Try again later."},
                headers={"Retry-After": str(remaining)},
            )

        # Throttled (1 search action per 5s for 1 hour)
        if state.throttled_at and now - state.throttled_at < THROTTLE_DURATION:
            elapsed_since_last = now - state.last_allowed
            if elapsed_since_last < THROTTLE_INTERVAL:
                # Still hitting the limit while throttled → escalate to 24hr block
                state.violations += 1
                if state.violations >= BLOCK_THRESHOLD:
                    state.blocked_until = now + BLOCK_DURATION
                    state.throttled_at = 0.0
                    log.warning("IP %s blocked for 24h (repeated rate limit violations)", ip)
                    return JSONResponse(
                        status_code=429,
                        content={"error": "Too many requests. Try again later."},
                        headers={"Retry-After": str(BLOCK_DURATION)},
                    )
                retry_after = int(THROTTLE_INTERVAL - elapsed_since_last) + 1
                return JSONResponse(
                    status_code=429,
                    content={"error": "Rate limited. Slow down."},
                    headers={"Retry-After": str(retry_after)},
                )
            # Allowed under throttle — reset violation counter on good behavior
            state.violations = 0
            state.last_allowed = now
            return await call_next(request)

        # Normal burst detection
        cutoff = now - BURST_WINDOW
        state.timestamps = [t for t in state.timestamps if t > cutoff]
        state.timestamps.append(now)

        if len(state.timestamps) > BURST_MAX:
            state.throttled_at = now
            state.last_allowed = now
            state.violations = 0
            log.info("IP %s throttled (>%d search actions in %.0fs)", ip, BURST_MAX, BURST_WINDOW)
            return JSONResponse(
                status_code=429,
                content={"error": "Rate limited. Slow down."},
                headers={"Retry-After": str(int(THROTTLE_INTERVAL))},
            )

        state.last_allowed = now
        return await call_next(request)
