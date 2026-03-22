"""Headless browser renderer for JavaScript-heavy pages (SPAs).

Manages a persistent Playwright browser instance with a pool of browser contexts
for concurrent rendering. Pages are rendered with a timeout, and the final DOM
HTML is returned for content extraction.
"""
import asyncio
import logging

from playwright.async_api import async_playwright, Browser, BrowserContext

from semantic_searcher.utils.robots import USER_AGENT

log = logging.getLogger(__name__)

# Minimum text length (chars) to consider a page "content-rich" without rendering
MIN_TEXT_LENGTH = 200


class RendererService:
    def __init__(self, max_concurrent: int = 3):
        self._pw = None
        self._browser: Browser | None = None
        self._max_concurrent = max_concurrent
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._context_pool: asyncio.Queue[BrowserContext] = asyncio.Queue()
        self._contexts: list[BrowserContext] = []
        self._started = False
        self._renders = 0

    @property
    def is_started(self) -> bool:
        return self._started

    @property
    def stats(self) -> dict:
        return {"started": self._started, "renders": self._renders, "concurrency": self._max_concurrent}

    async def start(self):
        if self._started:
            return
        self._pw = await async_playwright().start()
        self._browser = await self._pw.chromium.launch(
            headless=True,
            args=[
                "--disable-gpu",
                "--no-sandbox",
                "--disable-dev-shm-usage",
                "--disable-extensions",
            ],
        )
        # Create a pool of browser contexts for concurrent rendering
        for _ in range(self._max_concurrent):
            ctx = await self._browser.new_context(
                user_agent=USER_AGENT,
                java_script_enabled=True,
                ignore_https_errors=True,
                viewport={"width": 1280, "height": 720},
            )
            self._contexts.append(ctx)
            self._context_pool.put_nowait(ctx)
        self._started = True
        log.info("Renderer started (headless Chromium, %d contexts)", self._max_concurrent)

    async def stop(self):
        for ctx in self._contexts:
            try:
                await ctx.close()
            except Exception:
                pass
        self._contexts.clear()
        # Drain pool
        while not self._context_pool.empty():
            try:
                self._context_pool.get_nowait()
            except asyncio.QueueEmpty:
                break
        if self._browser:
            await self._browser.close()
            self._browser = None
        if self._pw:
            await self._pw.stop()
            self._pw = None
        self._started = False
        log.info("Renderer stopped")

    async def render(self, url: str, timeout_ms: int = 15000) -> str | None:
        """Render a URL in headless Chromium and return the final DOM HTML.

        Uses a pool of browser contexts for concurrent rendering.
        Returns None if rendering fails or times out.
        """
        if not self._started:
            await self.start()

        async with self._semaphore:
            ctx = await self._context_pool.get()
            page = None
            try:
                page = await ctx.new_page()

                # Block heavy resources to speed up rendering
                await page.route("**/*.{png,jpg,jpeg,gif,webp,svg,ico,woff,woff2,ttf,eot}",
                                 lambda route: route.abort())

                await page.goto(url, wait_until="networkidle", timeout=timeout_ms)

                # Extra wait for late JS renders
                await page.wait_for_timeout(1000)

                html = await page.content()
                self._renders += 1
                log.info("Rendered SPA: %s (%d chars)", url, len(html))
                return html

            except Exception as e:
                log.warning("Render failed for %s: %s", url, e)
                return None
            finally:
                if page:
                    try:
                        await page.close()
                    except Exception:
                        pass
                self._context_pool.put_nowait(ctx)


def needs_rendering(raw_html: str) -> bool:
    """Heuristic: does this page likely need JS rendering?

    Checks if the raw HTML has very little visible text content but contains
    JavaScript framework markers.
    """
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw_html, "lxml")

    # Remove script/style tags before measuring text
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()

    text = soup.get_text(separator=" ", strip=True)

    # If there's already substantial text, no need to render
    if len(text) >= MIN_TEXT_LENGTH:
        return False

    # Check for SPA framework markers in the raw HTML
    spa_markers = [
        'id="root"', 'id="app"', 'id="__next"', 'id="__nuxt"',
        "ng-app", "ng-version", "data-reactroot",
        "_next/static", "__NEXT_DATA__", "__NUXT__",
        "window.__INITIAL_STATE__", "window.__APP_STATE__",
    ]
    html_lower = raw_html.lower()
    for marker in spa_markers:
        if marker.lower() in html_lower:
            return True

    # Very little text + lots of script tags = likely SPA
    script_count = raw_html.lower().count("<script")
    if len(text) < 50 and script_count > 3:
        return True

    return False
