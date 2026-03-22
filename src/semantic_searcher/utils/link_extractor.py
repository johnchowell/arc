import re
from urllib.parse import urljoin, urlparse

from bs4 import BeautifulSoup

_SKIP_SCHEMES = {"javascript", "mailto", "tel", "data"}

_CDN_NOISE = {
    "googleapis.com",
    "google-analytics.com",
    "googletagmanager.com",
    "analytics.google.com",
    "w3.org",
    "schema.org",
    "fonts.googleapis.com",
    "fonts.gstatic.com",
    "cdnjs.cloudflare.com",
    "cdn.jsdelivr.net",
    "unpkg.com",
    "facebook.net",
    "fbcdn.net",
    "doubleclick.net",
    "googlesyndication.com",
    "google.com/recaptcha",
}

_STATIC_EXTENSIONS = {
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".ico", ".webp", ".avif",
    ".css", ".woff", ".woff2", ".ttf", ".eot", ".otf",
    ".map", ".ts", ".mp3", ".mp4", ".webm", ".ogg",
}

_JS_URL_RE = re.compile(r"""(['"`])(https?://[^\s'"`<>\\]+)\1""")


def extract_html_links(html: str, base_url: str) -> set[str]:
    """Extract URLs from HTML elements: a[href], link[href], iframe[src], area[href], form[action]."""
    soup = BeautifulSoup(html, "html.parser")
    urls: set[str] = set()

    for tag, attr in [
        ("a", "href"),
        ("link", "href"),
        ("iframe", "src"),
        ("area", "href"),
        ("form", "action"),
    ]:
        for el in soup.find_all(tag, **{attr: True}):
            raw = el.get(attr, "").strip()
            if not raw or raw.startswith("#"):
                continue
            parsed = urlparse(raw)
            if parsed.scheme and parsed.scheme.lower() in _SKIP_SCHEMES:
                continue
            resolved = urljoin(base_url, raw)
            final = urlparse(resolved)
            if final.scheme in ("http", "https"):
                # Strip fragment
                clean = final._replace(fragment="").geturl()
                urls.add(clean)

    return urls


def extract_js_urls(html: str) -> set[str]:
    """Extract https?:// URLs from inline <script> blocks."""
    soup = BeautifulSoup(html, "html.parser")
    urls: set[str] = set()

    for script in soup.find_all("script"):
        # Skip external scripts (no inline text)
        if script.get("src"):
            continue
        text = script.string
        if not text:
            continue
        for match in _JS_URL_RE.finditer(text):
            url = match.group(2)
            if _is_cdn_noise(url):
                continue
            if _is_static_asset(url):
                continue
            if _is_minified_bundle(url):
                continue
            parsed = urlparse(url)
            if parsed.scheme in ("http", "https"):
                clean = parsed._replace(fragment="").geturl()
                urls.add(clean)

    return urls


def extract_all_links(html: str, base_url: str) -> set[str]:
    """Union of HTML link extraction and JS URL extraction."""
    return extract_html_links(html, base_url) | extract_js_urls(html)


def _is_cdn_noise(url: str) -> bool:
    hostname = urlparse(url).hostname or ""
    return any(cdn in hostname for cdn in _CDN_NOISE)


def _is_static_asset(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in _STATIC_EXTENSIONS)


def _is_minified_bundle(url: str) -> bool:
    path = urlparse(url).path.lower()
    return ".min." in path or ".bundle." in path
