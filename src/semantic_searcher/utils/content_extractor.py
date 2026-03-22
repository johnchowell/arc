import logging
import re
from dataclasses import dataclass, field
from urllib.parse import urljoin

from bs4 import BeautifulSoup
from readability import Document

log = logging.getLogger(__name__)

_JUNK_IMAGE_PATTERNS = re.compile(
    r"CentralAutoLogin|/Special:|/count[./]|counter\.|"
    r"1x1|pixel\.|tracking|beacon|spacer|blank\.|"
    r"private-user-images\.githubusercontent\.com|"
    r"\.gif\?jwt=|"
    # Wikipedia/Wikimedia UI icons and tiny thumbnails
    r"OOjs_UI_icon|Blue_check\.svg|WMF_open_door|"
    r"Breezeicons|Ambox_|Question_book|Edit-clear|"
    r"Wiki_letter|Wiktionary-logo|Wikiquote-logo|"
    r"Wikidata-logo|Wikisource-logo|Wikivoyage-Logo|"
    r"Commons-logo|Wikinews-logo|Wikibooks-logo|"
    r"Wikiversity-logo|Wikispecies-logo|"
    r"Contactus-wmcolors|Wikipedia_mass_messenger|"
    r"Nospam_at\.svg|At_sign\.svg|"
    # GitHub generic repo/org images
    r"repository-images\.githubusercontent\.com|"
    r"avatars\.githubusercontent\.com",
    re.IGNORECASE,
)

_JUNK_EXTENSIONS = {".svg", ".ico"}

# Tiny Wikimedia thumbnails: /20px-, /40px-, /60px- etc.
_TINY_THUMB_RE = re.compile(r"/(\d+)px-")
_MIN_THUMB_PX = 100


def _is_junk_image(url: str) -> bool:
    if _JUNK_IMAGE_PATTERNS.search(url):
        return True
    # Check extension (before query string)
    path = url.split("?")[0].split("#")[0].lower()
    for ext in _JUNK_EXTENSIONS:
        if path.endswith(ext):
            return True
    # Filter tiny Wikimedia thumbnails (< 100px wide)
    m = _TINY_THUMB_RE.search(url)
    if m and int(m.group(1)) < _MIN_THUMB_PX:
        return True
    return False


@dataclass
class ExtractedContent:
    title: str = ""
    text: str = ""
    meta_description: str = ""
    image_urls: list[str] = field(default_factory=list)
    image_alts: dict[str, str] = field(default_factory=dict)  # url → alt
    published_date: str | None = None  # ISO date string from page metadata


def _extract_page_date(soup: BeautifulSoup) -> str | None:
    """Extract published/modified date from HTML meta tags, JSON-LD, or <time> elements."""
    # 1. Open Graph / article meta tags
    for attr_name in ("article:published_time", "article:modified_time",
                      "og:updated_time", "date", "pubdate", "publish_date",
                      "DC.date", "DC.date.issued"):
        tag = soup.find("meta", attrs={"property": attr_name}) or \
              soup.find("meta", attrs={"name": attr_name})
        if tag and tag.get("content"):
            return tag["content"][:25]

    # 2. JSON-LD datePublished / dateModified
    for script in soup.find_all("script", type="application/ld+json"):
        try:
            import json
            data = json.loads(script.string or "")
            if isinstance(data, list):
                data = data[0] if data else {}
            for key in ("dateModified", "datePublished", "uploadDate"):
                if key in data and data[key]:
                    return str(data[key])[:25]
        except Exception:
            pass

    # 3. <time> element with datetime attribute
    time_el = soup.find("time", attrs={"datetime": True})
    if time_el and time_el["datetime"]:
        return time_el["datetime"][:25]

    return None


def extract_content(html: str, base_url: str) -> ExtractedContent:
    result = ExtractedContent()

    # Get meta description and page date from raw HTML
    soup_raw = BeautifulSoup(html, "lxml")
    meta = soup_raw.find("meta", attrs={"name": "description"})
    if meta and meta.get("content"):
        result.meta_description = meta["content"][:2048]
    result.published_date = _extract_page_date(soup_raw)

    # Use readability for main content extraction
    doc = Document(html)
    result.title = doc.title() or ""

    article_html = doc.summary()
    soup = BeautifulSoup(article_html, "lxml")

    # Extract text
    result.text = soup.get_text(separator="\n", strip=True)

    # Extract images from article
    for img in soup.find_all("img"):
        src = img.get("src")
        if not src:
            continue
        abs_url = urljoin(base_url, src)
        if abs_url.startswith(("http://", "https://")) and not _is_junk_image(abs_url):
            result.image_urls.append(abs_url)
            alt = img.get("alt", "")
            if alt:
                result.image_alts[abs_url] = alt

    return result


def chunk_text(text: str, max_words: int = 55, overlap: int = 15) -> list[str]:
    words = text.split()
    if not words:
        return []
    step = max_words - overlap
    chunks = []
    for i in range(0, len(words), step):
        chunk = " ".join(words[i : i + max_words])
        if len(chunk.strip()) > 10:
            chunks.append(chunk)
    return chunks
