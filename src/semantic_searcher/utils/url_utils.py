import hashlib
from urllib.parse import urlparse, urlunparse, urlencode, parse_qs

import tldextract

TRACKING_PARAMS = {
    "utm_source", "utm_medium", "utm_campaign", "utm_term", "utm_content",
    "fbclid", "gclid", "ref", "source",
}


def normalize_url(url: str) -> str:
    parsed = urlparse(url)
    scheme = parsed.scheme.lower()
    host = parsed.hostname.lower() if parsed.hostname else ""
    port = parsed.port
    path = parsed.path.rstrip("/") or "/"
    # Sort query params and strip tracking
    params = parse_qs(parsed.query, keep_blank_values=True)
    filtered = {k: v for k, v in sorted(params.items()) if k not in TRACKING_PARAMS}
    query = urlencode(filtered, doseq=True)
    # Reconstruct without fragment
    netloc = host
    if port and not (scheme == "http" and port == 80) and not (scheme == "https" and port == 443):
        netloc = f"{host}:{port}"
    return urlunparse((scheme, netloc, path, "", query, ""))


def url_hash(url: str) -> str:
    return hashlib.sha256(normalize_url(url).encode()).hexdigest()


def is_same_domain(url1: str, url2: str) -> bool:
    return urlparse(url1).hostname == urlparse(url2).hostname


def extract_domain(url: str) -> str:
    """Extract registered domain (e.g. 'en.wikipedia.org' -> 'wikipedia.org')."""
    ext = tldextract.extract(url)
    if ext.domain and ext.suffix:
        return f"{ext.domain}.{ext.suffix}"
    return ext.domain or ""


# TLD suffixes that map to specific groups (handles ccTLDs like .ac.uk, .gov.au)
_EDU_SUFFIXES = {"edu", "ac.uk", "edu.au", "ac.jp", "edu.cn", "edu.br", "ac.in", "edu.mx"}
_GOV_SUFFIXES = {"gov", "gov.uk", "gov.au", "go.jp", "gov.cn", "gov.br", "gov.in"}


def extract_tld_group(url: str) -> str:
    """Classify URL into tld group: edu, gov, org, com, net, or other."""
    ext = tldextract.extract(url)
    suffix = ext.suffix.lower() if ext.suffix else ""

    if suffix in _EDU_SUFFIXES or suffix.startswith("edu.") or suffix.startswith("ac."):
        return "edu"
    if suffix in _GOV_SUFFIXES or suffix.startswith("gov.") or suffix.startswith("go."):
        return "gov"
    if suffix == "org" or suffix.endswith(".org"):
        return "org"
    if suffix == "com" or suffix.endswith(".com"):
        return "com"
    if suffix == "net" or suffix.endswith(".net"):
        return "net"
    return "other"
