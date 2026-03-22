from pydantic import BaseModel


# --- Search ---
class SearchResultItem(BaseModel):
    page_id: int
    url: str
    title: str
    snippet: str
    score: float
    text_score: float
    image_score: float
    indexed_at: str | None = None
    domain: str | None = None
    tld_group: str | None = None
    content_category: str | None = None


class SearchResponse(BaseModel):
    query: str
    results: list[SearchResultItem]
    total_results: int
    search_time_ms: float
    corrected_query: str | None = None
    original_query: str | None = None


# --- Image Search ---
class ImageSearchResultItem(BaseModel):
    image_url: str
    alt_text: str
    score: float
    page_id: int
    page_url: str
    page_title: str
    found_on: int = 1
    possibly_explicit: bool = False
    domain: str | None = None
    tld_group: str | None = None
    content_category: str | None = None


class ImageSearchResponse(BaseModel):
    query: str
    results: list[ImageSearchResultItem]
    total_results: int
    search_time_ms: float


# --- Filters ---
class FiltersResponse(BaseModel):
    domains: list[str]
    tld_groups: list[str]
    categories: list[str]


# --- Index ---
class IndexRequest(BaseModel):
    url: str


class IndexResponse(BaseModel):
    page_id: int | None = None
    url: str
    status: str
    message: str


# --- Crawl ---
class CrawlStartRequest(BaseModel):
    seed_urls: list[str]
    max_depth: int = 3
    max_pages: int = 10000


class CrawlAutoRequest(BaseModel):
    sample_size: int = 500
    max_depth: int = 2
    max_pages: int = 50000
    force_download: bool = False


class CrawlAutoResponse(BaseModel):
    message: str
    sample_size: int
    seed_urls: list[str]


class CTWatcherStatusResponse(BaseModel):
    is_running: bool
    domains_discovered: int
    domains_queued: int
    connected: bool


class HarvesterStatusResponse(BaseModel):
    is_running: bool
    pages_harvested: int
    links_queued: int


class SubdomainEnumStatusResponse(BaseModel):
    is_running: bool
    domains_enumerated: int
    subdomains_queued: int


class CrawlStatusResponse(BaseModel):
    is_running: bool
    pages_crawled: int
    queue_size: int
    crawling: int
    done: int
    failed: int
    max_pages: int
    max_depth: int
    started_at: str | None = None


# --- Pages ---
class PageResponse(BaseModel):
    id: int
    url: str
    title: str | None = None
    meta_description: str | None = None
    status: str
    indexed_at: str | None = None
    created_at: str | None = None


class PageListResponse(BaseModel):
    pages: list[PageResponse]
    total: int
    offset: int
    limit: int


# --- Health ---
class HealthResponse(BaseModel):
    status: str
    database: str
    clip_model: str
    gpu_available: bool


# --- Stats ---
class StatsResponse(BaseModel):
    pages_indexed: int
    text_embeddings: int
    image_embeddings: int
    pages_per_hour: float = 0.0


# --- Tokenizer ---
class TokenizeRequest(BaseModel):
    text: str


class TokenizeResponse(BaseModel):
    ids: list[int]
    tokens_count: int


class DetokenizeRequest(BaseModel):
    ids: list[int]


class DetokenizeResponse(BaseModel):
    text: str


class TrainTokenizerRequest(BaseModel):
    vocab_size: int = 30000
