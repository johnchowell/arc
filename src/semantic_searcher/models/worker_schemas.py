"""Pydantic schemas for distributed worker API."""

from pydantic import BaseModel, Field


# --- Worker Registration (admin) ---

class WorkerRegisterRequest(BaseModel):
    name: str
    endpoint_url: str
    capabilities: dict | None = None


class WorkerRegisterResponse(BaseModel):
    worker_id: str
    api_key: str  # returned once, never stored in plaintext
    message: str


class WorkerInfo(BaseModel):
    worker_id: str
    name: str
    endpoint_url: str
    status: str
    last_heartbeat: str | None = None
    shard_ids: list[str] | None = None
    reputation_score: float
    capabilities: dict | None = None


class WorkerListResponse(BaseModel):
    workers: list[WorkerInfo]


# --- Heartbeat ---

class HeartbeatRequest(BaseModel):
    shard_stats: dict | None = None  # {"shard-001": {"point_count": 5000}, ...}
    system_stats: dict | None = None  # {"cpu_pct": 45, "mem_pct": 30, "gpu_util": 80}


class HeartbeatResponse(BaseModel):
    status: str
    shard_assignments: list[str] | None = None  # current shard IDs for this worker


# --- Crawl Jobs ---

class CrawlJob(BaseModel):
    job_id: int
    url: str
    url_hash: str
    depth: int
    target_shard_id: str


class CrawlJobsResponse(BaseModel):
    jobs: list[CrawlJob]
    lease_duration_seconds: int = 300


class CrawlCompleteRequest(BaseModel):
    job_id: int
    status: str = Field(..., pattern="^(completed|failed|skipped)$")
    metadata: dict | None = None  # page metadata if completed


# --- Search (hub → worker) ---

class WorkerSearchRequest(BaseModel):
    query_vector: list[float]       # 768-dim MPNet
    query_text: str                 # for BM25
    clip_vector: list[float] | None = None  # 512-dim CLIP for image search
    limit: int = 200
    mode: str = "hybrid"
    filters: dict | None = None


class WorkerTextResult(BaseModel):
    page_id: int
    score: float
    chunk_text: str


class WorkerImageResult(BaseModel):
    image_url: str
    alt_text: str
    score: float
    page_ids: list[int]
    nsfw_score: float = 0.0
    icon_score: float = 0.0


class WorkerSearchResponse(BaseModel):
    worker_id: str
    shard_ids: list[str]
    text_results: list[WorkerTextResult]
    image_results: list[WorkerImageResult] = []
    search_time_ms: float


# --- Vector Replication (hub → worker, or primary → replica) ---

class VectorPoint(BaseModel):
    id: int
    dense_vector: list[float]
    bm25_text: str  # text for BM25 sparse encoding
    payload: dict


class VectorBatchRequest(BaseModel):
    shard_id: str
    collection: str = "text_chunks_v2"
    points: list[VectorPoint]


class VectorBatchResponse(BaseModel):
    accepted: int
    rejected: int = 0
    message: str = "ok"


# --- Shard Management (admin) ---

class ShardInfo(BaseModel):
    shard_id: str
    page_id_start: int
    page_id_end: int
    primary_worker_id: str | None = None
    replica_worker_ids: list[str] | None = None
    status: str
    point_count: int


class ShardListResponse(BaseModel):
    shards: list[ShardInfo]
