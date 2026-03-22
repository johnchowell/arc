from pydantic_settings import BaseSettings
from pathlib import Path


class Settings(BaseSettings):
    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}

    # MySQL
    mysql_host: str = "127.0.0.1"
    mysql_port: int = 3307
    mysql_user: str = "root"
    mysql_password: str = ""
    mysql_database: str = "semantic_searcher"

    # CLIP
    clip_model: str = "ViT-B-32"
    clip_pretrained: str = "laion2b_s34b_b79k"

    # Storage
    raid_path: str = "/media/raid/webindex"
    webindex_dir: str = "/media/raid/webindex"

    # Crawler
    crawler_workers: int = 50
    crawler_rate_limit: float = 0.25
    crawler_max_depth: int = 3
    crawler_max_pages: int = 10000

    # Indexer
    index_batch_size: int = 32
    index_batch_timeout: float = 2.0

    # Renderer
    renderer_concurrency: int = 10

    # Auto-crawl
    autocrawl_sample_size: int = 500
    autocrawl_max_depth: int = 2
    autocrawl_max_pages: int = 50000

    # CT log watcher
    certstream_url: str = "wss://certstream.calidog.io/"
    ct_flush_interval: int = 30
    ct_flush_batch_size: int = 500

    # Link harvester
    harvester_batch_size: int = 50
    harvester_batch_interval: float = 1.0
    harvester_idle_interval: float = 60.0

    # Subdomain enumerator
    enum_batch_size: int = 20
    enum_idle_interval: float = 30.0
    enum_dns_timeout: float = 2.0
    enum_dns_concurrency: int = 20

    # Search concurrency
    search_max_workers: int = 4  # max parallel search operations
    search_queue_max: int = 50   # max queued searches before rejecting

    # Qdrant
    qdrant_host: str = "127.0.0.1"
    qdrant_port: int = 6333

    # Server
    fastapi_port: int = 8900
    crawl_only: bool = False  # skip search index loading, serve only crawling

    # Distributed mode
    distributed_mode: bool = True
    hub_role: str = "hub"                 # "standalone", "hub", "worker"
    hub_url: str = ""                     # workers: hub API endpoint
    worker_api_key: str = ""              # workers: API key for hub auth
    worker_id: str = ""                   # workers: auto-generated UUID
    worker_listen_port: int = 8901
    shard_size_target: int = 500_000      # target vectors per shard
    shard_replication_factor: int = 2
    search_fanout_timeout_ms: int = 2000
    worker_heartbeat_interval: int = 30
    hub_api_secret: str = ""              # hub: master secret for generating keys

    @property
    def database_url(self) -> str:
        return (
            f"mysql+aiomysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )

    @property
    def sync_database_url(self) -> str:
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )

    @property
    def html_cache_dir(self) -> Path:
        return Path(self.webindex_dir) / "html"

    @property
    def tokenizer_dir(self) -> Path:
        return Path(self.webindex_dir) / "tokenizer"


settings = Settings()
