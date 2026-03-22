import datetime
from sqlalchemy import (
    BigInteger, Boolean, DateTime, Enum, Float, ForeignKey, Integer,
    LargeBinary, SmallInteger, String, Text, func,
)
from sqlalchemy.dialects.mysql import MEDIUMTEXT, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    pass


class Page(Base):
    __tablename__ = "pages"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    url: Mapped[str] = mapped_column(String(2048), nullable=False)
    url_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    title: Mapped[str | None] = mapped_column(String(1024))
    text_content: Mapped[str | None] = mapped_column(MEDIUMTEXT)
    meta_description: Mapped[str | None] = mapped_column(String(2048))
    language: Mapped[str | None] = mapped_column(String(10))
    domain: Mapped[str | None] = mapped_column(String(253), index=True)
    tld_group: Mapped[str | None] = mapped_column(String(10), index=True)
    content_category: Mapped[str | None] = mapped_column(String(20), index=True)
    nsfw_flag: Mapped[bool | None] = mapped_column(Boolean, nullable=True, index=True)
    status: Mapped[str] = mapped_column(
        Enum("pending", "indexing", "indexed", "failed", name="page_status"),
        default="pending",
    )
    indexed_at: Mapped[datetime.datetime | None] = mapped_column(DateTime)
    links_harvested_at: Mapped[datetime.datetime | None] = mapped_column(DateTime)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now()
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now()
    )
    last_modified: Mapped[datetime.datetime | None] = mapped_column(DateTime)

    text_embeddings: Mapped[list["TextEmbedding"]] = relationship(
        back_populates="page", cascade="all, delete-orphan"
    )
    image_sources: Mapped[list["ImagePageSource"]] = relationship(
        cascade="all, delete-orphan"
    )


class TextEmbedding(Base):
    __tablename__ = "text_embeddings"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    page_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("pages.id", ondelete="CASCADE"), nullable=False, index=True
    )
    chunk_index: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    chunk_text: Mapped[str] = mapped_column(Text, nullable=False)
    embedding: Mapped[bytes] = mapped_column(LargeBinary(2048), nullable=False)

    page: Mapped["Page"] = relationship(back_populates="text_embeddings")


class ImageEmbedding(Base):
    __tablename__ = "image_embeddings"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    image_url: Mapped[str] = mapped_column(String(2048), nullable=False)
    image_url_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    alt_text: Mapped[str | None] = mapped_column(String(1024))
    embedding: Mapped[bytes] = mapped_column(LargeBinary(2048), nullable=False)
    nsfw_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    icon_score: Mapped[float | None] = mapped_column(Float, nullable=True)

    page_sources: Mapped[list["ImagePageSource"]] = relationship(
        back_populates="image_embedding", cascade="all, delete-orphan"
    )


class ImagePageSource(Base):
    __tablename__ = "image_page_sources"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    image_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("image_embeddings.id", ondelete="CASCADE"), nullable=False, index=True
    )
    page_id: Mapped[int] = mapped_column(
        BigInteger, ForeignKey("pages.id", ondelete="CASCADE"), nullable=False, index=True
    )

    image_embedding: Mapped["ImageEmbedding"] = relationship(back_populates="page_sources")
    page: Mapped["Page"] = relationship(overlaps="image_sources")


class CrawlQueue(Base):
    __tablename__ = "crawl_queue"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    url: Mapped[str] = mapped_column(String(2048), nullable=False)
    url_hash: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    depth: Mapped[int] = mapped_column(Integer, default=0)
    priority: Mapped[float] = mapped_column(Float, default=0.0)
    source_page_id: Mapped[int | None] = mapped_column(
        BigInteger, ForeignKey("pages.id", ondelete="SET NULL")
    )
    status: Mapped[str] = mapped_column(
        Enum("queued", "crawling", "done", "failed", "skipped", name="crawl_status"),
        default="queued",
    )
    discovered_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now()
    )
    crawled_at: Mapped[datetime.datetime | None] = mapped_column(DateTime)

    source_page: Mapped["Page | None"] = relationship()


class CrawlSeen(Base):
    """Lightweight dedup table — stores url_hash of all URLs ever queued.

    This allows crawl_queue to stay small (only active rows) while
    preventing re-queuing of URLs that were already processed.
    """
    __tablename__ = "crawl_seen"

    url_hash: Mapped[str] = mapped_column(String(64), primary_key=True)


class SearchLog(Base):
    __tablename__ = "search_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    query: Mapped[str] = mapped_column(String(2048), nullable=False)
    searched_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now()
    )


class ClickLog(Base):
    __tablename__ = "click_logs"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    query: Mapped[str] = mapped_column(String(2048), nullable=False)
    url: Mapped[str] = mapped_column(String(2048), nullable=False)
    position: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    dwell_time_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    bounced: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    clicked_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now()
    )


class Feedback(Base):
    __tablename__ = "feedback"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    rating: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    category: Mapped[str | None] = mapped_column(String(50), nullable=True)
    message: Mapped[str | None] = mapped_column(Text, nullable=True)
    page: Mapped[str | None] = mapped_column(String(200), nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(512), nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, server_default=func.now()
    )


class CrawlState(Base):
    __tablename__ = "crawl_state"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    is_running: Mapped[bool] = mapped_column(Boolean, default=False)
    seed_urls: Mapped[dict | None] = mapped_column(JSON)
    max_depth: Mapped[int] = mapped_column(Integer, default=3)
    max_pages: Mapped[int] = mapped_column(Integer, default=10000)
    pages_crawled: Mapped[int] = mapped_column(Integer, default=0)
    started_at: Mapped[datetime.datetime | None] = mapped_column(DateTime)
    stopped_at: Mapped[datetime.datetime | None] = mapped_column(DateTime)


# --- Distributed worker/shard management ---

class Worker(Base):
    __tablename__ = "workers"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    worker_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    api_key_hash: Mapped[str] = mapped_column(String(128), nullable=False)
    endpoint_url: Mapped[str] = mapped_column(String(512), nullable=False)
    ip_address: Mapped[str | None] = mapped_column(String(45))  # IPv4 or IPv6
    status: Mapped[str] = mapped_column(
        Enum("active", "offline", "draining", "banned", name="worker_status"),
        default="active",
    )
    last_heartbeat: Mapped[datetime.datetime | None] = mapped_column(DateTime)
    ping_latency_ms: Mapped[float | None] = mapped_column(Float)  # last measured ping
    shard_ids: Mapped[dict | None] = mapped_column(JSON)
    capabilities: Mapped[dict | None] = mapped_column(JSON)
    reputation_score: Mapped[float] = mapped_column(Float, default=0.5)
    jobs_completed: Mapped[int] = mapped_column(Integer, default=0)
    jobs_failed: Mapped[int] = mapped_column(Integer, default=0)
    pages_indexed: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())


class Shard(Base):
    __tablename__ = "shards"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    shard_id: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    page_id_start: Mapped[int] = mapped_column(BigInteger, nullable=False)
    page_id_end: Mapped[int] = mapped_column(BigInteger, nullable=False)
    primary_worker_id: Mapped[str | None] = mapped_column(String(64))
    replica_worker_ids: Mapped[dict | None] = mapped_column(JSON)  # list of worker_id strings
    status: Mapped[str] = mapped_column(
        Enum("active", "rebalancing", "orphaned", name="shard_status"),
        default="active",
    )
    point_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
    updated_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())


class CrawlJobAssignment(Base):
    __tablename__ = "crawl_job_assignments"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    crawl_queue_id: Mapped[int] = mapped_column(BigInteger, nullable=False, index=True)
    worker_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    assigned_shard_id: Mapped[str] = mapped_column(String(64), nullable=False)
    status: Mapped[str] = mapped_column(
        Enum("assigned", "in_progress", "completed", "failed", name="job_assignment_status"),
        default="assigned",
    )
    assigned_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
    completed_at: Mapped[datetime.datetime | None] = mapped_column(DateTime)


class WorkerActivityLog(Base):
    """Rolling log of worker activity — heartbeats, ping latency, job stats."""
    __tablename__ = "worker_activity_log"

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    worker_id: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    ip_address: Mapped[str | None] = mapped_column(String(45))
    status: Mapped[str] = mapped_column(String(20), nullable=False)
    ping_latency_ms: Mapped[float | None] = mapped_column(Float)
    jobs_completed: Mapped[int] = mapped_column(Integer, default=0)
    jobs_failed: Mapped[int] = mapped_column(Integer, default=0)
    pages_indexed: Mapped[int] = mapped_column(Integer, default=0)
    system_stats: Mapped[dict | None] = mapped_column(JSON)  # cpu, mem, gpu, disk
    logged_at: Mapped[datetime.datetime] = mapped_column(DateTime, server_default=func.now())
