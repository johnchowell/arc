import difflib
import logging
import math
import re
import time
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from dataclasses import dataclass
from typing import Generator
from urllib.parse import urlparse

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Document,
    FieldCondition,
    Filter,
    Fusion,
    FusionQuery,
    MatchAny,
    MatchText,
    MatchValue,
    NamedVector,
    Prefetch,
    Range,
    SearchParams,
    SearchRequest,
)
from sqlalchemy import func as sa_func, select
from sqlalchemy.ext.asyncio import AsyncSession

from semantic_searcher.models.db import Page, TextEmbedding, ImageEmbedding, ImagePageSource
from semantic_searcher.services.clip_service import CLIPService
from semantic_searcher.services.qdrant_collections import (
    TEXT_CHUNKS_V2_COLLECTION,
    IMAGES_COLLECTION,
    STALE_THRESHOLD_DAYS,
    build_search_filter,
)
from semantic_searcher.utils.nsfw_scorer import NSFW_IMAGE_THRESHOLD, ICON_IMAGE_THRESHOLD

log = logging.getLogger(__name__)

_WORD_RE = re.compile(r"\b[a-zA-Z0-9]+(?:'[a-zA-Z]+)?\b")
_JUNK_RE = re.compile(
    r"^[\d\s\W]{20,}$"       # mostly numbers/symbols
    r"|^\s*[\d\s,.:;|/\\]+$"  # just delimiters and digits
)


def _tokenize(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text)}


_STOPWORDS: frozenset[str] = frozenset({
    "a", "an", "the", "and", "or", "but", "not", "no", "nor",
    "is", "are", "was", "were", "be", "been", "being", "am",
    "do", "does", "did", "doing", "done",
    "have", "has", "had", "having",
    "will", "would", "shall", "should", "may", "might", "can", "could", "must",
    "i", "me", "my", "we", "us", "our", "you", "your", "he", "him", "his",
    "she", "her", "it", "its", "they", "them", "their",
    "this", "that", "these", "those",
    "what", "which", "who", "whom", "whose", "where", "when", "how", "why",
    "in", "on", "at", "to", "for", "of", "with", "by", "from", "about",
    "into", "through", "during", "before", "after", "between", "under", "over",
    "if", "then", "so", "than", "too", "very", "just", "also",
    "up", "out", "off", "all", "each", "every", "both", "any", "some",
})


def _filter_stopwords(words: set[str]) -> set[str]:
    """Remove stopwords; fall back to full set if all are stopwords."""
    filtered = words - _STOPWORDS
    return filtered if filtered else words


def _keyword_overlap(query_words: set[str], text: str) -> float:
    """Fraction of content query words found in text (0.0–1.0)."""
    if not query_words:
        return 0.0
    content_words = _filter_stopwords(query_words)
    text_words = _tokenize(text)
    return len(content_words & text_words) / len(content_words)


def _is_junk_chunk(chunk: str) -> bool:
    """Return True if a chunk is boilerplate/code/noise."""
    if _JUNK_RE.match(chunk):
        return True
    words = chunk.split()
    if not words:
        return True
    # If more than half the tokens are non-alpha, it's junk
    alpha_count = sum(1 for w in words if any(c.isalpha() for c in w))
    if alpha_count / len(words) < 0.5:
        return True
    return False


@dataclass
class SearchResult:
    page_id: int
    url: str
    title: str
    snippet: str
    score: float
    text_score: float
    image_score: float
    indexed_at: str | None
    domain: str | None = None
    tld_group: str | None = None
    content_category: str | None = None


@dataclass
class ImageSearchResult:
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
    all_page_ids: list | None = None


@dataclass
class _RankedResult:
    """Internal scored result before snippet selection."""
    page_id: int
    url: str
    title: str
    score: float
    text_score: float
    image_score: float
    indexed_at: str | None
    domain: str | None
    tld_group: str | None
    content_category: str | None


_RESULT_CACHE_MAX = 512
_RESULT_CACHE_TTL = 300  # seconds
_WARMUP_QUERIES = 100  # pre-warm cache with top N popular queries


class SearchService:
    _RERANK_CANDIDATES = 15  # re-rank top N candidates from stage 1
    _RERANK_WEIGHT = 0.50     # blend: (1 - w) * stage1_score + w * rerank_score

    def __init__(self, clip: CLIPService, qdrant: QdrantClient | None = None, text_encoder=None):
        self.clip = clip
        self._qdrant = qdrant
        self._text_encoder = text_encoder  # MPNet for text search
        self._text_collection = TEXT_CHUNKS_V2_COLLECTION
        self._cross_encoder = None
        self._spell_checker = None  # SymSpell instance
        # Page metadata cache (loaded from MySQL — small and fast)
        self._pages: dict[int, dict] = {}
        # Domain root lookups for navigational queries
        self._domain_roots: dict[str, int] = {}    # domain -> page_id of root "/"
        self._domain_labels: dict[str, str] = {}   # label (e.g. "wikipedia") -> domain
        # Page-level NSFW flag (set of page_ids with explicit content)
        self._nsfw_pages: set[int] = set()
        # Stale pages (indexed_at older than STALE_THRESHOLD_DAYS)
        self._stale_pages: set[int] = set()
        # Chunk counts per page (for quality scoring)
        self._chunk_counts: dict[int, int] = {}
        # Precomputed freshness and quality signals
        self._page_age_days: dict[int, float] = {}
        self._page_quality: dict[int, float] = {}
        self._page_emb_diversity: dict[int, float] = {}
        self._domain_page_counts: dict[str, int] = defaultdict(int)
        # Result cache
        self._result_cache: OrderedDict[tuple, tuple] = OrderedDict()
        # Tracking IDs for incremental updates
        self._max_page_id: int = 0

    def load_cross_encoder(self):
        """Load the cross-encoder model for re-ranking (called after index load)."""
        try:
            from sentence_transformers import CrossEncoder
            t0 = time.time()
            self._cross_encoder = CrossEncoder(
                "cross-encoder/ms-marco-MiniLM-L-6-v2", device="cpu"
            )
            log.info("Cross-encoder loaded (%.1fs)", time.time() - t0)
        except Exception as e:
            log.warning("Failed to load cross-encoder: %s — re-ranking disabled", e)
            self._cross_encoder = None

    def init_spell_checker(self):
        """Initialize SymSpell spell checker."""
        try:
            import symspellpy
            from importlib.resources import files as _files
            t0 = time.time()
            self._spell_checker = symspellpy.SymSpell(max_dictionary_edit_distance=2, prefix_length=7)
            # Load built-in English frequency dictionaries from symspellpy package
            pkg = _files("symspellpy")
            dict_path = str(pkg / "frequency_dictionary_en_82_765.txt")
            self._spell_checker.load_dictionary(dict_path, term_index=0, count_index=1)
            bigram_path = str(pkg / "frequency_bigramdictionary_en_243_342.txt")
            self._spell_checker.load_bigram_dictionary(bigram_path, term_index=0, count_index=2)
            log.info("SymSpell spell checker loaded (%.1fs)", time.time() - t0)
        except Exception as e:
            log.warning("Failed to load spell checker: %s", e)
            self._spell_checker = None

    def _spell_suggest(self, query: str) -> str | None:
        """Spell correction via SymSpell. Returns corrected query or None."""
        if self._spell_checker is None:
            return None
        try:
            suggestions = self._spell_checker.lookup_compound(query, max_edit_distance=2)
            if suggestions and suggestions[0].term != query.lower():
                return suggestions[0].term
        except Exception as e:
            log.warning("Spell suggest failed: %s", e)
        return None

    def _rerank(self, query: str, results: list[_RankedResult],
                clip_snippets: dict[int, str], keyword_snippets: dict[int, str]) -> list[_RankedResult]:
        """Re-rank top candidates using the cross-encoder."""
        if self._cross_encoder is None or len(results) < 2:
            return results
        n = min(self._RERANK_CANDIDATES, len(results))
        top = results[:n]
        rest = results[n:]

        # Build (query, passage) pairs — use title + best available snippet
        pairs = []
        for r in top:
            snippet = keyword_snippets.get(r.page_id, "") or clip_snippets.get(r.page_id, "")
            passage = f"{r.title}. {snippet}" if snippet else r.title
            pairs.append((query, passage[:512]))

        try:
            ce_scores = self._cross_encoder.predict(pairs)
            # Normalize CE scores to [0, 1] using sigmoid
            ce_norm = 1.0 / (1.0 + np.exp(-np.array(ce_scores, dtype=np.float32)))

            # Normalize stage-1 scores to [0, 1]
            s1_scores = np.array([r.score for r in top], dtype=np.float32)
            s1_min, s1_max = s1_scores.min(), s1_scores.max()
            if s1_max > s1_min:
                s1_norm = (s1_scores - s1_min) / (s1_max - s1_min)
            else:
                s1_norm = np.ones_like(s1_scores)

            # Blend
            w = self._RERANK_WEIGHT
            blended = (1 - w) * s1_norm + w * ce_norm

            # Re-sort top candidates by blended score
            order = np.argsort(-blended)
            reranked = []
            for idx in order:
                r = top[idx]
                reranked.append(_RankedResult(
                    page_id=r.page_id, url=r.url, title=r.title,
                    score=round(float(blended[idx]), 4),
                    text_score=r.text_score, image_score=r.image_score,
                    indexed_at=r.indexed_at, domain=r.domain,
                    tld_group=r.tld_group, content_category=r.content_category,
                ))
            top = reranked
        except Exception as e:
            log.warning("Cross-encoder re-ranking failed: %s", e)
            return results

        return top + rest

    async def load_index(self, session: AsyncSession):
        """Load page metadata from MySQL. Qdrant handles vector storage."""
        log.info("Loading page metadata from database...")
        self._result_cache.clear()
        t0 = time.time()

        # Load pages — select only needed columns (skip text_content MEDIUMTEXT)
        result = await session.execute(
            select(
                Page.id, Page.url, Page.title, Page.meta_description,
                Page.indexed_at, Page.last_modified, Page.language, Page.domain,
                Page.tld_group, Page.content_category, Page.nsfw_flag,
            ).where(Page.status == "indexed")
        )
        page_rows = result.all()
        self._pages = {
            r.id: {
                "url": r.url,
                "title": r.title or "",
                "meta_description": r.meta_description or "",
                "indexed_at": r.indexed_at.isoformat() if r.indexed_at else None,
                "last_modified": r.last_modified.isoformat() if r.last_modified else None,
                "language": r.language,
                "domain": r.domain,
                "tld_group": r.tld_group,
                "content_category": r.content_category,
            }
            for r in page_rows
        }
        self._nsfw_pages = {r.id for r in page_rows if r.nsfw_flag}

        # Load chunk counts from MySQL (fast aggregate query)
        from sqlalchemy import text as sql_text
        cc_result = await session.execute(sql_text(
            "SELECT page_id, COUNT(*) as cnt FROM text_embeddings GROUP BY page_id"
        ))
        self._chunk_counts = {int(r[0]): int(r[1]) for r in cc_result.fetchall()}

        # Precompute freshness signal (page age in days) and stale set
        now_ts = datetime.now(timezone.utc)
        self._page_age_days = {}
        self._stale_pages = set()
        for r in page_rows:
            ref_date = r.last_modified or r.indexed_at
            if ref_date:
                age = (now_ts - ref_date.replace(tzinfo=timezone.utc)).total_seconds() / 86400
                self._page_age_days[r.id] = max(age, 0.0)
            else:
                self._page_age_days[r.id] = 365.0
            if self._page_age_days[r.id] > STALE_THRESHOLD_DAYS:
                self._stale_pages.add(r.id)

        log.info("Loaded %d pages (%d stale, %.1fs)",
                 len(self._pages), len(self._stale_pages), time.time() - t0)

        # Build domain root lookup for navigational intent detection
        self._domain_roots = {}
        label_to_domains: dict[str, set[str]] = defaultdict(set)
        for pid, meta in self._pages.items():
            url = meta["url"]
            parsed = urlparse(url)
            if parsed.path in ("/", ""):
                host = parsed.netloc.lower()
                self._domain_roots[host] = pid
                reg_domain = (meta.get("domain") or host).lower()
                if reg_domain not in self._domain_roots:
                    self._domain_roots[reg_domain] = pid
                label = reg_domain.split(".")[0] if "." in reg_domain else reg_domain
                label_to_domains[label].add(host)
        self._domain_labels = {}
        for label, hosts in label_to_domains.items():
            if len(hosts) == 1:
                self._domain_labels[label] = next(iter(hosts))
            else:
                def _host_priority(h: str) -> tuple[int, int]:
                    parts = h.split(".")
                    if len(parts) == 2:
                        return (0, len(h))
                    if parts[0] == "www":
                        return (1, len(h))
                    if parts[0] == "en":
                        return (2, len(h))
                    return (3, len(h))
                best = min(hosts, key=_host_priority)
                self._domain_labels[label] = best

        # Precompute quality signal per page
        # Count pages per domain for domain authority signal
        self._domain_page_counts = defaultdict(int)
        for meta in self._pages.values():
            d = meta.get("domain")
            if d:
                self._domain_page_counts[d] += 1

        self._page_quality = {}
        for pid, meta in self._pages.items():
            n_chunks = self._chunk_counts.get(pid, 0)
            if n_chunks <= 1:
                depth = 0.05
            elif n_chunks <= 3:
                depth = 0.15
            else:
                depth = min(n_chunks / 15.0, 1.0)
            tld = (meta.get("tld_group") or "").lower()
            if tld in ("edu", "gov"):
                tld_auth = 1.0
            elif tld == "org":
                tld_auth = 0.7
            else:
                tld_auth = 0.4
            title = meta.get("title", "")
            title_ok = 1.0 if (10 <= len(title) <= 80 and not title.isupper()) else 0.3
            meta_desc = meta.get("meta_description", "")
            has_meta = 1.0 if len(meta_desc) > 40 else 0.0
            dom = meta.get("domain", "")
            domain_auth = min(self._domain_page_counts.get(dom, 0) / 10.0, 1.0)
            has_date = 1.0 if meta.get("last_modified") else 0.0
            self._page_quality[pid] = (
                0.35 * depth + 0.18 * tld_auth + 0.15 * title_ok
                + 0.12 * domain_auth + 0.12 * has_meta + 0.08 * has_date
            )

        self._max_page_id = max(self._pages.keys()) if self._pages else 0

        # Log Qdrant stats if available
        if self._qdrant:
            try:
                tc = self._qdrant.get_collection(TEXT_CHUNKS_V2_COLLECTION)
                ic = self._qdrant.get_collection(IMAGES_COLLECTION)
                log.info("Qdrant: %d text chunks, %d images", tc.points_count, ic.points_count)
            except Exception:
                pass

        elapsed = time.time() - t0
        log.info("Index loaded: %d pages, chunk counts for %d pages (%.1fs)",
                 len(self._pages), len(self._chunk_counts), elapsed)

    def _compute_page_quality(self, pid: int) -> float:
        """Compute quality score for a single page."""
        meta = self._pages.get(pid, {})
        n_chunks = self._chunk_counts.get(pid, 0)
        if n_chunks <= 1:
            depth = 0.05
        elif n_chunks <= 3:
            depth = 0.15
        else:
            depth = min(n_chunks / 15.0, 1.0)
        tld = (meta.get("tld_group") or "").lower()
        if tld in ("edu", "gov"):
            tld_auth = 1.0
        elif tld == "org":
            tld_auth = 0.7
        else:
            tld_auth = 0.4
        title = meta.get("title", "")
        title_ok = 1.0 if (10 <= len(title) <= 80 and not title.isupper()) else 0.3
        meta_desc = meta.get("meta_description", "")
        has_meta = 1.0 if len(meta_desc) > 40 else 0.0
        dom = meta.get("domain", "")
        domain_auth = min(self._domain_page_counts.get(dom, 0) / 10.0, 1.0)
        has_date = 1.0 if meta.get("last_modified") else 0.0
        return (
            0.35 * depth + 0.18 * tld_auth + 0.15 * title_ok
            + 0.12 * domain_auth + 0.12 * has_meta + 0.08 * has_date
        )

    async def incremental_update(self, session: AsyncSession) -> tuple[int, int, int]:
        """Load new pages from MySQL into page metadata cache.

        Qdrant updates happen at index time via the indexer, so we only
        need to refresh the in-memory page metadata here.
        Returns (new_pages, 0, 0) — text/image counts always 0 since
        Qdrant handles those directly.
        """
        t0 = time.time()
        new_pages = 0

        result = await session.execute(
            select(
                Page.id, Page.url, Page.title, Page.meta_description,
                Page.domain, Page.tld_group, Page.content_category,
                Page.language, Page.indexed_at, Page.last_modified, Page.nsfw_flag,
            ).where(Page.id > self._max_page_id, Page.status == "indexed")
        )
        page_rows = result.all()
        if page_rows:
            now_dt = datetime.now(timezone.utc)
            for r in page_rows:
                self._pages[r.id] = {
                    "url": r.url,
                    "title": r.title or "",
                    "meta_description": r.meta_description or "",
                    "domain": r.domain,
                    "tld_group": r.tld_group,
                    "content_category": r.content_category,
                    "language": r.language,
                    "indexed_at": r.indexed_at.isoformat() if r.indexed_at else None,
                    "last_modified": r.last_modified.isoformat() if r.last_modified else None,
                }
                if r.nsfw_flag:
                    self._nsfw_pages.add(r.id)
                if r.domain:
                    self._domain_page_counts[r.domain] += 1
                ref_date = r.last_modified or r.indexed_at
                if ref_date:
                    self._page_age_days[r.id] = max((now_dt - ref_date.replace(tzinfo=timezone.utc)).total_seconds() / 86400.0, 0.0)
                else:
                    self._page_age_days[r.id] = 365.0
                if self._page_age_days.get(r.id, 365.0) > STALE_THRESHOLD_DAYS:
                    self._stale_pages.add(r.id)
                self._page_quality[r.id] = self._compute_page_quality(r.id)
                # Update domain roots for navigational detection
                parsed = urlparse(r.url)
                if parsed.path in ("/", ""):
                    host = parsed.netloc.lower()
                    self._domain_roots[host] = r.id
                    reg_domain = (r.domain or host).lower()
                    if reg_domain not in self._domain_roots:
                        self._domain_roots[reg_domain] = r.id
            self._max_page_id = max(r.id for r in page_rows)
            new_pages = len(page_rows)

            # Also refresh chunk counts for new pages
            from sqlalchemy import text as sql_text
            new_pids = [r.id for r in page_rows]
            # Batch query chunk counts
            if new_pids:
                placeholders = ",".join(str(p) for p in new_pids)
                cc_result = await session.execute(sql_text(
                    f"SELECT page_id, COUNT(*) as cnt FROM text_embeddings "
                    f"WHERE page_id IN ({placeholders}) GROUP BY page_id"
                ))
                for row in cc_result.fetchall():
                    self._chunk_counts[int(row[0])] = int(row[1])
                # Recompute quality for new pages with updated chunk counts
                for r in page_rows:
                    self._page_quality[r.id] = self._compute_page_quality(r.id)

        if new_pages:
            self._result_cache.clear()
            log.info("Incremental update: +%d pages (%.1fs)", new_pages, time.time() - t0)

        return new_pages, 0, 0

    def _detect_site_intent(self, query: str) -> int | None:
        """Detect navigational queries like 'wikipedia' and return the root page_id."""
        q = query.strip().lower()
        if q in self._domain_roots:
            return self._domain_roots[q]
        if q in self._domain_labels:
            return self._domain_roots.get(self._domain_labels[q])
        tokens = q.split()
        first = tokens[0] if tokens else ""
        if first in self._domain_labels and len(first) >= 3:
            return self._domain_roots.get(self._domain_labels[first])
        return None

    def _qdrant_text_search(self, query_vec: np.ndarray, query_text: str,
                            limit: int = 2000,
                            qdrant_filter: Filter | None = None) -> list[dict]:
        """Hybrid search: dense vectors + BM25 keyword scoring via Qdrant RRF fusion.

        Returns list of {page_id, score, chunk_text}.
        """
        if self._qdrant is None:
            return []
        try:
            results = self._qdrant.query_points(
                collection_name=self._text_collection,
                prefetch=[
                    Prefetch(
                        query=query_vec.tolist(),
                        using="dense",
                        limit=limit,
                        filter=qdrant_filter,
                    ),
                    Prefetch(
                        query=Document(text=query_text, model="Qdrant/bm25"),
                        using="text-bm25",
                        limit=min(limit * 2, 4000),
                        filter=qdrant_filter,
                    ),
                ],
                query=FusionQuery(fusion=Fusion.RRF),
                limit=limit,
                with_payload=["page_id", "chunk_text"],
            )
            return [
                {
                    "page_id": p.payload["page_id"],
                    "score": p.score,
                    "chunk_text": p.payload["chunk_text"],
                }
                for p in results.points
            ]
        except Exception as e:
            log.warning("Qdrant hybrid search failed: %s", e)
            return []

    def _qdrant_dense_only_search(self, query_vec: np.ndarray, limit: int = 2000,
                                   qdrant_filter: Filter | None = None) -> list[dict]:
        """Dense-only search for Phase 1 (fast). Returns list of {page_id, score, chunk_text}."""
        if self._qdrant is None:
            return []
        try:
            results = self._qdrant.query_points(
                collection_name=self._text_collection,
                query=query_vec.tolist(),
                using="dense",
                query_filter=qdrant_filter,
                limit=limit,
                with_payload=["page_id", "chunk_text"],
                search_params=SearchParams(exact=False, hnsw_ef=128),
            )
            return [
                {
                    "page_id": p.payload["page_id"],
                    "score": p.score,
                    "chunk_text": p.payload["chunk_text"],
                }
                for p in results.points
            ]
        except Exception as e:
            log.warning("Qdrant dense search failed: %s", e)
            return []

    def _qdrant_image_search(self, query_vec: np.ndarray, limit: int = 500,
                             qdrant_filter: Filter | None = None) -> list[dict]:
        """Search Qdrant images collection."""
        if self._qdrant is None:
            return []
        try:
            results = self._qdrant.query_points(
                collection_name=IMAGES_COLLECTION,
                query=query_vec.tolist(),
                query_filter=qdrant_filter,
                limit=limit,
                with_payload=True,
                with_vectors=True,  # needed for visual dedup
            )
            return [
                {
                    "id": p.id,
                    "score": p.score,
                    "vector": np.array(p.vector, dtype=np.float32),
                    "image_url": p.payload["image_url"],
                    "alt_text": p.payload.get("alt_text", ""),
                    "nsfw_score": p.payload.get("nsfw_score", 0.0),
                    "icon_score": p.payload.get("icon_score", 0.0),
                    "page_ids": p.payload.get("page_ids", []),
                    "domain": p.payload.get("domain"),
                    "tld_group": p.payload.get("tld_group"),
                    "content_category": p.payload.get("content_category"),
                }
                for p in results.points
            ]
        except Exception as e:
            log.warning("Qdrant image search failed: %s", e)
            return []

    # _keyword_match_scores removed — BM25 scoring is now handled natively
    # by Qdrant's hybrid Prefetch + RRF fusion in _qdrant_text_search

    def _pick_snippet(self, page_id: int, keyword_snippet: str, clip_snippet: str,
                      query_words: set[str] | None = None) -> str:
        """Pick the best snippet by query relevance: score candidates and pick highest overlap."""
        page = self._pages.get(page_id, {})
        meta = page.get("meta_description", "")
        candidates: list[tuple[str, int]] = []  # (text, priority) — lower priority wins ties
        if keyword_snippet:
            candidates.append((keyword_snippet, 0))
        if clip_snippet:
            candidates.append((clip_snippet[:300], 1))
        if meta and len(meta) > 40 and not _is_junk_chunk(meta):
            candidates.append((meta[:300], 2))
        if not candidates:
            return meta[:300] if meta else ""
        if query_words:
            # Score by keyword overlap, break ties by priority
            candidates.sort(key=lambda c: (-_keyword_overlap(query_words, c[0]), c[1]))
        return candidates[0][0]

    _TIMELY_RE = re.compile(r"\b(news|latest|today|update|recent|2025|2026)\b", re.IGNORECASE)

    def _freshness_score(self, pid: int, is_timely: bool) -> float:
        age = self._page_age_days.get(pid, 365.0)
        half_life = 30.0 if is_timely else 180.0
        return math.exp(-0.693 * age / half_life)

    _TRANSACTIONAL_RE = re.compile(r"\b(buy|price|download|install|order|purchase|shop|deal|cheap|coupon)\b", re.IGNORECASE)

    def _classify_intent(self, query: str, query_words: set[str]) -> str:
        if self._detect_site_intent(query) is not None:
            return "navigational"
        # Check if query looks like a domain
        q = query.strip().lower()
        if re.match(r"^[a-z0-9-]+\.[a-z]{2,}$", q):
            return "navigational"
        if self._TRANSACTIONAL_RE.search(query):
            return "transactional"
        return "informational"

    @staticmethod
    def _title_boost(query_words: set[str], title: str) -> float:
        """Graduated title match boost based on content word overlap.

        Returns multiplier: 1.0 (no boost) to 1.5 (full match).
        """
        content_words = _filter_stopwords(query_words)
        if not content_words:
            return 1.0
        title_words = _tokenize(title)
        overlap = len(content_words & title_words) / len(content_words)
        if overlap >= 1.0:
            return 1.5
        if overlap >= 0.5:
            # Linear interpolation: 50% overlap → 1.25x, 100% → 1.5x
            return 1.25 + 0.25 * (overlap - 0.5) / 0.5
        return 1.0

    # Intent-specific weight profiles: (text, image, keyword, exact, freshness, quality)
    _INTENT_WEIGHTS = {
        "informational": (0.40, 0.15, 0.13, 0.08, 0.07, 0.17),
        "navigational":  (0.25, 0.08, 0.18, 0.22, 0.05, 0.22),
        "transactional": (0.35, 0.12, 0.15, 0.10, 0.10, 0.18),
    }

    def _apply_domain_diversity(self, results: list[_RankedResult], max_per_domain: int = 3, window: int = 10) -> list[_RankedResult]:
        if len(results) <= window:
            return results
        top = results[:window]
        rest = results[window:]
        reordered: list[_RankedResult] = []
        displaced: list[_RankedResult] = []
        domain_counts: dict[str, int] = defaultdict(int)
        for r in top:
            d = r.domain or r.url
            if domain_counts[d] < max_per_domain:
                domain_counts[d] += 1
                reordered.append(r)
            else:
                displaced.append(r)
        return reordered + displaced + rest

    def _deduplicate_results(self, results: list[_RankedResult], max_check: int = 200) -> list[_RankedResult]:
        kept: list[_RankedResult] = []
        domain_titles: dict[str, list[str]] = defaultdict(list)
        for i, r in enumerate(results):
            if i >= max_check:
                kept.extend(results[i:])
                break
            d = r.domain or r.url
            title_lower = r.title.lower()
            is_dup = False
            existing = domain_titles[d]
            for existing_title in existing[-10:]:
                if difflib.SequenceMatcher(None, title_lower, existing_title).ratio() > 0.9:
                    is_dup = True
                    break
            if not is_dup:
                existing.append(title_lower)
                kept.append(r)
        return kept

    def _make_cache_key(self, query: str, mode: str, lang: str | None, safe_search: bool, domain: str | None, tld_groups: list[str] | None, categories: list[str] | None, no_correct: bool = False, lang_hint: str | None = None, title_must_contain: list[str] | None = None, date_range_days: int | None = None, edu_boost: bool = False) -> tuple:
        return (query, mode, lang, safe_search, domain, tuple(tld_groups or []), tuple(categories or []), no_correct, lang_hint, tuple(title_must_contain or []), date_range_days, edu_boost)

    def _build_qdrant_filter(self, lang: str | None, safe_search: bool, domain: str | None,
                             tld_groups: list[str] | None, categories: list[str] | None,
                             include_stale: bool = False) -> Filter | None:
        return build_search_filter(lang=lang, safe_search=safe_search, domain=domain,
                                   tld_groups=tld_groups, categories=categories,
                                   include_stale=include_stale)

    def _search_ranked(self, query: str, limit: int = 10, offset: int = 0, mode: str = "hybrid", lang: str | None = None, safe_search: bool = True, domain: str | None = None, tld_groups: list[str] | None = None, categories: list[str] | None = None, no_correct: bool = False, lang_hint: str | None = None, title_must_contain: list[str] | None = None, date_range_days: int | None = None, edu_boost: bool = False) -> tuple[list[_RankedResult], float, set[str], dict[int, str], dict[int, str], str | None, str | None]:
        """Core ranking. Returns (ranked_results, elapsed_ms, query_words, clip_snippets, keyword_snippets, corrected_query, original_query). Uses cache."""
        query = " ".join(query.split())
        title_must_lower = [p.lower() for p in title_must_contain] if title_must_contain else []
        cache_key = self._make_cache_key(query, mode, lang, safe_search, domain, tld_groups, categories, no_correct, lang_hint, title_must_contain, date_range_days, edu_boost=edu_boost)
        now = time.time()

        # Check cache
        cached = self._result_cache.get(cache_key)
        if cached is not None:
            ranked, elapsed_ms, query_words, clip_snippets, kw_snippets_cached, corrected_q, original_q, ts = cached
            if now - ts < _RESULT_CACHE_TTL:
                self._result_cache.move_to_end(cache_key)
                return ranked, elapsed_ms, query_words, clip_snippets, kw_snippets_cached, corrected_q, original_q
            else:
                del self._result_cache[cache_key]

        t0 = time.time()
        query_vec = self._text_encoder.encode_query(query) if self._text_encoder else self.clip.encode_query(query)
        clip_query_vec = self.clip.encode_query(query) if self._text_encoder else query_vec
        query_words = _tokenize(query)
        corrected_query: str | None = None
        original_query: str | None = None

        # Include stale pages when user explicitly sets a date range
        include_stale = date_range_days is not None
        qdrant_filter = self._build_qdrant_filter(lang, safe_search, domain, tld_groups, categories,
                                                  include_stale=include_stale)

        # Oversample: multiple chunks per page, plus dedup/filtering reduces count
        qdrant_fetch = min((offset + limit) * 10, 2000)

        # Text similarities via Qdrant hybrid (dense + BM25 via RRF fusion)
        text_scores: dict[int, float] = {}
        clip_snippets: dict[int, str] = {}
        chunk_candidates: dict[int, list[str]] = {}  # up to 3 chunks per page
        if mode in ("hybrid", "text"):
            hits = self._qdrant_text_search(query_vec, query, limit=qdrant_fetch, qdrant_filter=qdrant_filter)
            for h in hits:
                pid = h["page_id"]
                if pid not in text_scores or h["score"] > text_scores[pid]:
                    text_scores[pid] = h["score"]
                    clip_snippets[pid] = h["chunk_text"]
                cands = chunk_candidates.setdefault(pid, [])
                if len(cands) < 3:
                    cands.append(h["chunk_text"])

        # Pick best chunk per page by keyword overlap
        for pid, cands in chunk_candidates.items():
            if len(cands) > 1:
                best = max(cands, key=lambda c: _keyword_overlap(query_words, c))
                clip_snippets[pid] = best

        # Image similarities via Qdrant (always uses CLIP vectors)
        image_scores: dict[int, float] = {}
        if mode in ("hybrid", "image"):
            img_hits = self._qdrant_image_search(clip_query_vec, limit=min(qdrant_fetch, 500))
            for h in img_hits:
                for pid in h["page_ids"]:
                    if pid not in image_scores or h["score"] > image_scores[pid]:
                        image_scores[pid] = h["score"]

        keyword_snippets: dict[int, str] = {}

        # Classify intent and select weight profile
        is_timely = bool(self._TIMELY_RE.search(query))
        intent = self._classify_intent(query, query_words)

        all_page_ids = list(set(text_scores.keys()) | set(image_scores.keys()))

        # Combine scores — RRF fusion already blends semantic + keyword
        results: list[_RankedResult] = []
        for pid in all_page_ids:
            if pid not in self._pages:
                continue
            page_meta = self._pages[pid]
            if title_must_lower:
                title_lower = page_meta.get("title", "").lower()
                if not all(phrase in title_lower for phrase in title_must_lower):
                    continue
            ts = text_scores.get(pid, 0.0)
            is_ = image_scores.get(pid, 0.0)
            fs = self._freshness_score(pid, is_timely)
            qs = self._page_quality.get(pid, 0.4)
            # Apply intent modifiers before scoring
            if intent == "navigational":
                qs *= 1.3; fs *= 0.5
            elif intent == "transactional":
                fs *= 1.5
            kw = _keyword_overlap(query_words, page_meta["title"] + " " + clip_snippets.get(pid, ""))
            if mode == "image":
                combined = is_
            else:
                combined = 0.45 * ts + 0.12 * is_ + 0.07 * fs + 0.15 * qs + 0.21 * kw
                combined *= self._title_boost(query_words, page_meta["title"])
            # Language hint soft boost
            if lang_hint and page_meta.get("language") == lang_hint:
                combined += 0.03
            # Multiplicative penalty for stub/thin pages
            if qs < 0.3:
                combined *= (0.3 + 0.7 * (qs / 0.3))
            # Stale pages now handled by freshness decay + STALE_THRESHOLD_DAYS retrieval cutoff
            # Edu boost: strongly favor academic/scientific sources
            if edu_boost:
                tld = (page_meta.get("tld_group") or "").lower()
                cat = (page_meta.get("content_category") or "").lower()
                if tld in ("edu",):
                    combined *= 1.8
                elif tld in ("gov", "org"):
                    combined *= 1.3
                if cat in ("education", "science", "reference"):
                    combined *= 1.4
                elif cat in ("health", "law", "technology"):
                    combined *= 1.1
            results.append(_RankedResult(
                page_id=pid,
                url=page_meta["url"],
                title=page_meta["title"],
                score=round(combined, 4),
                text_score=round(ts, 4),
                image_score=round(is_, 4),
                indexed_at=page_meta["indexed_at"],
                domain=page_meta.get("domain"),
                tld_group=page_meta.get("tld_group"),
                content_category=page_meta.get("content_category"),
            ))

        # Boost root page for navigational queries
        root_pid = self._detect_site_intent(query)
        if root_pid is not None and root_pid in self._pages:
            top_score = max((r.score for r in results), default=0.5)
            boosted_score = round(top_score + 0.1, 4)
            found = False
            for r in results:
                if r.page_id == root_pid:
                    r.score = boosted_score
                    found = True
                    break
            if not found:
                page_meta = self._pages[root_pid]
                results.append(_RankedResult(
                    page_id=root_pid,
                    url=page_meta["url"],
                    title=page_meta["title"],
                    score=boosted_score,
                    text_score=0.0,
                    image_score=0.0,
                    indexed_at=page_meta["indexed_at"],
                    domain=page_meta.get("domain"),
                    tld_group=page_meta.get("tld_group"),
                    content_category=page_meta.get("content_category"),
                ))

        results.sort(key=lambda r: r.score, reverse=True)
        # Stage 2: cross-encoder re-ranking
        if mode != "image":
            results = self._rerank(query, results, clip_snippets, keyword_snippets)
        results = self._apply_domain_diversity(results)
        results = self._deduplicate_results(results)
        results = [r for r in results if r.score >= 0.03]
        if safe_search:
            results = [r for r in results if r.page_id not in self._nsfw_pages]
        if date_range_days:
            from datetime import timedelta
            cutoff = (datetime.now() - timedelta(days=date_range_days)).isoformat()
            results = [r for r in results if r.indexed_at and r.indexed_at >= cutoff]

        # Lazy spell correction — only if few results or low top score
        if not no_correct and (len(results) < 3 or (results and results[0].score < 0.08)):
            corrected = self._spell_suggest(query)
            if corrected:
                corrected_query = corrected
                original_query = query

        elapsed_ms = round((time.time() - t0) * 1000, 2)

        # Store in cache with LRU eviction
        self._result_cache[cache_key] = (results, elapsed_ms, query_words, clip_snippets, keyword_snippets, corrected_query, original_query, now)
        if len(self._result_cache) > _RESULT_CACHE_MAX:
            self._result_cache.popitem(last=False)

        return results, elapsed_ms, query_words, clip_snippets, keyword_snippets, corrected_query, original_query

    def search(self, query: str, limit: int = 10, offset: int = 0, mode: str = "hybrid", lang: str | None = None, safe_search: bool = True, domain: str | None = None, tld_groups: list[str] | None = None, categories: list[str] | None = None, no_correct: bool = False, lang_hint: str | None = None, title_must_contain: list[str] | None = None, date_range_days: int | None = None, edu_boost: bool = False) -> tuple[list[SearchResult], int, float, str | None, str | None]:
        ranked, elapsed_ms, query_words, clip_snippets, kw_snippets, corrected_query, original_query = self._search_ranked(
            query, limit=limit, offset=offset, mode=mode, lang=lang, safe_search=safe_search, domain=domain, tld_groups=tld_groups, categories=categories, no_correct=no_correct, lang_hint=lang_hint, title_must_contain=title_must_contain, date_range_days=date_range_days, edu_boost=edu_boost
        )
        total = len(ranked)
        page = ranked[offset:offset + limit]
        results = [
            SearchResult(
                page_id=r.page_id,
                url=r.url,
                title=r.title,
                snippet=self._pick_snippet(r.page_id, kw_snippets.get(r.page_id, ""), clip_snippets.get(r.page_id, ""), query_words),
                score=r.score,
                text_score=r.text_score,
                image_score=r.image_score,
                indexed_at=r.indexed_at,
                domain=r.domain,
                tld_group=r.tld_group,
                content_category=r.content_category,
            )
            for r in page
        ]
        return results, total, elapsed_ms, corrected_query, original_query

    def search_stream(self, query: str, limit: int = 10, offset: int = 0, mode: str = "hybrid", lang: str | None = None, safe_search: bool = True, domain: str | None = None, tld_groups: list[str] | None = None, categories: list[str] | None = None, no_correct: bool = False, lang_hint: str | None = None, title_must_contain: list[str] | None = None, date_range_days: int | None = None, edu_boost: bool = False) -> Generator[tuple[str, dict], None, None]:
        """Two-phase SSE streaming: fast dense-only results first, then keyword+cross-encoder rerank."""
        query = " ".join(query.split())
        title_must_lower = [p.lower() for p in title_must_contain] if title_must_contain else []
        cache_key = self._make_cache_key(query, mode, lang, safe_search, domain, tld_groups, categories, no_correct, lang_hint, title_must_contain, date_range_days, edu_boost=edu_boost)
        now = time.time()

        # Check cache — if hit, serve final results directly
        cached = self._result_cache.get(cache_key)
        if cached is not None:
            ranked, elapsed_ms, query_words, clip_snippets, kw_snippets_cached, corrected_q, original_q, ts = cached
            if now - ts < _RESULT_CACHE_TTL:
                self._result_cache.move_to_end(cache_key)
                total = len(ranked)
                meta: dict = {"query": query, "total_results": total, "search_time_ms": elapsed_ms}
                if corrected_q:
                    meta["corrected_query"] = corrected_q
                    meta["original_query"] = original_q
                yield ("meta", meta)
                page = ranked[offset:offset + limit]
                for r in page:
                    snippet = self._pick_snippet(r.page_id, kw_snippets_cached.get(r.page_id, ""), clip_snippets.get(r.page_id, ""), query_words)
                    yield ("result", {
                        "page_id": r.page_id, "url": r.url, "title": r.title,
                        "snippet": snippet, "score": r.score,
                        "text_score": r.text_score, "image_score": r.image_score,
                        "indexed_at": r.indexed_at, "domain": r.domain,
                        "tld_group": r.tld_group, "content_category": r.content_category,
                        "meta_description": self._pages.get(r.page_id, {}).get("meta_description", ""),
                    })
                yield ("done", {})
                return
            else:
                del self._result_cache[cache_key]

        t0 = time.time()

        # ── Phase 1: Dense vector search + basic scoring (fast) ──
        query_vec = self._text_encoder.encode_query(query) if self._text_encoder else self.clip.encode_query(query)
        clip_query_vec = self.clip.encode_query(query) if self._text_encoder else query_vec
        query_words = _tokenize(query)

        include_stale = date_range_days is not None
        qdrant_filter = self._build_qdrant_filter(lang, safe_search, domain, tld_groups, categories,
                                                  include_stale=include_stale)

        # Phase 1 fetches current + next page worth of results (20 pages * 10x oversample)
        p1_fetch = min((offset + limit * 2) * 10, 2000)

        text_scores: dict[int, float] = {}
        clip_snippets: dict[int, str] = {}
        chunk_candidates: dict[int, list[str]] = {}
        if mode in ("hybrid", "text"):
            hits = self._qdrant_dense_only_search(query_vec, limit=p1_fetch, qdrant_filter=qdrant_filter)
            for h in hits:
                pid = h["page_id"]
                if pid not in text_scores or h["score"] > text_scores[pid]:
                    text_scores[pid] = h["score"]
                    clip_snippets[pid] = h["chunk_text"]
                cands = chunk_candidates.setdefault(pid, [])
                if len(cands) < 3:
                    cands.append(h["chunk_text"])

        # Pick best chunk per page by keyword overlap
        for pid, cands in chunk_candidates.items():
            if len(cands) > 1:
                best = max(cands, key=lambda c: _keyword_overlap(query_words, c))
                clip_snippets[pid] = best

        image_scores: dict[int, float] = {}
        if mode in ("hybrid", "image"):
            img_hits = self._qdrant_image_search(clip_query_vec, limit=min(p1_fetch, 500))
            for h in img_hits:
                for pid in h["page_ids"]:
                    if pid not in image_scores or h["score"] > image_scores[pid]:
                        image_scores[pid] = h["score"]

        all_page_ids = set(text_scores.keys()) | set(image_scores.keys())
        is_timely = bool(self._TIMELY_RE.search(query))
        intent = self._classify_intent(query, query_words)

        # Build phase-1 results with dense-only scores + freshness + quality
        p1_results: list[_RankedResult] = []
        for pid in all_page_ids:
            if pid not in self._pages:
                continue
            page_meta = self._pages[pid]
            if title_must_lower:
                title_lower = page_meta.get("title", "").lower()
                if not all(phrase in title_lower for phrase in title_must_lower):
                    continue
            ts_score = text_scores.get(pid, 0.0)
            is_ = image_scores.get(pid, 0.0)
            fs = self._freshness_score(pid, is_timely)
            qs = self._page_quality.get(pid, 0.4)
            # Apply intent modifiers before scoring
            if intent == "navigational":
                qs *= 1.3; fs *= 0.5
            elif intent == "transactional":
                fs *= 1.5
            kw = _keyword_overlap(query_words, page_meta["title"] + " " + clip_snippets.get(pid, ""))
            if mode == "image":
                combined = is_
            else:
                combined = 0.38 * ts_score + 0.10 * is_ + 0.07 * fs + 0.18 * qs + 0.27 * kw
                combined *= self._title_boost(query_words, page_meta["title"])
            if lang_hint and page_meta.get("language") == lang_hint:
                combined += 0.03
            if qs < 0.3:
                combined *= (0.3 + 0.7 * (qs / 0.3))
            # Stale pages now handled by freshness decay + STALE_THRESHOLD_DAYS retrieval cutoff
            if edu_boost:
                _tld = (page_meta.get("tld_group") or "").lower()
                _cat = (page_meta.get("content_category") or "").lower()
                if _tld in ("edu",):
                    combined *= 1.8
                elif _tld in ("gov", "org"):
                    combined *= 1.3
                if _cat in ("education", "science", "reference"):
                    combined *= 1.4
                elif _cat in ("health", "law", "technology"):
                    combined *= 1.1
            p1_results.append(_RankedResult(
                page_id=pid, url=page_meta["url"], title=page_meta["title"],
                score=round(combined, 4), text_score=round(ts_score, 4),
                image_score=round(is_, 4), indexed_at=page_meta["indexed_at"],
                domain=page_meta.get("domain"), tld_group=page_meta.get("tld_group"),
                content_category=page_meta.get("content_category"),
            ))

        # Navigational boost
        root_pid = self._detect_site_intent(query)
        if root_pid is not None and root_pid in self._pages:
            top_score = max((r.score for r in p1_results), default=0.5)
            boosted_score = round(top_score + 0.1, 4)
            found = False
            for r in p1_results:
                if r.page_id == root_pid:
                    r.score = boosted_score
                    found = True
                    break
            if not found:
                pm = self._pages[root_pid]
                p1_results.append(_RankedResult(
                    page_id=root_pid, url=pm["url"], title=pm["title"],
                    score=boosted_score, text_score=0.0, image_score=0.0,
                    indexed_at=pm["indexed_at"], domain=pm.get("domain"),
                    tld_group=pm.get("tld_group"), content_category=pm.get("content_category"),
                ))

        p1_results.sort(key=lambda r: r.score, reverse=True)
        p1_results_unfiltered = list(p1_results)
        p1_results = self._apply_domain_diversity(p1_results)
        p1_results = self._deduplicate_results(p1_results)
        p1_results = [r for r in p1_results if r.score >= 0.03]
        if safe_search:
            p1_results = [r for r in p1_results if r.page_id not in self._nsfw_pages]
        if date_range_days:
            from datetime import timedelta
            cutoff = (datetime.now() - timedelta(days=date_range_days)).isoformat()
            p1_results = [r for r in p1_results if r.indexed_at and r.indexed_at >= cutoff]

        phase1_ms = round((time.time() - t0) * 1000, 2)

        # Yield phase-1 results immediately
        p1_page = p1_results[offset:offset + limit]
        yield ("meta", {"query": query, "total_results": len(p1_results), "search_time_ms": phase1_ms, "phase": 1})
        for r in p1_page:
            snippet = self._pick_snippet(r.page_id, "", clip_snippets.get(r.page_id, ""), query_words)
            yield ("result", {
                "page_id": r.page_id, "url": r.url, "title": r.title,
                "snippet": snippet, "score": r.score,
                "text_score": r.text_score, "image_score": r.image_score,
                "indexed_at": r.indexed_at, "domain": r.domain,
                "tld_group": r.tld_group, "content_category": r.content_category,
                "meta_description": self._pages.get(r.page_id, {}).get("meta_description", ""),
            })

        # ── Phase 2: Hybrid search (dense + BM25 via RRF) + cross-encoder reranking ──
        if mode == "image":
            yield ("done", {})
            self._result_cache[cache_key] = (p1_results, phase1_ms, query_words, clip_snippets, {}, None, None, now)
            if len(self._result_cache) > _RESULT_CACHE_MAX:
                self._result_cache.popitem(last=False)
            return

        # Run hybrid search (single Qdrant call with RRF fusion of dense + BM25)
        # Phase 2 only needs to refine current page results
        p2_fetch = min((offset + limit) * 10, 2000)
        hybrid_hits = self._qdrant_text_search(query_vec, query, limit=p2_fetch, qdrant_filter=qdrant_filter)
        hybrid_scores: dict[int, float] = {}
        p2_chunk_candidates: dict[int, list[str]] = {}
        for h in hybrid_hits:
            pid = h["page_id"]
            if pid not in hybrid_scores or h["score"] > hybrid_scores[pid]:
                hybrid_scores[pid] = h["score"]
                clip_snippets[pid] = h["chunk_text"]
            cands = p2_chunk_candidates.setdefault(pid, [])
            if len(cands) < 3:
                cands.append(h["chunk_text"])

        # Pick best chunk per page by keyword overlap
        for pid, cands in p2_chunk_candidates.items():
            if len(cands) > 1:
                best = max(cands, key=lambda c: _keyword_overlap(query_words, c))
                clip_snippets[pid] = best

        keyword_snippets: dict[int, str] = {}
        corrected_query: str | None = None
        original_query: str | None = None

        # Re-score with hybrid signals
        p2_results: list[_RankedResult] = []
        for r in p1_results_unfiltered:
            pid = r.page_id
            ts_score = hybrid_scores.get(pid, text_scores.get(pid, 0.0))
            is_ = r.image_score
            fs = self._freshness_score(pid, is_timely)
            qs = self._page_quality.get(pid, 0.4)
            # Apply intent modifiers before scoring
            if intent == "navigational":
                qs *= 1.3; fs *= 0.5
            elif intent == "transactional":
                fs *= 1.5
            kw = _keyword_overlap(query_words, r.title + " " + clip_snippets.get(pid, ""))
            if mode == "text":
                combined = 0.55 * ts_score + 0.08 * fs + 0.16 * qs + 0.21 * kw
            else:
                combined = 0.45 * ts_score + 0.12 * is_ + 0.07 * fs + 0.15 * qs + 0.21 * kw
            combined *= self._title_boost(query_words, r.title)
            if lang_hint and self._pages.get(pid, {}).get("language") == lang_hint:
                combined += 0.03
            if qs < 0.3:
                combined *= (0.3 + 0.7 * (qs / 0.3))
            # Stale pages now handled by freshness decay + STALE_THRESHOLD_DAYS retrieval cutoff
            if edu_boost:
                _tld = (self._pages.get(pid, {}).get("tld_group") or "").lower()
                _cat = (self._pages.get(pid, {}).get("content_category") or "").lower()
                if _tld in ("edu",):
                    combined *= 1.8
                elif _tld in ("gov", "org"):
                    combined *= 1.3
                if _cat in ("education", "science", "reference"):
                    combined *= 1.4
                elif _cat in ("health", "law", "technology"):
                    combined *= 1.1
            p2_results.append(_RankedResult(
                page_id=pid, url=r.url, title=r.title,
                score=round(combined, 4), text_score=r.text_score,
                image_score=r.image_score, indexed_at=r.indexed_at,
                domain=r.domain, tld_group=r.tld_group,
                content_category=r.content_category,
            ))

        # Navigational boost
        if root_pid is not None and root_pid in self._pages:
            top_score = max((r.score for r in p2_results), default=0.5)
            boosted_score = round(top_score + 0.1, 4)
            found = False
            for r in p2_results:
                if r.page_id == root_pid:
                    r.score = boosted_score
                    found = True
                    break
            if not found:
                pm = self._pages[root_pid]
                p2_results.append(_RankedResult(
                    page_id=root_pid, url=pm["url"], title=pm["title"],
                    score=boosted_score, text_score=0.0, image_score=0.0,
                    indexed_at=pm["indexed_at"], domain=pm.get("domain"),
                    tld_group=pm.get("tld_group"), content_category=pm.get("content_category"),
                ))

        p2_results.sort(key=lambda r: r.score, reverse=True)
        p2_results = self._rerank(query, p2_results, clip_snippets, keyword_snippets)
        p2_results = self._apply_domain_diversity(p2_results)
        p2_results = self._deduplicate_results(p2_results)
        p2_results = [r for r in p2_results if r.score >= 0.03]
        if safe_search:
            p2_results = [r for r in p2_results if r.page_id not in self._nsfw_pages]
        if date_range_days:
            from datetime import timedelta
            cutoff = (datetime.now() - timedelta(days=date_range_days)).isoformat()
            p2_results = [r for r in p2_results if r.indexed_at and r.indexed_at >= cutoff]

        # Lazy spell correction
        if not no_correct and (len(p2_results) < 3 or (p2_results and p2_results[0].score < 0.08)):
            corrected = self._spell_suggest(query)
            if corrected:
                corrected_query = corrected
                original_query = query

        elapsed_ms = round((time.time() - t0) * 1000, 2)

        # Check if phase-2 ordering differs from phase-1
        p1_ids = [r.page_id for r in p1_page]
        p2_page = p2_results[offset:offset + limit]
        p2_ids = [r.page_id for r in p2_page]

        if p1_ids != p2_ids or corrected_query:
            rerank_data: dict = {
                "search_time_ms": elapsed_ms,
                "total_results": len(p2_results),
                "results": [],
            }
            if corrected_query:
                rerank_data["corrected_query"] = corrected_query
                rerank_data["original_query"] = original_query
            for r in p2_page:
                snippet = self._pick_snippet(r.page_id, keyword_snippets.get(r.page_id, ""), clip_snippets.get(r.page_id, ""), query_words)
                rerank_data["results"].append({
                    "page_id": r.page_id, "url": r.url, "title": r.title,
                    "snippet": snippet, "score": r.score,
                    "text_score": r.text_score, "image_score": r.image_score,
                    "indexed_at": r.indexed_at, "domain": r.domain,
                    "tld_group": r.tld_group, "content_category": r.content_category,
                    "meta_description": self._pages.get(r.page_id, {}).get("meta_description", ""),
                })
            yield ("rerank", rerank_data)

        yield ("done", {})

        # Cache the final (phase-2) results
        self._result_cache[cache_key] = (p2_results, elapsed_ms, query_words, clip_snippets, keyword_snippets, corrected_query, original_query, now)
        if len(self._result_cache) > _RESULT_CACHE_MAX:
            self._result_cache.popitem(last=False)

    def search_images_stream(self, query: str, limit: int = 20, offset: int = 0, lang: str | None = None, safe_search: bool = True, domain: str | None = None, tld_groups: list[str] | None = None, categories: list[str] | None = None, icons_only: bool = False) -> Generator[tuple[str, dict], None, None]:
        """Yield (event_type, data_dict) tuples for SSE image streaming."""
        results, total, elapsed_ms = self.search_images(
            query, limit=limit, offset=offset, lang=lang, safe_search=safe_search,
            domain=domain, tld_groups=tld_groups, categories=categories, icons_only=icons_only,
        )
        yield ("meta", {"query": query, "total_results": total, "search_time_ms": elapsed_ms})

        for r in results:
            yield ("result", {
                "image_url": r.image_url,
                "alt_text": r.alt_text,
                "score": r.score,
                "page_id": r.page_id,
                "page_url": r.page_url,
                "page_title": r.page_title,
                "found_on": r.found_on,
                "possibly_explicit": r.possibly_explicit,
                "domain": r.domain,
                "tld_group": r.tld_group,
                "content_category": r.content_category,
                "all_page_ids": list(set(r.all_page_ids)) if r.all_page_ids else [r.page_id],
            })

        yield ("done", {})

    _IMAGE_DEDUP_THRESHOLD = 0.97  # cosine similarity above this = visual duplicate

    def search_images(self, query: str, limit: int = 20, offset: int = 0, lang: str | None = None, safe_search: bool = True, domain: str | None = None, tld_groups: list[str] | None = None, categories: list[str] | None = None, icons_only: bool = False) -> tuple[list[ImageSearchResult], int, float]:
        t0 = time.time()
        if self._qdrant is None:
            return [], 0, 0.0

        query_vec = self.clip.encode_query(query)

        # Build filter for images
        must = []
        if safe_search:
            must.append(FieldCondition(key="nsfw_score", range=Range(lt=NSFW_IMAGE_THRESHOLD)))
        if domain:
            must.append(FieldCondition(key="domain", match=MatchValue(value=domain)))
        if tld_groups:
            must.append(FieldCondition(key="tld_group", match=MatchAny(any=tld_groups)))
        if categories:
            must.append(FieldCondition(key="content_category", match=MatchAny(any=categories)))

        img_filter = Filter(must=must) if must else None
        img_hits = self._qdrant_image_search(query_vec, limit=500, qdrant_filter=img_filter)

        deduped: list[ImageSearchResult] = []
        selected_vecs: list[np.ndarray] = []

        for h in img_hits:
            score = h["score"]
            if score < 0.2:
                break

            page_ids = h["page_ids"]
            pid = None
            page = None
            for candidate_pid in page_ids:
                p = self._pages.get(candidate_pid)
                if p is None:
                    continue
                if lang and p.get("language") != lang:
                    continue
                pid = candidate_pid
                page = p
                break
            if page is None:
                continue

            is_explicit = h["nsfw_score"] >= NSFW_IMAGE_THRESHOLD
            if safe_search and is_explicit:
                continue

            is_icon = h["icon_score"] >= ICON_IMAGE_THRESHOLD
            if icons_only and not is_icon:
                continue
            if not icons_only and is_icon:
                continue

            # Visual dedup
            vec = h["vector"]
            is_visual_dup = False
            for i, sv in enumerate(selected_vecs):
                if np.dot(vec, sv) > self._IMAGE_DEDUP_THRESHOLD:
                    deduped[i].found_on += len(page_ids)
                    if deduped[i].all_page_ids is not None:
                        deduped[i].all_page_ids.extend(page_ids)
                    is_visual_dup = True
                    break
            if is_visual_dup:
                continue

            selected_vecs.append(vec)
            deduped.append(ImageSearchResult(
                image_url=h["image_url"],
                alt_text=h["alt_text"],
                score=round(score, 4),
                page_id=pid,
                page_url=page["url"],
                page_title=page["title"],
                found_on=len(page_ids),
                possibly_explicit=is_explicit,
                domain=page.get("domain"),
                tld_group=page.get("tld_group"),
                content_category=page.get("content_category"),
                all_page_ids=list(page_ids),
            ))

        total = len(deduped)
        elapsed_ms = (time.time() - t0) * 1000
        return deduped[offset:offset + limit], total, round(elapsed_ms, 2)

    def available_filters(self) -> dict:
        """Return distinct values for each filter field (for UI dropdowns)."""
        domains: set[str] = set()
        tld_groups: set[str] = set()
        categories: set[str] = set()
        for page in self._pages.values():
            if page.get("domain"):
                domains.add(page["domain"])
            if page.get("tld_group"):
                tld_groups.add(page["tld_group"])
            if page.get("content_category"):
                categories.add(page["content_category"])
        return {
            "domains": sorted(domains),
            "tld_groups": sorted(tld_groups),
            "categories": sorted(categories),
        }

    @property
    def stats(self) -> dict:
        text_count = 0
        image_count = 0
        if self._qdrant:
            try:
                tc = self._qdrant.get_collection(TEXT_CHUNKS_V2_COLLECTION)
                text_count = tc.points_count
            except Exception:
                pass
            try:
                ic = self._qdrant.get_collection(IMAGES_COLLECTION)
                image_count = ic.points_count
            except Exception:
                pass
        return {
            "pages": len(self._pages),
            "text_embeddings": text_count,
            "image_embeddings": image_count,
        }

    def warmup_cache(self, queries: list[str]) -> int:
        """Pre-warm the result cache with popular queries."""
        cached = 0
        for q in queries:
            q = q.strip()
            if not q or len(q) < 2:
                continue
            try:
                self._search_ranked(q, mode="hybrid", safe_search=True)
                cached += 1
            except Exception:
                pass
        return cached
