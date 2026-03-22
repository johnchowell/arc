import asyncio
import datetime
import logging
import httpx
import numpy as np
from langdetect import detect, LangDetectException
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from semantic_searcher.config import settings
from semantic_searcher.database import async_session
from semantic_searcher.models.db import Page, TextEmbedding, ImageEmbedding, ImagePageSource
from semantic_searcher.utils.robots import USER_AGENT
from semantic_searcher.services.clip_service import CLIPService
from semantic_searcher.services.renderer import RendererService, needs_rendering, MIN_TEXT_LENGTH
from semantic_searcher.utils.content_extractor import extract_content, chunk_text
from semantic_searcher.utils.image_loader import load_image
from semantic_searcher.utils.url_utils import normalize_url, url_hash, extract_domain, extract_tld_group
from semantic_searcher.services.content_classifier import classify_page
from semantic_searcher.utils.nsfw_scorer import (
    score_image_nsfw, score_page_text_nsfw, NSFW_IMAGE_THRESHOLD,
    score_image_icon,
)

log = logging.getLogger(__name__)


def _detect_language(text: str) -> str | None:
    try:
        return detect(text[:2000]) if text and len(text.strip()) > 20 else None
    except LangDetectException:
        return None


def _extract_content_sync(html: str, url: str):
    """Wrapper for thread-pool execution of content extraction."""
    return extract_content(html, url)


def _parse_date(date_str: str) -> datetime.datetime | None:
    """Best-effort parse of a date string from page metadata."""
    from dateutil import parser as dateutil_parser
    try:
        dt = dateutil_parser.parse(date_str, fuzzy=True)
        # Sanity check: reject dates too far in the future or before the web era
        if dt.year < 1990 or dt.year > datetime.datetime.now().year + 1:
            return None
        return dt.replace(tzinfo=None)  # store as naive UTC
    except Exception:
        return None


def _write_html_cache(uhash: str, html: str):
    """Write HTML cache file (for thread-pool execution)."""
    cache_dir = settings.html_cache_dir / uhash[:2]
    cache_dir.mkdir(parents=True, exist_ok=True)
    (cache_dir / f"{uhash}.html").write_text(html, encoding="utf-8")


class IndexerService:
    def __init__(self, clip: CLIPService | list[CLIPService], renderer: RendererService | None = None, qdrant=None, text_encoder=None):
        if isinstance(clip, list):
            self._clips = clip
        else:
            self._clips = [clip]
        self._clip_index = 0
        self._renderer = renderer
        self._qdrant = qdrant  # QdrantClient instance for dual-write
        self._text_encoder = text_encoder  # MPNet text encoder (optional, for v2 collection)

    async def _index_to_qdrant(self, page_id: int, url: str, title: str | None,
                               domain: str | None, tld_group: str | None,
                               content_category: str | None, language: str | None,
                               indexed_at, nsfw_flag: bool, chunks: list[str],
                               text_emb_ids: list[int], text_vectors: list,
                               meta_description: str | None = None,
                               last_modified=None):
        """Write text chunk embeddings to Qdrant text_chunks_v2 (best-effort, after MySQL commit)."""
        if self._qdrant is None:
            return
        try:
            from qdrant_client.models import Document, PointStruct
            title_str = title or ""
            points = []
            for i, (chunk, emb_id, vec) in enumerate(zip(chunks, text_emb_ids, text_vectors)):
                bm25_text = f"{title_str} {chunk}".strip()
                points.append(PointStruct(
                    id=emb_id,
                    vector={
                        "dense": vec.tolist(),
                        "text-bm25": Document(text=bm25_text, model="Qdrant/bm25"),
                    },
                    payload={
                        "page_id": page_id,
                        "chunk_index": i,
                        "chunk_text": chunk,
                        "mysql_emb_id": emb_id,
                        "url": url,
                        "title": title_str,
                        "domain": domain,
                        "tld_group": tld_group,
                        "content_category": content_category,
                        "language": language,
                        "nsfw_flag": nsfw_flag,
                        "is_stale": False,
                        "indexed_at": indexed_at.isoformat() if indexed_at else None,
                        "last_modified": last_modified.isoformat() if last_modified else None,
                        "meta_description": meta_description or "",
                    },
                ))
            if points:
                self._qdrant.upsert("text_chunks_v2", points=points, wait=False)
        except Exception as e:
            log.warning("Qdrant text index failed for page %d: %s", page_id, e)

    async def _index_images_to_qdrant(self, image_rows: list[dict]):
        """Write image embeddings to Qdrant (best-effort)."""
        if self._qdrant is None or not image_rows:
            return
        try:
            from qdrant_client.models import PointStruct
            points = []
            for img in image_rows:
                points.append(PointStruct(
                    id=img["id"],
                    vector=img["vector"].tolist(),
                    payload={
                        "image_url": img["image_url"],
                        "alt_text": img.get("alt_text") or "",
                        "nsfw_score": img.get("nsfw_score", 0.0),
                        "icon_score": img.get("icon_score", 0.0),
                        "page_ids": img.get("page_ids", []),
                        "mysql_img_id": img["id"],
                        "domain": img.get("domain"),
                        "tld_group": img.get("tld_group"),
                        "content_category": img.get("content_category"),
                        "language": img.get("language"),
                    },
                ))
            if points:
                self._qdrant.upsert("images", points=points, wait=False)
        except Exception as e:
            log.warning("Qdrant image index failed: %s", e)

    @property
    def clips(self) -> list[CLIPService]:
        return self._clips

    @property
    def clip(self) -> CLIPService:
        """Round-robin across CLIP workers."""
        c = self._clips[self._clip_index]
        self._clip_index = (self._clip_index + 1) % len(self._clips)
        return c

    async def index_batch(
        self, items: list[tuple[str, str]], session: AsyncSession | None = None, clip: CLIPService | None = None,
    ) -> list[Page | None]:
        """Index multiple (url, html) pairs with batched CLIP encoding.

        Each phase is committed separately to avoid long transactions and deadlocks.
        The session parameter is deprecated and ignored — sessions are managed internally.
        """
        clip = clip or self.clip
        results_map: dict[int, Page | None] = {}

        # ── Phase 1a: DB get-or-create page rows ──
        page_rows = []  # (idx, page_id, uhash, normalized, html)
        async with async_session() as phase1_session:
            conn = await phase1_session.connection()

            for idx, (url, html) in enumerate(items):
                try:
                    normalized = normalize_url(url)
                    uhash = url_hash(url)

                    result = await phase1_session.execute(
                        select(Page.id, Page.status).where(Page.url_hash == uhash)
                    )
                    row = result.first()

                    if row and row.status == "indexed":
                        r = await phase1_session.execute(select(Page).where(Page.id == row.id))
                        results_map[idx] = r.scalar_one()
                        continue

                    if row is None:
                        await conn.execute(text(
                            "INSERT IGNORE INTO pages (url, url_hash, status) VALUES (:url, :uhash, 'indexing')"
                        ), {"url": normalized, "uhash": uhash})
                        result = await phase1_session.execute(
                            select(Page.id, Page.status).where(Page.url_hash == uhash)
                        )
                        row = result.first()
                        if row is None:
                            results_map[idx] = None
                            continue
                        if row.status == "indexed":
                            r = await phase1_session.execute(select(Page).where(Page.id == row.id))
                            results_map[idx] = r.scalar_one()
                            continue

                    page_id = row.id
                    await conn.execute(
                        Page.__table__.update().where(Page.__table__.c.id == page_id).values(status="indexing")
                    )
                    page_rows.append((idx, page_id, uhash, normalized, html))

                except Exception as e:
                    log.error("Batch prepare failed for %s: %s", url, e)
                    results_map[idx] = None

            await phase1_session.commit()

        if not page_rows:
            return [results_map.get(i) for i in range(len(items))]

        # ── Phase 1b: Parallel content extraction in thread pool ──
        loop = asyncio.get_running_loop()
        extract_tasks = []
        for idx, page_id, uhash, normalized, html in page_rows:
            extract_tasks.append(loop.run_in_executor(None, _extract_content_sync, html, normalized))

        extract_results = await asyncio.gather(*extract_tasks, return_exceptions=True)

        # SPA rendering for pages that need it (concurrent via renderer pool)
        spa_render_tasks = []
        spa_indices = []
        for i, ((idx, page_id, uhash, normalized, html), result) in enumerate(zip(page_rows, extract_results)):
            if isinstance(result, Exception):
                continue
            if len(result.text) < MIN_TEXT_LENGTH and needs_rendering(html) and self._renderer:
                spa_render_tasks.append(self._renderer.render(normalized))
                spa_indices.append(i)

        if spa_render_tasks:
            spa_results = await asyncio.gather(*spa_render_tasks, return_exceptions=True)
            for spa_i, spa_result in zip(spa_indices, spa_results):
                if isinstance(spa_result, Exception) or spa_result is None:
                    continue
                idx, page_id, uhash, normalized, html = page_rows[spa_i]
                # Re-extract from rendered HTML in thread pool
                new_content = await loop.run_in_executor(None, _extract_content_sync, spa_result, normalized)
                extract_results[spa_i] = new_content
                page_rows[spa_i] = (idx, page_id, uhash, normalized, spa_result)

        # Language detection in thread pool + build prepared list
        prepared = []
        lang_tasks = []
        prep_indices = []
        for i, ((idx, page_id, uhash, normalized, html), content_result) in enumerate(zip(page_rows, extract_results)):
            if isinstance(content_result, Exception):
                log.error("Content extraction failed for page %d: %s", page_id, content_result)
                results_map[idx] = None
                continue
            lang_tasks.append(loop.run_in_executor(None, _detect_language, content_result.text))
            prep_indices.append(i)

        lang_results = await asyncio.gather(*lang_tasks, return_exceptions=True)

        # HTML cache writes in thread pool (fire all at once)
        cache_tasks = []
        for pi, lang in zip(prep_indices, lang_results):
            idx, page_id, uhash, normalized, html = page_rows[pi]
            content = extract_results[pi]
            lang_val = lang if not isinstance(lang, Exception) else None
            domain = extract_domain(normalized)
            tld_group = extract_tld_group(normalized)
            chunks = chunk_text(content.text)
            prepared.append((idx, page_id, uhash, normalized, content, chunks, domain, tld_group, lang_val))
            cache_tasks.append(loop.run_in_executor(None, _write_html_cache, uhash, html))

        # Update page metadata + clear old embeddings in DB, while cache writes run in background
        cache_gather = asyncio.gather(*cache_tasks, return_exceptions=True) if cache_tasks else None

        async with async_session() as phase3_session:
            phase3_conn = await phase3_session.connection()
            for idx, page_id, uhash, normalized, content, chunks, domain, tld_group, lang in prepared:
                last_mod = _parse_date(content.published_date) if content.published_date else None
                await phase3_conn.execute(
                    Page.__table__.update().where(Page.__table__.c.id == page_id).values(
                        title=content.title[:1024] if content.title else None,
                        text_content=content.text,
                        meta_description=content.meta_description or None,
                        language=lang,
                        last_modified=last_mod,
                    )
                )
                await phase3_conn.execute(
                    TextEmbedding.__table__.delete().where(TextEmbedding.__table__.c.page_id == page_id)
                )
                await phase3_conn.execute(
                    ImagePageSource.__table__.delete().where(ImagePageSource.__table__.c.page_id == page_id)
                )
            await phase3_session.commit()

        if cache_gather:
            await cache_gather

        if not prepared:
            return [results_map.get(i) for i in range(len(items))]

        # ── Phase 2 + 3: Overlap text CLIP encoding (GPU) with image downloading (network) ──
        all_chunks = []
        chunk_map = []
        for prep_idx, (idx, page_id, uhash, normalized, content, chunks, domain, tld_group, lang) in enumerate(prepared):
            for ci, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_map.append((prep_idx, ci))

        # Start text encoding (MPNet for Qdrant, CLIP for classification) and image downloading
        async def _encode_texts():
            if all_chunks and self._text_encoder:
                return await self._text_encoder.encode_texts_async(all_chunks)
            elif all_chunks:
                return await clip.encode_texts_async(all_chunks)
            return None

        async def _encode_texts_clip():
            """CLIP vectors needed for content classification + NSFW scoring."""
            if all_chunks and self._text_encoder:
                return await clip.encode_texts_async(all_chunks)
            return None  # If no text_encoder, _encode_texts already returns CLIP vectors

        async def _download_images():
            images = []
            img_map = []
            async with httpx.AsyncClient(
                follow_redirects=True,
                headers={"User-Agent": USER_AGENT},
                timeout=5,
            ) as client:
                fetch_tasks = []
                fetch_meta = []
                for prep_idx, (idx, page_id, uhash, normalized, content, chunks, domain, tld_group, lang) in enumerate(prepared):
                    if content.image_urls:
                        for img_url in content.image_urls[:5]:
                            fetch_tasks.append(load_image(img_url, client))
                            fetch_meta.append((prep_idx, img_url, content.image_alts.get(img_url)))

                if fetch_tasks:
                    results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
                    for (prep_idx, img_url, alt_text), result in zip(fetch_meta, results):
                        if isinstance(result, Exception) or result is None:
                            continue
                        images.append(result)
                        img_map.append((prep_idx, img_url, alt_text))
            return images, img_map

        text_task = asyncio.create_task(_encode_texts())
        clip_text_task = asyncio.create_task(_encode_texts_clip())
        image_dl_task = asyncio.create_task(_download_images())

        all_text_vectors, all_clip_vectors, (all_images, image_map) = await asyncio.gather(
            text_task, clip_text_task, image_dl_task
        )
        # Use CLIP vectors for classification (falls back to text_vectors if no separate CLIP encoding)
        classification_vectors = all_clip_vectors if all_clip_vectors is not None else all_text_vectors

        # Encode images on GPU (must wait for downloads)
        all_image_vectors = None
        if all_images:
            all_image_vectors = await clip.encode_images_async(all_images)

        # ── Phase 4: Batched DB writes in single transaction ──
        try:
            async with async_session() as write_session:
                write_conn = await write_session.connection()

                # Pre-compute classification + NSFW per page (uses CLIP vectors)
                text_vec_idx = 0
                page_meta = []  # (prep_idx, content_category, nsfw_flag, chunk_count)
                for prep_idx, (idx, page_id, uhash, normalized, content, chunks, domain, tld_group, lang) in enumerate(prepared):
                    page_text_vecs = []
                    for ci in range(len(chunks)):
                        page_text_vecs.append(classification_vectors[text_vec_idx + ci])
                    page_text_array = np.array(page_text_vecs) if page_text_vecs else np.array([])
                    content_category = classify_page(clip, page_text_array)
                    nsfw_flag = score_page_text_nsfw(clip, page_text_array)
                    page_meta.append((prep_idx, content_category, nsfw_flag, len(chunks)))
                    text_vec_idx += len(chunks)

                # Batch INSERT all text embeddings
                text_vec_idx = 0
                text_rows = []
                for prep_idx, (idx, page_id, uhash, normalized, content, chunks, domain, tld_group, lang) in enumerate(prepared):
                    for ci, chunk in enumerate(chunks):
                        vec = all_text_vectors[text_vec_idx]
                        text_vec_idx += 1
                        text_rows.append({
                            "page_id": page_id,
                            "chunk_index": ci,
                            "chunk_text": chunk,
                            "embedding": vec.tobytes(),
                        })
                if text_rows:
                    await write_conn.execute(TextEmbedding.__table__.insert(), text_rows)

                # Batch image dedup: single WHERE IN query for all image hashes
                if image_map:
                    all_img_hashes = [url_hash(img_url) for _, img_url, _ in image_map]
                    existing_result = await write_conn.execute(
                        select(ImageEmbedding.__table__.c.id, ImageEmbedding.__table__.c.image_url_hash)
                        .where(ImageEmbedding.__table__.c.image_url_hash.in_(all_img_hashes))
                    )
                    existing_map = {r.image_url_hash: r.id for r in existing_result.fetchall()}

                    # Split into new vs existing images
                    new_image_rows = []
                    image_nsfw_scores = {}  # img_hash -> nsfw_score
                    image_hash_list = []  # parallel to image_map
                    for img_idx, ((prep_idx, img_url, alt_text), img_hash) in enumerate(zip(image_map, all_img_hashes)):
                        vec = all_image_vectors[img_idx]
                        img_nsfw = score_image_nsfw(clip, vec)
                        img_icon = score_image_icon(clip, vec)
                        image_nsfw_scores[img_hash] = img_nsfw
                        image_hash_list.append(img_hash)
                        if img_hash not in existing_map:
                            new_image_rows.append({
                                "image_url": img_url,
                                "image_url_hash": img_hash,
                                "alt_text": alt_text,
                                "embedding": vec.tobytes(),
                                "nsfw_score": img_nsfw,
                                "icon_score": img_icon,
                            })

                    # Batch INSERT new images
                    if new_image_rows:
                        await write_conn.execute(ImageEmbedding.__table__.insert(), new_image_rows)
                        # Fetch IDs for newly inserted images
                        new_hashes = [r["image_url_hash"] for r in new_image_rows]
                        new_result = await write_conn.execute(
                            select(ImageEmbedding.__table__.c.id, ImageEmbedding.__table__.c.image_url_hash)
                            .where(ImageEmbedding.__table__.c.image_url_hash.in_(new_hashes))
                        )
                        for r in new_result.fetchall():
                            existing_map[r.image_url_hash] = r.id

                    # Batch INSERT image_page_sources
                    ips_rows = []
                    for img_idx, (prep_idx, img_url, alt_text) in enumerate(image_map):
                        img_hash = image_hash_list[img_idx]
                        image_emb_id = existing_map.get(img_hash)
                        if image_emb_id is None:
                            continue
                        _, page_id = prepared[prep_idx][0], prepared[prep_idx][1]
                        page_id = prepared[prep_idx][1]
                        ips_rows.append({"image_id": image_emb_id, "page_id": page_id})

                    if ips_rows:
                        # Use INSERT IGNORE for dedup
                        await write_conn.execute(text(
                            "INSERT IGNORE INTO image_page_sources (image_id, page_id) VALUES (:image_id, :page_id)"
                        ), ips_rows)

                    # Update page nsfw_flag if any image exceeds threshold
                    for img_idx, (prep_idx, img_url, alt_text) in enumerate(image_map):
                        img_hash = image_hash_list[img_idx]
                        if image_nsfw_scores.get(img_hash, 0) >= NSFW_IMAGE_THRESHOLD:
                            # Find the page_meta entry for this prep_idx and set nsfw
                            for pm_idx, (pm_prep, cat, nsfw, cc) in enumerate(page_meta):
                                if pm_prep == prep_idx:
                                    page_meta[pm_idx] = (pm_prep, cat, True, cc)
                                    break

                # Batch UPDATE all page statuses
                now = datetime.datetime.utcnow()
                for prep_idx, (idx, page_id, uhash, normalized, content, chunks, domain, tld_group, lang) in enumerate(prepared):
                    _, content_category, nsfw_flag, _ = page_meta[prep_idx]
                    await write_conn.execute(
                        Page.__table__.update().where(Page.__table__.c.id == page_id).values(
                            status="indexed", indexed_at=now,
                            domain=domain, tld_group=tld_group,
                            content_category=content_category,
                            nsfw_flag=nsfw_flag,
                        )
                    )

                await write_session.commit()

            # Dual-write to Qdrant (best-effort, after MySQL commit)
            # Fetch text_embedding IDs for Qdrant point IDs
            now = datetime.datetime.utcnow()
            qdrant_tasks = []
            for prep_idx, (idx, page_id, uhash, normalized, content, chunks, domain, tld_group, lang) in enumerate(prepared):
                _, content_category, nsfw_flag, _ = page_meta[prep_idx]
                last_mod = _parse_date(content.published_date) if content.published_date else None
                # Query text_embedding IDs for this page
                async with async_session() as id_session:
                    id_result = await id_session.execute(
                        select(TextEmbedding.__table__.c.id)
                        .where(TextEmbedding.__table__.c.page_id == page_id)
                        .order_by(TextEmbedding.__table__.c.chunk_index)
                    )
                    emb_ids = [r[0] for r in id_result.fetchall()]
                # Get text vectors for this page
                page_text_vecs = []
                vec_start = sum(len(prepared[pi][5]) for pi in range(prep_idx))
                for ci in range(len(chunks)):
                    page_text_vecs.append(all_text_vectors[vec_start + ci])
                qdrant_tasks.append(self._index_to_qdrant(
                    page_id, normalized, content.title, domain, tld_group,
                    content_category, lang, now, nsfw_flag, chunks,
                    emb_ids, page_text_vecs,
                    meta_description=content.meta_description, last_modified=last_mod,
                ))
            if qdrant_tasks:
                await asyncio.gather(*qdrant_tasks, return_exceptions=True)

            # Also write images to Qdrant
            if image_map and self._qdrant:
                qdrant_img_rows = []
                for img_idx, (prep_idx, img_url, alt_text) in enumerate(image_map):
                    img_hash = image_hash_list[img_idx]
                    image_emb_id = existing_map.get(img_hash)
                    if image_emb_id is None:
                        continue
                    page_id = prepared[prep_idx][1]
                    p_idx2, idx2, p_id2 = prep_idx, prepared[prep_idx][0], page_id
                    _, cat2, nsfw2, _ = page_meta[prep_idx]
                    _, _, _, _, _, _, domain2, tld2, lang2 = prepared[prep_idx]
                    qdrant_img_rows.append({
                        "id": image_emb_id,
                        "image_url": img_url,
                        "alt_text": alt_text,
                        "vector": all_image_vectors[img_idx],
                        "nsfw_score": image_nsfw_scores.get(img_hash, 0.0),
                        "icon_score": 0.0,
                        "page_ids": [page_id],
                        "domain": domain2,
                        "tld_group": tld2,
                        "content_category": cat2,
                        "language": lang2,
                    })
                if qdrant_img_rows:
                    await self._index_images_to_qdrant(qdrant_img_rows)

            # Log results and fetch final page objects
            for prep_idx, (idx, page_id, uhash, normalized, content, chunks, domain, tld_group, lang) in enumerate(prepared):
                _, content_category, nsfw_flag, _ = page_meta[prep_idx]
                n_images = sum(1 for pm, _, _ in image_map if pm == prep_idx)
                log.info("Indexed: %s (%d text chunks, %d images, cat=%s)", normalized, len(chunks), n_images, content_category)

            async with async_session() as read_session:
                page_ids = [p[1] for p in prepared]
                result = await read_session.execute(
                    select(Page).where(Page.id.in_(page_ids))
                )
                pages_by_id = {p.id: p for p in result.scalars().all()}
                for prep_idx, (idx, page_id, uhash, normalized, content, chunks, domain, tld_group, lang) in enumerate(prepared):
                    results_map[idx] = pages_by_id.get(page_id)

        except Exception as e:
            log.error("Batch write failed, falling back to per-page writes: %s", e)
            await self._fallback_per_page_write(
                prepared, all_text_vectors, classification_vectors, all_image_vectors, image_map, clip, results_map,
            )

        return [results_map.get(i) for i in range(len(items))]

    async def _fallback_per_page_write(
        self,
        prepared: list,
        all_text_vectors,
        classification_vectors,
        all_image_vectors,
        image_map: list,
        clip: CLIPService,
        results_map: dict,
    ):
        """Per-page fallback when batch write fails."""
        text_vec_idx = 0
        image_vec_idx = 0

        for prep_idx, (idx, page_id, uhash, normalized, content, chunks, domain, tld_group, lang) in enumerate(prepared):
            try:
                page_text_vecs = []
                page_class_vecs = []
                for ci in range(len(chunks)):
                    page_text_vecs.append(all_text_vectors[text_vec_idx + ci])
                    page_class_vecs.append(classification_vectors[text_vec_idx + ci])
                page_class_array = np.array(page_class_vecs) if page_class_vecs else np.array([])
                content_category = classify_page(clip, page_class_array)
                nsfw_flag = score_page_text_nsfw(clip, page_class_array)

                async with async_session() as write_session:
                    write_conn = await write_session.connection()

                    for ci, chunk in enumerate(chunks):
                        vec = all_text_vectors[text_vec_idx]
                        text_vec_idx += 1
                        await write_conn.execute(
                            TextEmbedding.__table__.insert().values(
                                page_id=page_id,
                                chunk_index=ci,
                                chunk_text=chunk,
                                embedding=vec.tobytes(),
                            )
                        )

                    n_images = 0
                    while image_vec_idx < len(image_map) and image_map[image_vec_idx][0] == prep_idx:
                        _, img_url, alt_text = image_map[image_vec_idx]
                        vec = all_image_vectors[image_vec_idx]
                        image_vec_idx += 1
                        img_hash = url_hash(img_url)

                        existing = await write_conn.execute(
                            select(ImageEmbedding.__table__.c.id).where(
                                ImageEmbedding.__table__.c.image_url_hash == img_hash
                            )
                        )
                        row = existing.first()
                        img_nsfw = score_image_nsfw(clip, vec)
                        img_icon = score_image_icon(clip, vec)
                        if row:
                            image_emb_id = row.id
                        else:
                            result = await write_conn.execute(
                                ImageEmbedding.__table__.insert().values(
                                    image_url=img_url,
                                    image_url_hash=img_hash,
                                    alt_text=alt_text,
                                    embedding=vec.tobytes(),
                                    nsfw_score=img_nsfw,
                                    icon_score=img_icon,
                                )
                            )
                            image_emb_id = result.lastrowid

                        if img_nsfw >= NSFW_IMAGE_THRESHOLD:
                            nsfw_flag = True

                        await write_conn.execute(text(
                            "INSERT IGNORE INTO image_page_sources (image_id, page_id) VALUES (:img_id, :page_id)"
                        ), {"img_id": image_emb_id, "page_id": page_id})
                        n_images += 1

                    now = datetime.datetime.utcnow()
                    await write_conn.execute(
                        Page.__table__.update().where(Page.__table__.c.id == page_id).values(
                            status="indexed", indexed_at=now,
                            domain=domain, tld_group=tld_group,
                            content_category=content_category,
                            nsfw_flag=nsfw_flag,
                        )
                    )
                    await write_session.commit()

                log.info("Indexed (fallback): %s (%d text chunks, %d images, cat=%s)", normalized, len(chunks), n_images, content_category)
                async with async_session() as read_session:
                    r = await read_session.execute(select(Page).where(Page.id == page_id))
                    results_map[idx] = r.scalar_one()

            except Exception as e:
                log.error("Fallback write failed for page %d: %s", page_id, e)
                try:
                    async with async_session() as err_session:
                        err_conn = await err_session.connection()
                        await err_conn.execute(
                            Page.__table__.update().where(Page.__table__.c.id == page_id).values(status="failed")
                        )
                        await err_session.commit()
                except Exception:
                    pass
                results_map[idx] = None
                while image_vec_idx < len(image_map) and image_map[image_vec_idx][0] == prep_idx:
                    image_vec_idx += 1

    async def index_url(self, url: str, session: AsyncSession, html: str | None = None) -> Page | None:
        normalized = normalize_url(url)
        uhash = url_hash(url)

        # Check if already indexed
        result = await session.execute(
            select(Page.id, Page.status).where(Page.url_hash == uhash)
        )
        row = result.first()
        if row and row.status == "indexed":
            log.info("Already indexed: %s", normalized)
            r = await session.execute(select(Page).where(Page.id == row.id))
            return r.scalar_one()

        # Fetch if no HTML provided
        if html is None:
            async with httpx.AsyncClient(
                follow_redirects=True,
                headers={"User-Agent": USER_AGENT},
            ) as client:
                try:
                    resp = await client.get(normalized, timeout=30)
                    if resp.status_code != 200:
                        log.warning("Failed to fetch %s: %d", normalized, resp.status_code)
                        return None
                    content_type = resp.headers.get("content-type", "")
                    if "text/html" not in content_type:
                        log.info("Skipping non-HTML: %s (%s)", normalized, content_type)
                        return None
                    html = resp.text
                except Exception as e:
                    log.error("Error fetching %s: %s", normalized, e)
                    return None

        # Get or create page_id
        conn = await session.connection()
        if row is None:
            await conn.execute(text(
                "INSERT IGNORE INTO pages (url, url_hash, status) VALUES (:url, :uhash, 'indexing')"
            ), {"url": normalized, "uhash": uhash})
            result = await session.execute(
                select(Page.id, Page.status).where(Page.url_hash == uhash)
            )
            row = result.first()
            if row is None:
                return None
            if row.status == "indexed":
                r = await session.execute(select(Page).where(Page.id == row.id))
                return r.scalar_one()

        page_id = row.id
        await conn.execute(
            Page.__table__.update().where(Page.__table__.c.id == page_id).values(status="indexing")
        )

        clip = self.clip
        domain = extract_domain(normalized)
        tld_group = extract_tld_group(normalized)

        try:
            content = extract_content(html, normalized)

            # SPA fallback
            if len(content.text) < MIN_TEXT_LENGTH and needs_rendering(html) and self._renderer:
                rendered_html = await self._renderer.render(normalized)
                if rendered_html:
                    content = extract_content(rendered_html, normalized)
                    html = rendered_html

            lang = _detect_language(content.text)

            last_mod = _parse_date(content.published_date) if content.published_date else None
            await conn.execute(
                Page.__table__.update().where(Page.__table__.c.id == page_id).values(
                    title=content.title[:1024] if content.title else None,
                    text_content=content.text,
                    meta_description=content.meta_description or None,
                    language=lang,
                    last_modified=last_mod,
                )
            )

            # Cache HTML
            cache_dir = settings.html_cache_dir / uhash[:2]
            cache_dir.mkdir(parents=True, exist_ok=True)
            (cache_dir / f"{uhash}.html").write_text(html, encoding="utf-8")

            # Clear old embeddings
            await conn.execute(
                TextEmbedding.__table__.delete().where(TextEmbedding.__table__.c.page_id == page_id)
            )
            await conn.execute(
                ImagePageSource.__table__.delete().where(ImagePageSource.__table__.c.page_id == page_id)
            )
            await session.commit()

            # Generate text embeddings (async)
            chunks = chunk_text(content.text)
            content_category = "other"
            nsfw_flag = False
            if chunks:
                # CLIP vectors needed for classification; MPNet vectors for Qdrant
                clip_vectors = await clip.encode_texts_async(chunks)
                content_category = classify_page(clip, clip_vectors)
                nsfw_flag = score_page_text_nsfw(clip, clip_vectors)
                if self._text_encoder:
                    text_vectors = await self._text_encoder.encode_texts_async(chunks)
                else:
                    text_vectors = clip_vectors
                async with async_session() as write_session:
                    write_conn = await write_session.connection()
                    for i, (chunk, vec) in enumerate(zip(chunks, text_vectors)):
                        await write_conn.execute(
                            TextEmbedding.__table__.insert().values(
                                page_id=page_id,
                                chunk_index=i,
                                chunk_text=chunk,
                                embedding=vec.tobytes(),
                            )
                        )
                    await write_session.commit()

            # Generate image embeddings (parallel fetch + async encode)
            n_images = 0
            if content.image_urls:
                async with httpx.AsyncClient(
                    follow_redirects=True,
                    headers={"User-Agent": USER_AGENT},
                    timeout=5,
                ) as client:
                    fetch_tasks = [load_image(img_url, client) for img_url in content.image_urls[:20]]
                    results = await asyncio.gather(*fetch_tasks, return_exceptions=True)
                    images = []
                    img_meta = []
                    for img_url, result in zip(content.image_urls[:20], results):
                        if isinstance(result, Exception) or result is None:
                            continue
                        images.append(result)
                        img_meta.append(img_url)
                    if images:
                        img_vectors = await clip.encode_images_async(images)
                        async with async_session() as write_session:
                            write_conn = await write_session.connection()
                            for img_url, vec in zip(img_meta, img_vectors):
                                img_hash = url_hash(img_url)
                                img_nsfw = score_image_nsfw(clip, vec)
                                img_icon = score_image_icon(clip, vec)
                                existing = await write_conn.execute(
                                    select(ImageEmbedding.__table__.c.id).where(
                                        ImageEmbedding.__table__.c.image_url_hash == img_hash
                                    )
                                )
                                row = existing.first()
                                if row:
                                    image_emb_id = row.id
                                else:
                                    result = await write_conn.execute(
                                        ImageEmbedding.__table__.insert().values(
                                            image_url=img_url,
                                            image_url_hash=img_hash,
                                            alt_text=content.image_alts.get(img_url),
                                            embedding=vec.tobytes(),
                                            nsfw_score=img_nsfw,
                                            icon_score=img_icon,
                                        )
                                    )
                                    image_emb_id = result.lastrowid
                                if img_nsfw >= NSFW_IMAGE_THRESHOLD:
                                    nsfw_flag = True
                                await write_conn.execute(text(
                                    "INSERT IGNORE INTO image_page_sources (image_id, page_id) VALUES (:img_id, :page_id)"
                                ), {"img_id": image_emb_id, "page_id": page_id})
                                n_images += 1
                            await write_session.commit()

            # Mark indexed
            async with async_session() as final_session:
                final_conn = await final_session.connection()
                now = datetime.datetime.utcnow()
                await final_conn.execute(
                    Page.__table__.update().where(Page.__table__.c.id == page_id).values(
                        status="indexed", indexed_at=now,
                        domain=domain, tld_group=tld_group,
                        content_category=content_category,
                        nsfw_flag=nsfw_flag,
                    )
                )
                await final_session.commit()

            log.info("Indexed: %s (%d text chunks, %d images, cat=%s)", normalized, len(chunks), n_images, content_category)

            # Dual-write to Qdrant
            if chunks and self._qdrant:
                last_mod = _parse_date(content.published_date) if content.published_date else None
                async with async_session() as id_session:
                    id_result = await id_session.execute(
                        select(TextEmbedding.__table__.c.id)
                        .where(TextEmbedding.__table__.c.page_id == page_id)
                        .order_by(TextEmbedding.__table__.c.chunk_index)
                    )
                    emb_ids = [r[0] for r in id_result.fetchall()]
                await self._index_to_qdrant(
                    page_id, normalized, content.title, domain, tld_group,
                    content_category, lang, now, nsfw_flag, chunks,
                    emb_ids, list(text_vectors),
                    meta_description=content.meta_description, last_modified=last_mod,
                )

            r = await session.execute(select(Page).where(Page.id == page_id))
            return r.scalar_one()

        except Exception as e:
            await session.rollback()
            try:
                async with async_session() as err_session:
                    err_conn = await err_session.connection()
                    await err_conn.execute(
                        Page.__table__.update().where(Page.__table__.c.id == page_id).values(status="failed")
                    )
                    await err_session.commit()
            except Exception:
                pass
            log.error("Indexing failed for %s: %s", normalized, e)
            raise
