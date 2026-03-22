import asyncio
import json
import logging

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import text as sql_text

from semantic_searcher.database import async_session
from semantic_searcher.models.schemas import (
    FiltersResponse,
    ImageSearchResponse,
    ImageSearchResultItem,
    SearchResponse,
    SearchResultItem,
)
from semantic_searcher.services.search_queue import SearchQueueFullError

log = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["search"])


def _log_search(query: str):
    """Fire-and-forget search logging. Works from both async and thread contexts."""
    async def _write():
        try:
            async with async_session() as session:
                await session.execute(
                    sql_text("INSERT INTO search_logs (query) VALUES (:q)"),
                    {"q": query[:2048]},
                )
                await session.commit()
        except Exception:
            pass
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_write())
    except RuntimeError:
        # Running in a thread pool — schedule on the main loop
        try:
            import threading
            loop = asyncio.get_event_loop_policy().get_event_loop()
            loop.call_soon_threadsafe(asyncio.ensure_future, _write())
        except Exception:
            pass


def _detect_lang_from_header(request: Request) -> str | None:
    """Parse Accept-Language header and return top 2-letter language code."""
    header = request.headers.get("accept-language")
    if not header:
        return None
    # Parse entries like "en-US,en;q=0.9,fr;q=0.8"
    best_lang = None
    best_q = -1.0
    for part in header.split(","):
        part = part.strip()
        if not part:
            continue
        if ";q=" in part:
            lang_part, q_part = part.split(";q=", 1)
            try:
                q = float(q_part.strip())
            except ValueError:
                q = 0.0
        else:
            lang_part = part
            q = 1.0
        lang_part = lang_part.strip()
        if q > best_q and lang_part != "*":
            best_q = q
            best_lang = lang_part
    if best_lang:
        # Return just the 2-letter code (e.g. "en" from "en-US")
        return best_lang[:2].lower()
    return None


class SearchRequest(BaseModel):
    q: str = Field(..., min_length=1)
    limit: int = Field(10, ge=1, le=100)
    offset: int = Field(0, ge=0)
    mode: str = Field("hybrid", pattern="^(hybrid|text|image)$")
    lang: str | None = Field(None, max_length=10)
    safe_search: bool = Field(True)
    domain: str | None = Field(None, max_length=253)
    tld_groups: list[str] | None = Field(None)
    categories: list[str] | None = Field(None)
    no_correct: bool = Field(False)
    title_must_contain: list[str] | None = Field(None)
    date_range_days: int | None = Field(None, ge=1, le=3650)
    edu_boost: bool = Field(False)
    log: bool = Field(False)  # Only log when user actually submits a search


class ImageSearchRequest(BaseModel):
    q: str = Field(..., min_length=1)
    limit: int = Field(20, ge=1, le=100)
    offset: int = Field(0, ge=0)
    lang: str | None = Field(None, max_length=10)
    safe_search: bool = Field(True)
    domain: str | None = Field(None, max_length=253)
    tld_groups: list[str] | None = Field(None)
    categories: list[str] | None = Field(None)
    icons_only: bool = Field(False)
    edu_boost: bool = Field(False)


def _do_search(request: Request, q: str, limit: int, offset: int, mode: str, lang: str | None, safe_search: bool, domain: str | None = None, tld_groups: list[str] | None = None, categories: list[str] | None = None, no_correct: bool = False, title_must_contain: list[str] | None = None, date_range_days: int | None = None, edu_boost: bool = False, should_log: bool = False):
    if should_log and offset == 0:
        _log_search(q)
    search_svc = request.app.state.search_service
    lang_hint = _detect_lang_from_header(request) if lang is None else None
    results, total, elapsed_ms, corrected_query, original_query = search_svc.search(q, limit=limit, offset=offset, mode=mode, lang=lang, safe_search=safe_search, domain=domain, tld_groups=tld_groups, categories=categories, no_correct=no_correct, lang_hint=lang_hint, title_must_contain=title_must_contain, date_range_days=date_range_days, edu_boost=edu_boost)
    return SearchResponse(
        query=q,
        results=[
            SearchResultItem(
                page_id=r.page_id,
                url=r.url,
                title=r.title,
                snippet=r.snippet,
                score=r.score,
                text_score=r.text_score,
                image_score=r.image_score,
                indexed_at=r.indexed_at,
                domain=r.domain,
                tld_group=r.tld_group,
                content_category=r.content_category,
            )
            for r in results
        ],
        total_results=total,
        search_time_ms=elapsed_ms,
        corrected_query=corrected_query,
        original_query=original_query,
    )


def _do_image_search(request: Request, q: str, limit: int, offset: int, lang: str | None, safe_search: bool, domain: str | None = None, tld_groups: list[str] | None = None, categories: list[str] | None = None, icons_only: bool = False, edu_boost: bool = False):
    search_svc = request.app.state.search_service
    results, total, elapsed_ms = search_svc.search_images(q, limit=limit, offset=offset, lang=lang, safe_search=safe_search, domain=domain, tld_groups=tld_groups, categories=categories, icons_only=icons_only)
    return ImageSearchResponse(
        query=q,
        results=[
            ImageSearchResultItem(
                image_url=r.image_url,
                alt_text=r.alt_text,
                score=r.score,
                page_id=r.page_id,
                page_url=r.page_url,
                page_title=r.page_title,
                found_on=r.found_on,
                possibly_explicit=r.possibly_explicit,
                domain=r.domain,
                tld_group=r.tld_group,
                content_category=r.content_category,
            )
            for r in results
        ],
        total_results=total,
        search_time_ms=elapsed_ms,
    )


@router.get("/search", response_model=SearchResponse)
async def search_get(
    request: Request,
    q: str = Query(..., min_length=1),
    limit: int = Query(10, ge=1, le=100),
    offset: int = Query(0, ge=0),
    mode: str = Query("hybrid", pattern="^(hybrid|text|image)$"),
    lang: str | None = Query(None, max_length=10),
    safe_search: bool = Query(True),
    domain: str | None = Query(None, max_length=253),
    tld_groups: list[str] | None = Query(None),
    categories: list[str] | None = Query(None),
    no_correct: bool = Query(False),
    title_must_contain: list[str] | None = Query(None),
):
    queue = request.app.state.search_queue
    try:
        return await queue.execute(_do_search, request, q, limit, offset, mode, lang, safe_search, domain, tld_groups, categories, no_correct=no_correct, title_must_contain=title_must_contain)
    except SearchQueueFullError as e:
        return JSONResponse(status_code=503, content={"error": "Server busy", "queued": e.queued, "active": e.active}, headers={"Retry-After": "2"})


@router.post("/search", response_model=SearchResponse)
async def search_post(request: Request, body: SearchRequest):
    queue = request.app.state.search_queue
    try:
        return await queue.execute(_do_search, request, body.q, body.limit, body.offset, body.mode, body.lang, body.safe_search, body.domain, body.tld_groups, body.categories, no_correct=body.no_correct, title_must_contain=body.title_must_contain, date_range_days=body.date_range_days, edu_boost=body.edu_boost, should_log=body.log)
    except SearchQueueFullError as e:
        return JSONResponse(status_code=503, content={"error": "Server busy", "queued": e.queued, "active": e.active}, headers={"Retry-After": "2"})


@router.get("/images/search", response_model=ImageSearchResponse)
async def search_images_get(
    request: Request,
    q: str = Query(..., min_length=1),
    limit: int = Query(20, ge=1, le=100),
    offset: int = Query(0, ge=0),
    lang: str | None = Query(None, max_length=10),
    safe_search: bool = Query(True),
    domain: str | None = Query(None, max_length=253),
    tld_groups: list[str] | None = Query(None),
    categories: list[str] | None = Query(None),
    icons_only: bool = Query(False),
):
    queue = request.app.state.search_queue
    try:
        return await queue.execute(_do_image_search, request, q, limit, offset, lang, safe_search, domain, tld_groups, categories, icons_only=icons_only)
    except SearchQueueFullError as e:
        return JSONResponse(status_code=503, content={"error": "Server busy", "queued": e.queued, "active": e.active}, headers={"Retry-After": "2"})


@router.post("/images/search", response_model=ImageSearchResponse)
async def search_images_post(request: Request, body: ImageSearchRequest):
    queue = request.app.state.search_queue
    try:
        return await queue.execute(_do_image_search, request, body.q, body.limit, body.offset, body.lang, body.safe_search, body.domain, body.tld_groups, body.categories, icons_only=body.icons_only, edu_boost=body.edu_boost)
    except SearchQueueFullError as e:
        return JSONResponse(status_code=503, content={"error": "Server busy", "queued": e.queued, "active": e.active}, headers={"Retry-After": "2"})


def _sse_event(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data, ensure_ascii=False)}\n\n"


@router.post("/search/stream")
async def search_stream(request: Request, body: SearchRequest):
    if body.log and body.offset == 0:
        _log_search(body.q)
    search_svc = request.app.state.search_service
    queue = request.app.state.search_queue
    lang_hint = _detect_lang_from_header(request) if body.lang is None else None

    def generate():
        for event, data in search_svc.search_stream(
            body.q, limit=body.limit, offset=body.offset, mode=body.mode,
            lang=body.lang, safe_search=body.safe_search, domain=body.domain,
            tld_groups=body.tld_groups, categories=body.categories,
            no_correct=body.no_correct, lang_hint=lang_hint,
            title_must_contain=body.title_must_contain,
            date_range_days=body.date_range_days,
            edu_boost=body.edu_boost,
        ):
            yield (event, data)

    async def stream():
        try:
            async for event, data in queue.execute_streaming(generate):
                yield _sse_event(event, data)
        except SearchQueueFullError as e:
            yield _sse_event("error", {"message": f"Server busy ({e.queued} queued)"})
        except Exception as exc:
            log.exception("SSE search error")
            yield _sse_event("error", {"message": str(exc)})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.post("/images/search/stream")
async def search_images_stream(request: Request, body: ImageSearchRequest):
    search_svc = request.app.state.search_service
    queue = request.app.state.search_queue

    def generate():
        for event, data in search_svc.search_images_stream(
            body.q, limit=body.limit, offset=body.offset,
            lang=body.lang, safe_search=body.safe_search, domain=body.domain,
            tld_groups=body.tld_groups, categories=body.categories,
            icons_only=body.icons_only,
        ):
            yield (event, data)

    try:
        events = await queue.execute_generator(generate)
    except SearchQueueFullError as e:
        return JSONResponse(status_code=503, content={"error": "Server busy", "queued": e.queued, "active": e.active}, headers={"Retry-After": "2"})

    def stream():
        try:
            for event, data in events:
                yield _sse_event(event, data)
        except Exception as exc:
            log.exception("SSE image search error")
            yield _sse_event("error", {"message": str(exc)})

    return StreamingResponse(
        stream(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/filters", response_model=FiltersResponse)
async def get_filters(request: Request):
    search_svc = request.app.state.search_service
    return FiltersResponse(**search_svc.available_filters())


@router.get("/search/queue")
async def search_queue_status(request: Request):
    queue = request.app.state.search_queue
    return queue.stats


async def _get_suggestions(q: str, limit: int = 5) -> list[str]:
    """Shared suggestion lookup from search_logs."""
    if len(q) < 2:
        return []
    try:
        async with async_session() as session:
            result = await session.execute(
                sql_text(
                    "SELECT query, COUNT(*) as cnt FROM search_logs "
                    "WHERE query LIKE :prefix GROUP BY query "
                    "ORDER BY cnt DESC LIMIT :limit"
                ),
                {"prefix": q + "%", "limit": limit},
            )
            return [row[0] for row in result.fetchall()]
    except Exception:
        log.exception("Autocomplete error")
        return []


@router.get("/autocomplete")
async def autocomplete(
    q: str = Query("", min_length=0),
    limit: int = Query(5, ge=1, le=50),
):
    return {"suggestions": await _get_suggestions(q, limit)}


@router.get("/suggest")
async def suggest(
    q: str = Query("", min_length=0),
):
    """OpenSearch Suggestions format for browser address bar integration.
    Returns: [query, [suggestions...]]
    """
    suggestions = await _get_suggestions(q, 8)
    from fastapi.responses import JSONResponse
    return JSONResponse(content=[q, suggestions])


@router.get("/related")
async def related(
    q: str = Query("", min_length=0),
    limit: int = Query(5, ge=1, le=50),
):
    if not q.strip():
        return {"related": []}
    try:
        words = q.strip().split()
        merged: dict[str, int] = {}
        async with async_session() as session:
            for word in words:
                result = await session.execute(
                    sql_text(
                        "SELECT query, COUNT(*) as cnt FROM search_logs "
                        "WHERE query LIKE :pattern AND query != :q "
                        "GROUP BY query ORDER BY cnt DESC LIMIT :limit"
                    ),
                    {"pattern": "%" + word + "%", "q": q, "limit": limit},
                )
                for row in result.fetchall():
                    merged[row[0]] = merged.get(row[0], 0) + row[1]
        sorted_results = sorted(merged.items(), key=lambda x: x[1], reverse=True)
        return {"related": [r[0] for r in sorted_results[:limit]]}
    except Exception:
        log.exception("Related searches error")
        return {"related": []}


# --- Click tracking ---

class ClickRequest(BaseModel):
    query: str = Field(..., max_length=2048)
    url: str = Field(..., max_length=2048)
    position: int = Field(..., ge=0)


class BounceRequest(BaseModel):
    query: str = Field(..., max_length=2048)
    url: str = Field(..., max_length=2048)
    dwell_time_ms: int = Field(..., ge=0)


def _log_click(query: str, url: str, position: int):
    """Fire-and-forget click logging."""
    async def _write():
        try:
            async with async_session() as session:
                await session.execute(
                    sql_text(
                        "INSERT INTO click_logs (query, url, position) "
                        "VALUES (:q, :url, :pos)"
                    ),
                    {"q": query[:2048], "url": url[:2048], "pos": position},
                )
                await session.commit()
        except Exception:
            pass
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_write())
    except RuntimeError:
        pass


@router.post("/click")
async def log_click(body: ClickRequest):
    _log_click(body.query, body.url, body.position)
    return {"ok": True}


@router.post("/bounce")
async def log_bounce(body: BounceRequest):
    """Update the most recent click log for this query+url with dwell time and bounce flag."""
    bounce_threshold_ms = 5000  # under 5 seconds = bounce
    async def _update():
        try:
            async with async_session() as session:
                await session.execute(
                    sql_text(
                        "UPDATE click_logs SET dwell_time_ms = :dwell, bounced = :bounced "
                        "WHERE id = ("
                        "  SELECT id FROM (SELECT id FROM click_logs "
                        "  WHERE query = :q AND url = :url "
                        "  ORDER BY clicked_at DESC LIMIT 1) AS t"
                        ")"
                    ),
                    {
                        "dwell": body.dwell_time_ms,
                        "bounced": body.dwell_time_ms < bounce_threshold_ms,
                        "q": body.query[:2048],
                        "url": body.url[:2048],
                    },
                )
                await session.commit()
        except Exception:
            pass
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(_update())
    except RuntimeError:
        pass
    return {"ok": True}


@router.get("/image/sources")
async def image_sources(
    url: str = Query(..., max_length=2048),
    page_ids: str = Query("", description="Comma-separated page IDs from visual dedup"),
):
    """Return all pages where this image URL was found, plus image metadata."""
    try:
        async with async_session() as session:
            img_result = await session.execute(
                sql_text(
                    "SELECT ie.id, ie.alt_text, ie.nsfw_score, ie.icon_score "
                    "FROM image_embeddings ie "
                    "WHERE ie.image_url_hash = SHA2(:url, 256) "
                    "LIMIT 1"
                ),
                {"url": url},
            )
            img_row = img_result.fetchone()
            alt_text = img_row[1] if img_row else None
            nsfw_score = img_row[2] if img_row else None
            icon_score = img_row[3] if img_row else None

            # Use page_ids from Qdrant visual dedup if provided, else fall back to MySQL
            pid_list = []
            if page_ids:
                try:
                    pid_list = [int(x) for x in page_ids.split(",") if x.strip()]
                except ValueError:
                    pid_list = []

            if pid_list:
                placeholders = ",".join(str(p) for p in pid_list[:500])
                pages_result = await session.execute(
                    sql_text(
                        f"SELECT p.url, p.title, p.domain, p.indexed_at "
                        f"FROM pages p "
                        f"WHERE p.id IN ({placeholders}) AND p.status = 'indexed' "
                        f"ORDER BY p.indexed_at DESC"
                    ),
                )
            elif img_row:
                pages_result = await session.execute(
                    sql_text(
                        "SELECT p.url, p.title, p.domain, p.indexed_at "
                        "FROM image_page_sources ips "
                        "JOIN pages p ON p.id = ips.page_id "
                        "WHERE ips.image_id = :img_id "
                        "ORDER BY p.indexed_at DESC"
                    ),
                    {"img_id": img_row[0]},
                )
            else:
                return {"image_url": url, "alt_text": None, "metadata": {}, "sources": []}

            sources = [
                {
                    "url": row[0],
                    "title": row[1] or "Untitled",
                    "domain": row[2],
                    "indexed_at": row[3].isoformat() if row[3] else None,
                }
                for row in pages_result.fetchall()
            ]

            return {
                "image_url": url,
                "alt_text": alt_text,
                "metadata": {
                    "nsfw_score": round(nsfw_score, 3) if nsfw_score is not None else None,
                    "icon_score": round(icon_score, 3) if icon_score is not None else None,
                    "found_on_pages": len(sources),
                },
                "sources": sources,
            }
    except Exception:
        log.exception("Image sources error")
        return {"image_url": url, "alt_text": None, "metadata": {}, "sources": []}


# --- Feedback ---

class FeedbackRequest(BaseModel):
    rating: int = Field(..., ge=1, le=5)
    category: str | None = Field(None, max_length=50)
    message: str | None = Field(None, max_length=5000)
    page: str | None = Field(None, max_length=200)


@router.post("/feedback")
async def submit_feedback(body: FeedbackRequest, request: Request):
    try:
        ua = (request.headers.get("user-agent") or "")[:512]
        async with async_session() as session:
            await session.execute(
                sql_text(
                    "INSERT INTO feedback (rating, category, message, page, user_agent) "
                    "VALUES (:rating, :cat, :msg, :page, :ua)"
                ),
                {
                    "rating": body.rating,
                    "cat": body.category,
                    "msg": body.message,
                    "page": body.page,
                    "ua": ua,
                },
            )
            await session.commit()
    except Exception:
        log.exception("Feedback save error")
        return JSONResponse(status_code=500, content={"error": "Failed to save feedback"})
    return {"ok": True}


@router.get("/feedback/summary")
async def feedback_summary():
    """Return aggregate feedback stats."""
    try:
        async with async_session() as session:
            result = await session.execute(
                sql_text(
                    "SELECT rating, COUNT(*) as cnt FROM feedback "
                    "GROUP BY rating ORDER BY rating"
                )
            )
            ratings = {row[0]: row[1] for row in result.fetchall()}
            total = sum(ratings.values())
            avg = sum(r * c for r, c in ratings.items()) / total if total else 0
            return {"total": total, "average": round(avg, 2), "ratings": ratings}
    except Exception:
        return {"total": 0, "average": 0, "ratings": {}}
