import time

import torch
from fastapi import APIRouter, Depends, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from semantic_searcher.config import settings
from semantic_searcher.database import get_session
from semantic_searcher.models.schemas import (
    HealthResponse, StatsResponse, PageResponse, PageListResponse,
    TokenizeRequest, TokenizeResponse, DetokenizeRequest, DetokenizeResponse,
    TrainTokenizerRequest,
)
from semantic_searcher.models.db import Page
from sqlalchemy import select, func

router = APIRouter(prefix="/api", tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health(request: Request, session: AsyncSession = Depends(get_session)):
    # Check DB
    db_status = "ok"
    try:
        await session.execute(text("SELECT 1"))
    except Exception:
        db_status = "error"

    clip_status = "loaded" if request.app.state.clip_services[0].model is not None else "not loaded"
    gpu = torch.cuda.is_available()

    overall = "healthy" if db_status == "ok" and clip_status == "loaded" else "degraded"
    return HealthResponse(status=overall, database=db_status, clip_model=clip_status, gpu_available=gpu)


_stats_cache: dict = {"count": 0, "rate": 0.0, "ts": 0.0}
_STATS_CACHE_TTL = 30  # seconds


@router.get("/stats", response_model=StatsResponse)
async def stats(request: Request, session: AsyncSession = Depends(get_session)):
    s = request.app.state.search_service.stats
    now = time.monotonic()
    # Refresh DB count + crawl rate at most once per TTL — shared across all clients
    if now - _stats_cache["ts"] > _STATS_CACHE_TTL:
        try:
            result = await session.execute(
                select(func.count(Page.id)).where(Page.status == "indexed")
            )
            _stats_cache["count"] = result.scalar() or 0
            rate_result = await session.execute(text(
                "SELECT COUNT(*) FROM pages"
                " WHERE status='indexed' AND indexed_at > NOW() - INTERVAL 1 HOUR"
            ))
            _stats_cache["rate"] = float(rate_result.scalar() or 0)
        except Exception:
            _stats_cache["count"] = s["pages"]
        _stats_cache["ts"] = now
    return StatsResponse(
        pages_indexed=max(_stats_cache["count"], s["pages"]),
        text_embeddings=s["text_embeddings"],
        image_embeddings=s["image_embeddings"],
        pages_per_hour=_stats_cache["rate"],
    )


@router.get("/pages", response_model=PageListResponse)
async def list_pages(
    offset: int = 0,
    limit: int = 50,
    session: AsyncSession = Depends(get_session),
):
    total_result = await session.execute(select(func.count(Page.id)))
    total = total_result.scalar()
    result = await session.execute(
        select(Page).order_by(Page.id.desc()).offset(offset).limit(limit)
    )
    pages = result.scalars().all()
    return PageListResponse(
        pages=[
            PageResponse(
                id=p.id, url=p.url, title=p.title, meta_description=p.meta_description,
                status=p.status,
                indexed_at=p.indexed_at.isoformat() if p.indexed_at else None,
                created_at=p.created_at.isoformat() if p.created_at else None,
            )
            for p in pages
        ],
        total=total, offset=offset, limit=limit,
    )


@router.get("/pages/{page_id}", response_model=PageResponse)
async def get_page(page_id: int, session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(Page).where(Page.id == page_id))
    page = result.scalar_one_or_none()
    if not page:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Page not found")
    return PageResponse(
        id=page.id, url=page.url, title=page.title, meta_description=page.meta_description,
        status=page.status,
        indexed_at=page.indexed_at.isoformat() if page.indexed_at else None,
        created_at=page.created_at.isoformat() if page.created_at else None,
    )


@router.delete("/pages/{page_id}")
async def delete_page(page_id: int, request: Request, session: AsyncSession = Depends(get_session)):
    result = await session.execute(select(Page).where(Page.id == page_id))
    page = result.scalar_one_or_none()
    if not page:
        from fastapi import HTTPException
        raise HTTPException(status_code=404, detail="Page not found")
    await session.delete(page)
    await session.commit()
    # Refresh search index
    search_svc = request.app.state.search_service
    from semantic_searcher.database import async_session
    async with async_session() as fresh:
        await search_svc.load_index(fresh)
    return {"message": "Page deleted", "page_id": page_id}


# --- Tokenizer endpoints ---
@router.post("/tokenizer/encode", response_model=TokenizeResponse)
async def tokenize(body: TokenizeRequest, request: Request):
    tok = request.app.state.tokenizer_service
    if not tok.is_trained():
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Tokenizer not trained yet")
    ids = tok.encode(body.text)
    return TokenizeResponse(ids=ids, tokens_count=len(ids))


@router.post("/tokenizer/decode", response_model=DetokenizeResponse)
async def detokenize(body: DetokenizeRequest, request: Request):
    tok = request.app.state.tokenizer_service
    if not tok.is_trained():
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="Tokenizer not trained yet")
    text = tok.decode(body.ids)
    return DetokenizeResponse(text=text)


@router.post("/tokenizer/train")
async def train_tokenizer(
    body: TrainTokenizerRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    tok = request.app.state.tokenizer_service
    # Gather corpus from indexed pages
    result = await session.execute(
        select(Page.text_content).where(Page.status == "indexed", Page.text_content.is_not(None))
    )
    texts = [row[0] for row in result.all() if row[0]]
    if not texts:
        from fastapi import HTTPException
        raise HTTPException(status_code=400, detail="No indexed pages to train on")
    tok.train(texts, vocab_size=body.vocab_size)
    return {"message": "Tokenizer trained", "vocab_size": tok.vocab_size}
