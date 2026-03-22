from fastapi import APIRouter, Depends, Request
from sqlalchemy.ext.asyncio import AsyncSession

from semantic_searcher.database import get_session, async_session
from semantic_searcher.models.schemas import IndexRequest, IndexResponse

router = APIRouter(prefix="/api", tags=["index"])


@router.post("/index", response_model=IndexResponse)
async def index_url(
    body: IndexRequest,
    request: Request,
    session: AsyncSession = Depends(get_session),
):
    indexer = request.app.state.indexer_service
    search_svc = request.app.state.search_service
    try:
        page = await indexer.index_url(body.url, session)
        if page:
            async with async_session() as fresh:
                await search_svc.load_index(fresh)
            return IndexResponse(
                page_id=page.id, url=page.url, status=page.status, message="Indexed successfully"
            )
        return IndexResponse(url=body.url, status="failed", message="Could not fetch or index URL")
    except Exception as e:
        return IndexResponse(url=body.url, status="failed", message=str(e))
