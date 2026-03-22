import asyncio

from fastapi import APIRouter, HTTPException, Request

from semantic_searcher.config import settings
from semantic_searcher.models.schemas import (
    CrawlAutoRequest,
    CrawlAutoResponse,
    CrawlStartRequest,
    CrawlStatusResponse,
    CTWatcherStatusResponse,
    HarvesterStatusResponse,
    SubdomainEnumStatusResponse,
)
from semantic_searcher.utils.commoncrawl import CommonCrawlSeeder

router = APIRouter(prefix="/api/crawl", tags=["crawl"])

_seeder = CommonCrawlSeeder()


@router.post("/start")
async def start_crawl(body: CrawlStartRequest, request: Request):
    crawler = request.app.state.crawler_service
    await crawler.start(
        seed_urls=body.seed_urls,
        max_depth=body.max_depth,
        max_pages=body.max_pages,
    )
    return {"message": "Crawler started", "seed_urls": body.seed_urls}


@router.post("/stop")
async def stop_crawl(request: Request):
    crawler = request.app.state.crawler_service
    await crawler.stop()
    return {"message": "Crawler stopped"}


@router.get("/status", response_model=CrawlStatusResponse)
async def crawl_status(request: Request):
    crawler = request.app.state.crawler_service
    status = await crawler.status()
    return CrawlStatusResponse(**status)


@router.post("/auto", response_model=CrawlAutoResponse)
async def auto_crawl(body: CrawlAutoRequest, request: Request):
    """Bootstrap crawling from Common Crawl web graph domains."""
    crawler = request.app.state.crawler_service
    if crawler.is_running:
        raise HTTPException(status_code=409, detail="Crawler already running")

    # Download + sample in a thread (blocking I/O)
    domains = await asyncio.to_thread(
        _seeder.sample, body.sample_size
    )
    seed_urls = CommonCrawlSeeder.domains_to_urls(domains)

    await crawler.start(
        seed_urls=seed_urls,
        max_depth=body.max_depth,
        max_pages=body.max_pages,
    )
    return CrawlAutoResponse(
        message=f"Auto-crawl started with {len(seed_urls)} seed URLs from Common Crawl",
        sample_size=len(seed_urls),
        seed_urls=seed_urls,
    )


@router.post("/ct/start")
async def start_ct_watcher(request: Request):
    """Start the Certificate Transparency log watcher."""
    ct_watcher = request.app.state.ct_watcher
    if ct_watcher.is_running:
        raise HTTPException(status_code=409, detail="CT watcher already running")
    ct_watcher.start()
    return {"message": "CT watcher started"}


@router.post("/ct/stop")
async def stop_ct_watcher(request: Request):
    """Stop the Certificate Transparency log watcher."""
    ct_watcher = request.app.state.ct_watcher
    if not ct_watcher.is_running:
        raise HTTPException(status_code=409, detail="CT watcher not running")
    await ct_watcher.stop()
    return {"message": "CT watcher stopped"}


@router.get("/ct/status", response_model=CTWatcherStatusResponse)
async def ct_watcher_status(request: Request):
    """Get CT watcher statistics."""
    ct_watcher = request.app.state.ct_watcher
    return CTWatcherStatusResponse(**ct_watcher.stats)


@router.post("/harvester/start")
async def start_harvester(request: Request):
    """Start the link harvester."""
    harvester = request.app.state.link_harvester
    if harvester.is_running:
        raise HTTPException(status_code=409, detail="Link harvester already running")
    harvester.start()
    return {"message": "Link harvester started"}


@router.post("/harvester/stop")
async def stop_harvester(request: Request):
    """Stop the link harvester."""
    harvester = request.app.state.link_harvester
    if not harvester.is_running:
        raise HTTPException(status_code=409, detail="Link harvester not running")
    await harvester.stop()
    return {"message": "Link harvester stopped"}


@router.get("/harvester/status", response_model=HarvesterStatusResponse)
async def harvester_status(request: Request):
    """Get link harvester statistics."""
    harvester = request.app.state.link_harvester
    return HarvesterStatusResponse(**harvester.stats)


@router.post("/subdomain-enum/start")
async def start_subdomain_enum(request: Request):
    """Start the subdomain enumerator."""
    enum = request.app.state.subdomain_enum
    if enum.is_running:
        raise HTTPException(status_code=409, detail="Subdomain enumerator already running")
    enum.start()
    return {"message": "Subdomain enumerator started"}


@router.post("/subdomain-enum/stop")
async def stop_subdomain_enum(request: Request):
    """Stop the subdomain enumerator."""
    enum = request.app.state.subdomain_enum
    if not enum.is_running:
        raise HTTPException(status_code=409, detail="Subdomain enumerator not running")
    await enum.stop()
    return {"message": "Subdomain enumerator stopped"}


@router.get("/subdomain-enum/status", response_model=SubdomainEnumStatusResponse)
async def subdomain_enum_status(request: Request):
    """Get subdomain enumerator statistics."""
    enum = request.app.state.subdomain_enum
    return SubdomainEnumStatusResponse(**enum.stats)
