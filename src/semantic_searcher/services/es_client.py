"""Elasticsearch client management and index schema."""

import logging
from elasticsearch import Elasticsearch, AsyncElasticsearch
from semantic_searcher.config import settings

log = logging.getLogger(__name__)

_INDEX_SETTINGS = {
    "number_of_shards": 1,
    "number_of_replicas": 0,
}

_INDEX_MAPPINGS = {
    "properties": {
        "page_id": {"type": "long"},
        "url": {"type": "keyword"},
        "title": {
            "type": "text",
            "analyzer": "standard",
            "fields": {"raw": {"type": "keyword"}},
        },
        "domain": {"type": "keyword"},
        "tld_group": {"type": "keyword"},
        "content_category": {"type": "keyword"},
        "language": {"type": "keyword"},
        "indexed_at": {"type": "date"},
        "nsfw_flag": {"type": "boolean"},
        "chunks": {"type": "text", "analyzer": "standard"},
    },
}


def get_sync_es() -> Elasticsearch:
    return Elasticsearch(f"http://{settings.es_host}:{settings.es_port}")


def get_async_es() -> AsyncElasticsearch:
    return AsyncElasticsearch(f"http://{settings.es_host}:{settings.es_port}")


async def ensure_index(es: AsyncElasticsearch | None = None):
    """Create the pages index if it does not exist."""
    own = es is None
    if own:
        es = get_async_es()
    try:
        if not await es.indices.exists(index=settings.es_index):
            await es.indices.create(
                index=settings.es_index,
                settings=_INDEX_SETTINGS,
                mappings=_INDEX_MAPPINGS,
            )
            log.info("Created ES index '%s'", settings.es_index)
        else:
            log.info("ES index '%s' already exists", settings.es_index)
    finally:
        if own:
            await es.close()
