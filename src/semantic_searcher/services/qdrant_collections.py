"""Qdrant collection management and schema definitions."""

import logging
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    HnswConfigDiff,
    MatchAny,
    MatchText,
    MatchValue,
    Modifier,
    NamedVector,
    PayloadSchemaType,
    PointStruct,
    ScalarQuantization,
    ScalarQuantizationConfig,
    ScalarType,
    SearchParams,
    SparseIndexParams,
    SparseVectorParams,
    VectorParams,
)

log = logging.getLogger(__name__)

TEXT_CHUNKS_COLLECTION = "text_chunks"
TEXT_CHUNKS_V2_COLLECTION = "text_chunks_v2"
IMAGES_COLLECTION = "images"

VECTOR_DIM = 512
VECTOR_DIM_V2 = 768  # MPNet-base-v2


def ensure_collections(client: QdrantClient):
    """Create Qdrant collections if they don't exist."""
    _ensure_text_chunks_v2(client)
    _ensure_images(client)


def _ensure_text_chunks(client: QdrantClient):
    if client.collection_exists(TEXT_CHUNKS_COLLECTION):
        log.info("Qdrant collection '%s' already exists", TEXT_CHUNKS_COLLECTION)
        return

    client.create_collection(
        collection_name=TEXT_CHUNKS_COLLECTION,
        vectors_config={
            "dense": VectorParams(
                size=VECTOR_DIM,
                distance=Distance.DOT,
                on_disk=True,
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    ),
                ),
            ),
        },
        sparse_vectors_config={
            "text-bm25": SparseVectorParams(
                modifier=Modifier.IDF,
                index=SparseIndexParams(on_disk=True),
            ),
        },
        hnsw_config=HnswConfigDiff(on_disk=False),
        on_disk_payload=True,
    )
    log.info("Created Qdrant collection '%s'", TEXT_CHUNKS_COLLECTION)

    # Create payload indexes for filtering
    for field, schema in [
        ("page_id", PayloadSchemaType.INTEGER),
        ("domain", PayloadSchemaType.KEYWORD),
        ("tld_group", PayloadSchemaType.KEYWORD),
        ("content_category", PayloadSchemaType.KEYWORD),
        ("language", PayloadSchemaType.KEYWORD),
        ("nsfw_flag", PayloadSchemaType.BOOL),
        ("is_stale", PayloadSchemaType.BOOL),
        ("mysql_emb_id", PayloadSchemaType.INTEGER),
    ]:
        client.create_payload_index(TEXT_CHUNKS_COLLECTION, field, field_schema=schema)
    log.info("Created payload indexes for '%s'", TEXT_CHUNKS_COLLECTION)


def _ensure_text_chunks_v2(client: QdrantClient):
    if client.collection_exists(TEXT_CHUNKS_V2_COLLECTION):
        log.info("Qdrant collection '%s' already exists", TEXT_CHUNKS_V2_COLLECTION)
        return

    client.create_collection(
        collection_name=TEXT_CHUNKS_V2_COLLECTION,
        vectors_config={
            "dense": VectorParams(
                size=VECTOR_DIM_V2,
                distance=Distance.COSINE,
                on_disk=True,
                quantization_config=ScalarQuantization(
                    scalar=ScalarQuantizationConfig(
                        type=ScalarType.INT8,
                        quantile=0.99,
                        always_ram=True,
                    ),
                ),
            ),
        },
        sparse_vectors_config={
            "text-bm25": SparseVectorParams(
                modifier=Modifier.IDF,
                index=SparseIndexParams(on_disk=True),
            ),
        },
        hnsw_config=HnswConfigDiff(on_disk=False),
        on_disk_payload=True,
    )
    log.info("Created Qdrant collection '%s'", TEXT_CHUNKS_V2_COLLECTION)

    for field, schema in [
        ("page_id", PayloadSchemaType.INTEGER),
        ("domain", PayloadSchemaType.KEYWORD),
        ("tld_group", PayloadSchemaType.KEYWORD),
        ("content_category", PayloadSchemaType.KEYWORD),
        ("language", PayloadSchemaType.KEYWORD),
        ("nsfw_flag", PayloadSchemaType.BOOL),
        ("is_stale", PayloadSchemaType.BOOL),
        ("mysql_emb_id", PayloadSchemaType.INTEGER),
    ]:
        client.create_payload_index(TEXT_CHUNKS_V2_COLLECTION, field, field_schema=schema)
    log.info("Created payload indexes for '%s'", TEXT_CHUNKS_V2_COLLECTION)


def _ensure_images(client: QdrantClient):
    if client.collection_exists(IMAGES_COLLECTION):
        log.info("Qdrant collection '%s' already exists", IMAGES_COLLECTION)
        return

    client.create_collection(
        collection_name=IMAGES_COLLECTION,
        vectors_config=VectorParams(
            size=VECTOR_DIM,
            distance=Distance.DOT,
        ),
        hnsw_config=HnswConfigDiff(on_disk=True),
    )
    log.info("Created Qdrant collection '%s'", IMAGES_COLLECTION)

    for field, schema in [
        ("nsfw_score", PayloadSchemaType.FLOAT),
        ("icon_score", PayloadSchemaType.FLOAT),
        ("domain", PayloadSchemaType.KEYWORD),
        ("tld_group", PayloadSchemaType.KEYWORD),
        ("content_category", PayloadSchemaType.KEYWORD),
        ("mysql_img_id", PayloadSchemaType.INTEGER),
    ]:
        client.create_payload_index(IMAGES_COLLECTION, field, field_schema=schema)
    log.info("Created payload indexes for '%s'", IMAGES_COLLECTION)


STALE_THRESHOLD_DAYS = 365  # Hard retrieval cutoff; freshness decay handles age penalization


def build_search_filter(
    lang: str | None = None,
    safe_search: bool = True,
    domain: str | None = None,
    tld_groups: list[str] | None = None,
    categories: list[str] | None = None,
    include_stale: bool = False,
) -> Filter | None:
    """Build a Qdrant filter from search parameters.

    By default, stale pages (>90 days old) are excluded.
    Set include_stale=True to include them (e.g. for date-range queries).
    """
    must = []
    must_not = []
    if lang:
        must.append(FieldCondition(key="language", match=MatchValue(value=lang)))
    if domain:
        must.append(FieldCondition(key="domain", match=MatchValue(value=domain)))
    if safe_search:
        must.append(FieldCondition(key="nsfw_flag", match=MatchValue(value=False)))
    if not include_stale:
        # Exclude stale pages; null values (untagged) pass through
        must_not.append(FieldCondition(key="is_stale", match=MatchValue(value=True)))
    if tld_groups:
        must.append(FieldCondition(key="tld_group", match=MatchAny(any=tld_groups)))
    if categories:
        must.append(FieldCondition(key="content_category", match=MatchAny(any=categories)))
    if must or must_not:
        return Filter(must=must or None, must_not=must_not or None)
    return None
