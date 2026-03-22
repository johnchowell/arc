#!/usr/bin/env python3
"""Migrate vectors from central Qdrant to distributed worker shards.

Reads vectors from the hub's Qdrant text_chunks_v2 collection, sends them to
the appropriate worker's shard via the worker API.

Usage:
    PYTHONPATH=src python scripts/migrate_to_shards.py [--batch-size 500] [--dry-run]
"""

import argparse
import asyncio
import logging
import time

import httpx
from qdrant_client import QdrantClient
from sqlalchemy import select

from semantic_searcher.config import settings
from semantic_searcher.database import async_session
from semantic_searcher.models.db import Shard, Worker

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


async def get_shard_worker_map() -> dict[str, dict]:
    """Build mapping: shard_id -> {worker_endpoint, page_id_start, page_id_end}."""
    async with async_session() as session:
        shards = (await session.execute(select(Shard).where(Shard.status == "active"))).scalars().all()
        workers = (await session.execute(select(Worker).where(Worker.status == "active"))).scalars().all()
        worker_map = {w.worker_id: w.endpoint_url for w in workers}

        result = {}
        for s in shards:
            if s.primary_worker_id and s.primary_worker_id in worker_map:
                result[s.shard_id] = {
                    "endpoint": worker_map[s.primary_worker_id],
                    "start": s.page_id_start,
                    "end": s.page_id_end,
                }
        return result


async def migrate(batch_size: int, dry_run: bool):
    # Connect to hub Qdrant
    client = QdrantClient(host=settings.qdrant_host, port=settings.qdrant_port, timeout=120)
    info = client.get_collection("text_chunks_v2")
    total = info.points_count
    log.info("Source collection: %d points", total)

    # Get shard/worker mapping
    shard_map = await get_shard_worker_map()
    if not shard_map:
        log.error("No active shards with assigned workers. Register workers and assign shards first.")
        return
    log.info("Shard map: %d shards with workers", len(shard_map))

    # Build page_id range → shard lookup
    def find_shard(page_id: int) -> str | None:
        for sid, info in shard_map.items():
            if info["start"] <= page_id <= info["end"]:
                return sid
        return None

    # Scroll through all points and distribute to workers
    async with httpx.AsyncClient(timeout=30) as http:
        migrated = 0
        skipped = 0
        next_offset = None
        t_start = time.time()

        while True:
            result = client.scroll(
                collection_name="text_chunks_v2",
                limit=batch_size,
                offset=next_offset,
                with_payload=True,
                with_vectors=True,
            )
            points, next_offset = result
            if not points:
                break

            # Group points by target shard
            shard_batches: dict[str, list] = {}
            for p in points:
                page_id = p.payload.get("page_id", 0)
                sid = find_shard(page_id)
                if not sid:
                    skipped += 1
                    continue
                shard_batches.setdefault(sid, []).append({
                    "id": p.id,
                    "dense_vector": p.vector["dense"] if isinstance(p.vector, dict) else p.vector,
                    "bm25_text": f"{p.payload.get('title', '')} {p.payload.get('chunk_text', '')}".strip(),
                    "payload": p.payload,
                })

            # Send to each worker
            for sid, batch in shard_batches.items():
                endpoint = shard_map[sid]["endpoint"]
                if dry_run:
                    migrated += len(batch)
                    continue

                try:
                    resp = await http.post(
                        f"{endpoint}/api/worker/vector-batch",
                        json={
                            "shard_id": sid,
                            "collection": "text_chunks_v2",
                            "points": batch,
                        },
                    )
                    if resp.status_code == 200:
                        migrated += len(batch)
                    else:
                        log.warning("Worker %s returned %d for shard %s", endpoint, resp.status_code, sid)
                except Exception as e:
                    log.warning("Failed to send to %s: %s", endpoint, e)

            elapsed = time.time() - t_start
            rate = migrated / elapsed if elapsed > 0 else 0
            eta = (total - migrated) / rate / 3600 if rate > 0 else 0
            log.info("Migrated %d / %d (skipped %d, %.0f/sec, ETA %.1fh)%s",
                     migrated, total, skipped, rate, eta, " [DRY RUN]" if dry_run else "")

            if next_offset is None:
                break

    elapsed = time.time() - t_start
    log.info("Migration complete: %d points in %.1f min", migrated, elapsed / 60)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    asyncio.run(migrate(args.batch_size, args.dry_run))


if __name__ == "__main__":
    main()
