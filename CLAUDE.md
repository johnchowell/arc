# ArcSearch - CLAUDE.md

## Overview
ArcSearch (formerly Semantic Searcher) is a self-hosted web search engine with semantic vector search, web crawling, and indexing.

## Architecture
- **FastAPI + Uvicorn** on port 8900, managed by systemd (`arcsearch.service`)
- **MySQL 8.0** on 127.0.0.1:3306 (default port), database `semantic_searcher`, user `root`, no password
- **Qdrant** vector database (Docker container) on 127.0.0.1:6333/6334
  - Storage at `/home/jc/qdrant_storage`
  - `text_chunks` collection: 512-dim dense (INT8 quantized, always_ram) + BM25 sparse (on_disk=True)
  - `images` collection: 512-dim flat vectors
  - HNSW: in RAM (on_disk=False)
  - Hybrid search: dense + BM25 → RRF fusion
- **CLIP ViT-B-32** on 2x RTX 5070 GPUs (cuda:0, cuda:1)
- **Cross-encoder** re-ranking: `cross-encoder/ms-marco-MiniLM-L-6-v2` on CPU
- **Playwright** headless Chromium for SPA rendering (lazy-loaded, not started at boot)
- **SymSpell** spell correction

## Key Paths
| What | Path |
|------|------|
| App code | `/arcsearch/` |
| Source code | `/arcsearch/src/semantic_searcher/` |
| Python venv | `/arcsearch/venv/` |
| Static files | `/arcsearch/static/` |
| .env config | `/arcsearch/.env` |
| Qdrant data | `/home/jc/qdrant_storage/` |
| Webindex/HTML cache | `/drive/webindex/` |
| Tokenizer | `/drive/webindex/tokenizer/` |
| Systemd service | `/etc/systemd/system/arcsearch.service` |

## Key Files
- `src/semantic_searcher/main.py` — App lifespan, startup logic
- `src/semantic_searcher/services/searcher.py` — Hybrid RRF search, two-phase streaming
- `src/semantic_searcher/services/home/jc/qdrant_storage_collections.py` — Collection schemas
- `src/semantic_searcher/services/search_queue.py` — Bounded concurrency
- `src/semantic_searcher/services/indexer.py` — Dual-write (MySQL + Qdrant)
- `src/semantic_searcher/services/renderer.py` — Lazy Playwright browser pool
- `src/semantic_searcher/services/clip_service.py` — CLIP encoding (text + image)
- `src/semantic_searcher/config.py` — Pydantic settings (reads .env)
- `src/semantic_searcher/routers/search.py` — Search API (SSE streaming)
- `scripts/home/jc/qdrant_storage_bulk_load.py` — Bulk load MySQL → Qdrant

## .env Configuration
Key differences from the original machine:
- `MYSQL_PORT=3306` (was 3307)
- `MYSQL_PASSWORD=` (empty, was set)
- `RAID_PATH=/drive/webindex` (was /media/raid/webindex)
- `WEBINDEX_DIR=/drive/webindex` (was /media/raid/webindex)

## Commands
```bash
# Service management
sudo systemctl start|stop|restart arcsearch
journalctl -u arcsearch -f

# Qdrant container
sudo docker start|stop qdrant
# Dashboard: http://localhost:6333/dashboard

# Python
source /arcsearch/venv/bin/activate
export PYTHONPATH=/arcsearch/src

# Run tests
cd /arcsearch && source venv/bin/activate && pytest tests/ -v

# Manual run (dev)
cd /arcsearch && source venv/bin/activate && PYTHONPATH=src uvicorn semantic_searcher.main:app --host 0.0.0.0 --port 8900
```

## Index Stats (as of 2026-03-11)
- 387K+ indexed pages
- ~7.8M text embeddings (with BM25 sparse vectors)
- ~59K image embeddings

## Qdrant Config Details
- qdrant-client v1.17.0
- Dense vectors: on_disk=True, INT8 quantized (always_ram=True)
- BM25 sparse: on_disk=True, IDF modifier
- HNSW: on_disk=False (in RAM for speed)
- Payloads: on_disk_payload=True

## Search Features
- Two-phase SSE streaming: Phase 1 (dense-only fast), Phase 2 (hybrid RRF + cross-encoder)
- SymSpell spell correction (lazy, triggers when results < 5 or top score < 0.10)
- Cross-encoder top-10 reranking (_RERANK_WEIGHT=0.35)
- Freshness scoring, domain diversity, title dedup, navigational boost
- Rate limiting (burst: >6 in 2s → throttle, 10 violations → 24h ban)

## Network
- This machine (192.168.1.99) runs the app on port 8900
- The original machine proxies via nginx:
  - `engine.jchowell.com` → this machine:8900
  - `engine.jchowell.com/dashboard` → this machine:6333 (Qdrant dashboard, basic auth)

## Hardware
- 2x NVIDIA RTX 5070 (12GB VRAM each)
- 128GB DDR5 RAM
- 7.3TB RAID at /drive
- 913GB root SSD

## Migration Notes (2026-03-11)
- Migrated from original server (46GB DDR4, 2x older GPUs)
- Qdrant storage transferred as-is (no re-indexing needed)
- MySQL dump transferred directly
- Playwright set to lazy-load (not started at boot)
- Elasticsearch removed (was deprecated, replaced by Qdrant BM25)
