from unittest.mock import MagicMock, patch
from dataclasses import dataclass

import numpy as np
import pytest

from semantic_searcher.services.searcher import SearchService, ImageSearchResult


def _make_qdrant_point(id, score, vector, payload):
    """Create a mock Qdrant ScoredPoint."""
    p = MagicMock()
    p.id = id
    p.score = score
    p.vector = vector.tolist()
    p.payload = payload
    return p


def _make_service(n_images: int, dim: int = 512) -> SearchService:
    """Build a SearchService with mock Qdrant returning synthetic image results."""
    clip = MagicMock()
    qdrant = MagicMock()
    svc = SearchService(clip, qdrant=qdrant)

    svc._pages = {
        1: {"url": "https://a.com", "title": "Page A", "meta_description": "", "indexed_at": None,
            "language": None, "domain": "a.com", "tld_group": "com", "content_category": None},
        2: {"url": "https://b.com", "title": "Page B", "meta_description": "", "indexed_at": None,
            "language": None, "domain": "b.com", "tld_group": "com", "content_category": None},
    }

    if n_images > 0:
        rng = np.random.default_rng(42)
        base = rng.standard_normal(dim).astype(np.float32)
        base /= np.linalg.norm(base)
        noise = rng.standard_normal((n_images, dim)).astype(np.float32) * 0.04
        embs = base + noise
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        embs = embs / norms

        # Store for test use
        svc._embs = embs
        svc._base_vec = base.copy()

        def _mock_query_points(**kwargs):
            query_vec = np.array(kwargs.get("query", base), dtype=np.float32)
            limit = kwargs.get("limit", 500)
            # Compute cosine similarities
            scores = embs @ query_vec
            indices = np.argsort(-scores)[:limit]
            points = []
            for idx in indices:
                score = float(scores[idx])
                if score < 0.2:
                    break
                points.append(_make_qdrant_point(
                    id=int(idx),
                    score=score,
                    vector=embs[idx],
                    payload={
                        "image_url": f"https://img.com/{idx}.jpg",
                        "alt_text": f"alt {idx}",
                        "nsfw_score": 0.0,
                        "icon_score": 0.0,
                        "page_ids": [1 + (idx % 2)],
                        "domain": "img.com",
                        "tld_group": "com",
                        "content_category": None,
                    },
                ))
            result = MagicMock()
            result.points = points
            return result

        qdrant.query_points.side_effect = _mock_query_points
    else:
        svc._qdrant = None  # no Qdrant = empty results

    return svc


class TestSearchImagesOrdering:
    def test_results_sorted_by_score_descending(self):
        svc = _make_service(10)
        svc.clip.encode_query.return_value = svc._embs[0]

        results, _, _ = svc.search_images("test", limit=10)
        scores = [r.score for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_top_result_matches_aligned_vector(self):
        svc = _make_service(10)
        svc.clip.encode_query.return_value = svc._embs[3]

        results, _, _ = svc.search_images("test", limit=5)
        assert results[0].image_url == "https://img.com/3.jpg"
        assert results[0].score == pytest.approx(1.0, abs=1e-3)


class TestSearchImagesLimit:
    def test_respects_limit(self):
        svc = _make_service(20)
        svc.clip.encode_query.return_value = svc._base_vec

        results, _, _ = svc.search_images("test", limit=5)
        assert len(results) == 5

    def test_limit_larger_than_index(self):
        svc = _make_service(3)
        svc.clip.encode_query.return_value = svc._base_vec

        results, _, _ = svc.search_images("test", limit=100)
        assert len(results) == 3


class TestSearchImagesEmpty:
    def test_empty_index_returns_empty(self):
        svc = _make_service(0)
        results, _, elapsed = svc.search_images("test", limit=10)
        assert results == []
        assert elapsed >= 0
        svc.clip.encode_query.assert_not_called()


class TestSearchImagesPageContext:
    def test_results_contain_page_metadata(self):
        svc = _make_service(4)
        svc.clip.encode_query.return_value = svc._embs[0]

        results, _, _ = svc.search_images("test", limit=4)
        for r in results:
            assert isinstance(r, ImageSearchResult)
            assert r.page_url in ("https://a.com", "https://b.com")
            assert r.page_title in ("Page A", "Page B")
            assert r.image_url.startswith("https://img.com/")

    def test_skips_images_with_missing_page(self):
        svc = _make_service(4)
        del svc._pages[2]
        svc.clip.encode_query.return_value = svc._base_vec

        results, _, _ = svc.search_images("test", limit=10)
        for r in results:
            assert r.page_id == 1


class TestSearchImagesElapsed:
    def test_returns_positive_elapsed_ms(self):
        svc = _make_service(5)
        svc.clip.encode_query.return_value = svc._base_vec

        _, _, elapsed = svc.search_images("test")
        assert isinstance(elapsed, float)
        assert elapsed >= 0
