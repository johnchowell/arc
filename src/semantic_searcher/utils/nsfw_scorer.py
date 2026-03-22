"""NSFW scoring utilities for CLIP-based classification.

Shared constants and helpers used by both the indexer (at index time)
and the searcher (for backfill / startup).
"""

import numpy as np

from semantic_searcher.services.clip_service import CLIPService

# ── Image NSFW prompts ──────────────────────────────────────────────
NSFW_IMAGE_PROMPTS = [
    "explicit nudity",
    "pornographic content",
    "naked person",
    "sexual content",
    "erotic photo",
    "nude body",
    "adult sexual imagery",
    "graphic sexual act",
]

NSFW_IMAGE_THRESHOLD = 0.30  # cosine similarity above this = possibly explicit

# ── Text NSFW zero-shot prompts ─────────────────────────────────────
NSFW_TEXT_PROMPTS = [
    "pornographic sexual content",
    "explicit erotic fiction",
    "graphic description of sexual acts",
    "adult pornography website",
    "nude sexual imagery",
]

SAFE_TEXT_PROMPTS = [
    "educational article about science and nature",
    "news report about politics and current events",
    "technical documentation and reference material",
    "family-friendly entertainment and media",
    "business and finance information",
]

NSFW_TEXT_TEMP = 0.1       # softmax temperature for zero-shot classification
NSFW_TEXT_MEDIAN = 0.7     # median P(nsfw) across page chunks to flag a page

# ── Cached prompt vectors (populated on first use) ──────────────────
_nsfw_image_vecs: np.ndarray | None = None
_nsfw_text_vecs: np.ndarray | None = None
_safe_text_vecs: np.ndarray | None = None


def _get_nsfw_image_vecs(clip: CLIPService) -> np.ndarray:
    global _nsfw_image_vecs
    if _nsfw_image_vecs is None:
        _nsfw_image_vecs = clip.encode_texts(NSFW_IMAGE_PROMPTS)
    return _nsfw_image_vecs


def _get_nsfw_text_vecs(clip: CLIPService) -> tuple[np.ndarray, np.ndarray]:
    global _nsfw_text_vecs, _safe_text_vecs
    if _nsfw_text_vecs is None:
        _nsfw_text_vecs = clip.encode_texts(NSFW_TEXT_PROMPTS)
        _safe_text_vecs = clip.encode_texts(SAFE_TEXT_PROMPTS)
    return _nsfw_text_vecs, _safe_text_vecs


def score_image_nsfw(clip: CLIPService, image_vec: np.ndarray) -> float:
    """Return max cosine similarity of an image embedding to NSFW prompts."""
    nsfw_vecs = _get_nsfw_image_vecs(clip)
    # image_vec: (512,)  nsfw_vecs: (K, 512)
    sims = image_vec @ nsfw_vecs.T  # (K,)
    return float(np.max(sims))


# ── Icon / UI element prompts ──────────────────────────────────────
ICON_IMAGE_PROMPTS = [
    "small website icon or favicon",
    "UI button icon or toolbar symbol",
    "simple flat vector pictogram",
    "app icon or interface glyph",
    "navigation menu icon or hamburger button",
    "social media share button icon",
    "arrow or chevron UI element",
    "simple geometric logo mark on solid background",
]

ICON_IMAGE_THRESHOLD = 0.28  # cosine similarity above this = likely an icon

_icon_image_vecs: np.ndarray | None = None


def _get_icon_image_vecs(clip: CLIPService) -> np.ndarray:
    global _icon_image_vecs
    if _icon_image_vecs is None:
        _icon_image_vecs = clip.encode_texts(ICON_IMAGE_PROMPTS)
    return _icon_image_vecs


def score_image_icon(clip: CLIPService, image_vec: np.ndarray) -> float:
    """Return max cosine similarity of an image embedding to icon/UI prompts."""
    icon_vecs = _get_icon_image_vecs(clip)
    sims = image_vec @ icon_vecs.T
    return float(np.max(sims))


def score_page_text_nsfw(clip: CLIPService, chunk_vecs: np.ndarray) -> bool:
    """Return True if a page's text chunks are classified as NSFW.

    Uses zero-shot contrastive classification: median P(nsfw) >= threshold.
    """
    if chunk_vecs is None or len(chunk_vecs) == 0:
        return False

    nsfw_vecs, safe_vecs = _get_nsfw_text_vecs(clip)

    nsfw_sims = chunk_vecs @ nsfw_vecs.T  # (N, K1)
    safe_sims = chunk_vecs @ safe_vecs.T  # (N, K2)
    nsfw_max = nsfw_sims.max(axis=1)      # (N,)
    safe_max = safe_sims.max(axis=1)      # (N,)

    # P(nsfw) via temperature-scaled softmax
    p_nsfw = np.exp(nsfw_max / NSFW_TEXT_TEMP) / (
        np.exp(nsfw_max / NSFW_TEXT_TEMP) + np.exp(safe_max / NSFW_TEXT_TEMP)
    )

    return float(np.median(p_nsfw)) >= NSFW_TEXT_MEDIAN
