"""CLIP zero-shot content classification for web pages.

Reuses existing chunk embeddings — no additional CLIP encoding of page content needed.
"""

import numpy as np

from semantic_searcher.services.clip_service import CLIPService

CATEGORIES = [
    "news", "science", "technology", "education", "business", "health",
    "entertainment", "sports", "arts", "politics", "shopping",
    "history", "law", "reference", "travel", "food",
    "other",
]

_CATEGORY_PROMPTS: dict[str, list[str]] = {
    "news": [
        "breaking news article about current events",
        "news report covering recent happenings",
        "journalism and press coverage of events",
    ],
    "science": [
        "scientific research paper and academic study",
        "biology chemistry physics laboratory experiment",
        "peer-reviewed scientific journal publication",
    ],
    "technology": [
        "software programming and computer technology",
        "tech startup and digital innovation",
        "information technology and computing systems",
    ],
    "education": [
        "educational course material and learning resources",
        "university lecture notes and academic curriculum",
        "teaching tutorial and instructional content",
    ],
    "business": [
        "business finance and corporate management",
        "stock market investing and economic analysis",
        "company earnings report and financial statements",
    ],
    "health": [
        "medical health information and disease treatment",
        "healthcare wellness and fitness advice",
        "clinical medicine diagnosis and patient care",
    ],
    "entertainment": [
        "movie film television show and celebrity news",
        "music concerts and entertainment media",
        "video game and digital entertainment content",
    ],
    "sports": [
        "professional sports game scores and highlights",
        "athletic competition and team standings",
        "football basketball soccer sports coverage",
    ],
    "arts": [
        "visual art painting sculpture and gallery exhibition",
        "literature poetry and creative writing",
        "cultural arts museum and artistic expression",
    ],
    "politics": [
        "political election campaign and government policy",
        "congressional legislation and political debate",
        "geopolitics diplomacy and international relations",
    ],
    "shopping": [
        "online shopping product reviews and prices",
        "e-commerce store and retail merchandise",
        "buy purchase deals and consumer products",
    ],
    "history": [
        "historical account of past events and civilizations",
        "history of wars battles and ancient empires",
        "biography of historical figures and their era",
    ],
    "law": [
        "legal statute regulation and court ruling",
        "law firm attorney and legal advice services",
        "constitutional rights criminal justice and legislation",
    ],
    "reference": [
        "encyclopedia dictionary and factual reference material",
        "how-to guide manual and frequently asked questions",
        "wiki knowledge base and informational directory",
    ],
    "travel": [
        "travel destination guide and vacation planning",
        "hotel flight booking and tourism itinerary",
        "adventure tourism sightseeing and travel reviews",
    ],
    "food": [
        "recipe cooking instructions and meal preparation",
        "restaurant review dining and food criticism",
        "nutrition diet and culinary arts ingredients",
    ],
    "other": [
        "personal homepage and portfolio website",
        "error page placeholder or under construction notice",
        "unstructured miscellaneous content without clear topic",
    ],
}

# Cache for encoded category prompt vectors
_category_vectors: np.ndarray | None = None
_category_labels: list[str] = []


def _get_category_vectors(clip: CLIPService) -> tuple[np.ndarray, list[str]]:
    """Encode category prompts (cached after first call). Returns (vectors, labels)."""
    global _category_vectors, _category_labels
    if _category_vectors is not None:
        return _category_vectors, _category_labels

    all_prompts = []
    labels = []
    for cat in CATEGORIES:
        for prompt in _CATEGORY_PROMPTS[cat]:
            all_prompts.append(prompt)
            labels.append(cat)

    _category_vectors = clip.encode_texts(all_prompts)  # (N_prompts, 512)
    _category_labels = labels
    return _category_vectors, labels


def classify_page(clip: CLIPService, chunk_embeddings: np.ndarray) -> str:
    """Classify a page's content category using its existing chunk embeddings.

    Args:
        clip: CLIPService instance (used to encode category prompts on first call)
        chunk_embeddings: (N_chunks, 512) array of text chunk embeddings

    Returns:
        Category slug string (e.g. "news", "science", etc.)
    """
    if chunk_embeddings is None or len(chunk_embeddings) == 0:
        return "other"

    cat_vecs, cat_labels = _get_category_vectors(clip)

    # (N_chunks, N_prompts) similarity matrix
    sims = chunk_embeddings @ cat_vecs.T

    # Per chunk: best matching category
    votes: dict[str, int] = {}
    for chunk_idx in range(len(sims)):
        best_prompt_idx = int(np.argmax(sims[chunk_idx]))
        cat = cat_labels[best_prompt_idx]
        votes[cat] = votes.get(cat, 0) + 1

    # Plurality winner
    return max(votes, key=votes.get)
