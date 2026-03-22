"""MPNet-based text encoder for semantic search (replaces CLIP for text-to-text)."""

import asyncio
import functools
import logging

import numpy as np
import torch

log = logging.getLogger(__name__)

_TEXT_BATCH_SIZE = 256


class TextEncoderService:
    """Wraps sentence-transformers MPNet for text encoding."""

    MODEL_NAME = "all-mpnet-base-v2"
    EMBEDDING_DIM = 768

    def __init__(self, device: str | None = None):
        self._model = None
        self._device_str = device
        self.device = None

    async def start(self):
        if self._device_str:
            self.device = self._device_str
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        log.info("Loading %s on %s", self.MODEL_NAME, self.device)
        from sentence_transformers import SentenceTransformer
        self._model = SentenceTransformer(self.MODEL_NAME, device=self.device)
        log.info("Text encoder loaded (%s, %dd)", self.MODEL_NAME, self.EMBEDDING_DIM)

    @property
    def embedding_dim(self) -> int:
        return self.EMBEDDING_DIM

    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode texts in sub-batches. Returns (N, 768) float32 array."""
        all_vecs = []
        for i in range(0, len(texts), _TEXT_BATCH_SIZE):
            batch = texts[i:i + _TEXT_BATCH_SIZE]
            vecs = self._model.encode(batch, normalize_embeddings=True, show_progress_bar=False)
            all_vecs.append(vecs)
        return np.concatenate(all_vecs, axis=0).astype(np.float32)

    async def encode_texts_async(self, texts: list[str]) -> np.ndarray:
        """Non-blocking text encoding."""
        return await asyncio.to_thread(self.encode_texts, texts)

    @functools.lru_cache(maxsize=2048)
    def _encode_query_cached(self, query: str) -> bytes:
        vec = self.encode_texts([query])[0]
        return vec.tobytes()

    def encode_query(self, query: str) -> np.ndarray:
        return np.frombuffer(self._encode_query_cached(query), dtype=np.float32).copy()
