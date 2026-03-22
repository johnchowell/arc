import asyncio
import functools
import logging
import numpy as np
import torch
import open_clip
from PIL import Image

from semantic_searcher.config import settings

log = logging.getLogger(__name__)

# Max tokens per CLIP forward pass to avoid OOM
_TEXT_BATCH_SIZE = 256
_IMAGE_BATCH_SIZE = 64


class CLIPService:
    def __init__(self, device: str | None = None):
        self.model = None
        self.preprocess = None
        self.tokenizer = None
        self._device_str = device
        self.device = None

    async def start(self):
        if self._device_str:
            self.device = torch.device(self._device_str)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        log.info("Loading CLIP %s (%s) on %s", settings.clip_model, settings.clip_pretrained, self.device)
        self.model, _, self.preprocess = open_clip.create_model_and_transforms(
            settings.clip_model, pretrained=settings.clip_pretrained, device=self.device,
        )
        self.tokenizer = open_clip.get_tokenizer(settings.clip_model)
        self.model.eval()
        log.info("CLIP model loaded")

    @property
    def embedding_dim(self) -> int:
        return 512

    @torch.no_grad()
    def encode_texts(self, texts: list[str]) -> np.ndarray:
        """Encode texts in sub-batches to avoid OOM. Synchronous — use encode_texts_async for non-blocking."""
        all_features = []
        for i in range(0, len(texts), _TEXT_BATCH_SIZE):
            batch = texts[i:i + _TEXT_BATCH_SIZE]
            tokens = self.tokenizer(batch).to(self.device)
            features = self.model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu())
        return torch.cat(all_features, dim=0).numpy().astype(np.float32)

    @torch.no_grad()
    def encode_images(self, images: list[Image.Image]) -> np.ndarray:
        """Encode images in sub-batches. Synchronous — use encode_images_async for non-blocking."""
        all_features = []
        for i in range(0, len(images), _IMAGE_BATCH_SIZE):
            batch_imgs = images[i:i + _IMAGE_BATCH_SIZE]
            batch = torch.stack([self.preprocess(img) for img in batch_imgs]).to(self.device)
            features = self.model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
            all_features.append(features.cpu())
        return torch.cat(all_features, dim=0).numpy().astype(np.float32)

    async def encode_texts_async(self, texts: list[str]) -> np.ndarray:
        """Non-blocking text encoding — runs in a thread to free the event loop."""
        return await asyncio.to_thread(self.encode_texts, texts)

    async def encode_images_async(self, texts: list[Image.Image]) -> np.ndarray:
        """Non-blocking image encoding — runs in a thread to free the event loop."""
        return await asyncio.to_thread(self.encode_images, texts)

    @functools.lru_cache(maxsize=2048)
    def _encode_query_cached(self, query: str) -> bytes:
        """Cache query vectors as bytes so lru_cache can store them (hashable)."""
        vec = self.encode_texts([query])[0]
        return vec.tobytes()

    @torch.no_grad()
    def encode_query(self, query: str) -> np.ndarray:
        return np.frombuffer(self._encode_query_cached(query), dtype=np.float32).copy()
