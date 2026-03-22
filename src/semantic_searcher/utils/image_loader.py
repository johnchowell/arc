import logging
from io import BytesIO

import httpx
from PIL import Image

log = logging.getLogger(__name__)

MAX_IMAGE_SIZE = 10 * 1024 * 1024  # 10MB
MIN_IMAGE_WIDTH = 150
MIN_IMAGE_HEIGHT = 150


async def load_image(url: str, client: httpx.AsyncClient) -> Image.Image | None:
    try:
        resp = await client.get(url, timeout=5)
        if resp.status_code != 200:
            return None
        content_type = resp.headers.get("content-type", "")
        if not content_type.startswith("image/"):
            return None
        if len(resp.content) > MAX_IMAGE_SIZE:
            return None
        img = Image.open(BytesIO(resp.content)).convert("RGB")
        if img.width < MIN_IMAGE_WIDTH or img.height < MIN_IMAGE_HEIGHT:
            log.debug("Image too small (%dx%d): %s", img.width, img.height, url)
            return None
        return img
    except Exception as e:
        log.debug("Failed to load image %s: %s", url, e)
        return None
