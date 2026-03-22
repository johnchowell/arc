"""API key authentication for distributed worker endpoints."""

import hashlib
import hmac
import logging
import secrets

from fastapi import Depends, HTTPException, Request
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from semantic_searcher.database import get_session
from semantic_searcher.models.db import Worker

log = logging.getLogger(__name__)

_bearer = HTTPBearer()


def hash_api_key(key: str) -> str:
    """One-way hash of an API key using SHA-256 with salt prefix."""
    return hashlib.sha256(key.encode()).hexdigest()


def generate_api_key() -> str:
    """Generate a cryptographically secure API key."""
    return secrets.token_urlsafe(48)


async def verify_worker(
    request: Request,
    credentials: HTTPAuthorizationCredentials = Depends(_bearer),
    session: AsyncSession = Depends(get_session),
) -> Worker:
    """Dependency that validates the worker API key and returns the Worker row.

    Checks:
    1. Bearer token is present
    2. X-Worker-ID header is present
    3. Token hash matches the stored hash for that worker
    4. Worker is not banned or offline
    """
    worker_id = request.headers.get("X-Worker-ID")
    if not worker_id:
        raise HTTPException(status_code=401, detail="Missing X-Worker-ID header")

    key_hash = hash_api_key(credentials.credentials)

    result = await session.execute(
        select(Worker).where(Worker.worker_id == worker_id)
    )
    worker = result.scalar_one_or_none()

    if worker is None:
        raise HTTPException(status_code=401, detail="Unknown worker")

    if not hmac.compare_digest(worker.api_key_hash, key_hash):
        raise HTTPException(status_code=401, detail="Invalid API key")

    if worker.status == "banned":
        raise HTTPException(status_code=403, detail="Worker is banned")

    return worker
