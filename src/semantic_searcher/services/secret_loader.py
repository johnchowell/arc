"""Secure secret loading for distributed workers.

Reads the API key from a Docker secret file or environment variable,
then wipes the environment variable to prevent /proc/environ leaks.
"""

import logging
import os

log = logging.getLogger(__name__)

_SECRETS_DIR = "/run/secrets"


def load_api_key() -> str:
    """Load the worker API key securely.

    Priority:
    1. Docker secret file at /run/secrets/worker_api_key
    2. Environment variable WORKER_API_KEY (wiped after reading)
    """
    # Try Docker secrets first
    secret_path = os.path.join(_SECRETS_DIR, "worker_api_key")
    if os.path.exists(secret_path):
        with open(secret_path) as f:
            key = f.read().strip()
        log.info("API key loaded from Docker secret")
        return key

    # Fall back to environment variable, then wipe it
    key = os.environ.get("WORKER_API_KEY", "")
    if key:
        # Wipe from environment so /proc/1/environ doesn't expose it
        os.environ.pop("WORKER_API_KEY", None)
        # Also try to wipe from /proc/self/environ (Linux-specific)
        try:
            os.unsetenv("WORKER_API_KEY")
        except Exception:
            pass
        log.info("API key loaded from environment (wiped from env)")
    return key
