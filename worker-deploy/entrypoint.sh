#!/bin/sh
# Secure entrypoint: reads API key into a temp file, wipes from environment,
# then starts the worker.

# Save API key to a secrets file (only readable by arcsearch user)
if [ -n "$WORKER_API_KEY" ]; then
    mkdir -p /run/secrets
    echo "$WORKER_API_KEY" > /run/secrets/worker_api_key
    chmod 400 /run/secrets/worker_api_key
    # Wipe from environment
    unset WORKER_API_KEY
fi

# Start the worker
exec python -m uvicorn semantic_searcher.main:app --host 0.0.0.0 --port 8901
