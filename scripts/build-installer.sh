#!/bin/bash
set -e

# Build the self-extracting worker installer
# Packages source + pyproject.toml + static files into the install-worker.sh script

cd /arcsearch

echo "Building ArcSearch worker installer..."

# Create a temp tar of the source
TMPTAR=$(mktemp /tmp/arcsearch-src-XXXXXX.tar.gz)

tar czf "$TMPTAR" \
    --exclude='__pycache__' \
    --exclude='*.pyc' \
    --exclude='.git' \
    --exclude='venv' \
    --exclude='scripts' \
    --exclude='worker-deploy' \
    --exclude='*.log' \
    --exclude='.index_cache' \
    --exclude='*.npy' \
    --exclude='*.pkl' \
    --exclude='*.egg-info' \
    -C /arcsearch \
    src/ pyproject.toml static/ Dockerfile.worker docker-compose.worker.yml

# Build the installer: header script + base64 archive
OUTFILE="/arcsearch/dist/install-arcsearch-worker.sh"
mkdir -p /arcsearch/dist

# Copy the installer script (up to the marker)
head -n -1 /arcsearch/install-worker.sh > "$OUTFILE"

# Append the marker and base64 archive
echo "__ARCSEARCH_SOURCE_ARCHIVE__" >> "$OUTFILE"
base64 "$TMPTAR" >> "$OUTFILE"

chmod +x "$OUTFILE"
rm "$TMPTAR"

SIZE=$(du -h "$OUTFILE" | cut -f1)
echo "Built: $OUTFILE ($SIZE)"
echo "To install on a new machine:"
echo "  scp $OUTFILE user@host:~/"
echo "  ssh user@host 'chmod +x install-arcsearch-worker.sh && ./install-arcsearch-worker.sh --hub-url https://hub:8900 --api-key <key> --worker-id <id>'"
