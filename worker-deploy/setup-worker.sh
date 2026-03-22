#!/bin/bash
set -e

WORKER_DIR=/home/jc/arcsearch-worker

echo "=== Setting up ArcSearch crawl worker ==="

# Create data dir
mkdir -p "$WORKER_DIR/data/html" "$WORKER_DIR/data/tokenizer"

# Create venv if not exists
if [ ! -d "$WORKER_DIR/venv" ]; then
    echo "Creating Python venv..."
    python3 -m venv "$WORKER_DIR/venv"
fi

source "$WORKER_DIR/venv/bin/activate"

echo "Installing dependencies..."
pip install --upgrade pip wheel setuptools -q
pip install torch --index-url https://download.pytorch.org/whl/cu124 -q 2>&1 | tail -1
pip install -e "$WORKER_DIR" -q 2>&1 | tail -3
pip install sentence-transformers -q 2>&1 | tail -1

# Install playwright browsers
echo "Installing Playwright Chromium..."
playwright install chromium 2>&1 | tail -2

echo "=== Setup complete ==="
echo "Start with: sudo systemctl start arcsearch-worker"
