#!/bin/bash
set -e

# Set up a development instance of ArcSearch
# Uses a separate database (semantic_searcher_dev) so you can test without touching production.

echo "=== ArcSearch Dev Instance Setup ==="

# Create dev database
echo "Creating dev database..."
mysql -u root -e "CREATE DATABASE IF NOT EXISTS semantic_searcher_dev" 2>/dev/null || \
    echo "Note: MySQL not available locally — set MYSQL_HOST in dev.env"

# Create dev data directory
mkdir -p /tmp/arcsearch-dev/data/html /tmp/arcsearch-dev/data/tokenizer

# Copy dev.env if not exists
if [ ! -f .env.dev ]; then
    cp dev.env .env.dev
    echo "Created .env.dev from template"
fi

echo ""
echo "Dev instance ready!"
echo ""
echo "To run in dev mode:"
echo "  source venv/bin/activate"
echo "  cp dev.env .env          # Use dev config"
echo "  PYTHONPATH=src uvicorn semantic_searcher.main:app --host 0.0.0.0 --port 8902 --reload"
echo ""
echo "To run tests:"
echo "  PYTHONPATH=src pytest tests/ -v"
echo ""
echo "Dev database: semantic_searcher_dev"
echo "Dev port: 8902 (different from prod 8900)"
echo "Dev data: /tmp/arcsearch-dev/data/"
echo ""
echo "IMPORTANT: Never commit .env files or API keys."
