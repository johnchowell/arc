#!/bin/bash
set -e

# Build a worker package with an embedded API key.
# Run this on the hub to register a new worker and create a ready-to-deploy package.
#
# Usage:
#   ./scripts/build-worker-package.sh --name "worker-alice" --endpoint "http://alice.example.com:8901"
#
# Output: dist/arcsearch-worker-<worker_id>.sh (self-extracting installer with baked key)

cd /arcsearch

NAME=""
ENDPOINT=""
HUB_URL="http://$(hostname -I | awk '{print $1}'):8900"

while [[ $# -gt 0 ]]; do
    case $1 in
        --name)      NAME="$2"; shift 2;;
        --endpoint)  ENDPOINT="$2"; shift 2;;
        --hub-url)   HUB_URL="$2"; shift 2;;
        *)           echo "Unknown: $1"; exit 1;;
    esac
done

[ -z "$NAME" ] && read -p "Worker name: " NAME
[ -z "$ENDPOINT" ] && ENDPOINT="http://0.0.0.0:8901"

echo "=== Registering worker on hub ==="
RESPONSE=$(curl -s -X POST "$HUB_URL/api/worker/register" \
    -H "Content-Type: application/json" \
    -d "{\"name\": \"$NAME\", \"endpoint_url\": \"$ENDPOINT\"}")

WORKER_ID=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['worker_id'])")
API_KEY=$(echo "$RESPONSE" | python3 -c "import sys,json; print(json.load(sys.stdin)['api_key'])")

if [ -z "$WORKER_ID" ] || [ -z "$API_KEY" ]; then
    echo "ERROR: Failed to register worker"
    echo "$RESPONSE"
    exit 1
fi

echo "  Worker ID: $WORKER_ID"
echo "  API Key:   ${API_KEY:0:10}..."
echo ""

echo "=== Building installer ==="
# Build the base installer
bash /arcsearch/scripts/build-installer.sh

echo "=== Embedding credentials ==="
mkdir -p /arcsearch/dist

# Create the final installer with baked credentials
OUTFILE="/arcsearch/dist/arcsearch-worker-${WORKER_ID:0:8}.sh"
sed \
    -e "s|^HUB_URL=\"\"|HUB_URL=\"$HUB_URL\"|" \
    -e "s|^API_KEY=\"\"|API_KEY=\"$API_KEY\"|" \
    -e "s|^WORKER_ID=\"\"|WORKER_ID=\"$WORKER_ID\"|" \
    -e "s|^WORKER_NAME=\"\"|WORKER_NAME=\"$NAME\"|" \
    /arcsearch/dist/install-arcsearch-worker.sh > "$OUTFILE"

chmod +x "$OUTFILE"

SIZE=$(du -h "$OUTFILE" | cut -f1)
echo ""
echo "=== Worker package built ==="
echo "  File: $OUTFILE ($SIZE)"
echo "  Worker: $NAME ($WORKER_ID)"
echo "  Hub: $HUB_URL"
echo ""
echo "Deploy to the worker machine:"
echo "  scp $OUTFILE user@worker-host:~/"
echo "  ssh user@worker-host 'chmod +x $(basename $OUTFILE) && sudo ./$(basename $OUTFILE)'"
echo ""
echo "The API key is baked into the installer — no manual configuration needed."
