#!/bin/bash
set -e

# ============================================================================
# ArcSearch Distributed Worker Installer
#
# Installs a self-contained crawl worker that connects to an ArcSearch hub.
# Includes: embedded Qdrant, MPNet text encoder, CLIP image encoder, crawler.
#
# Usage:
#   chmod +x install-worker.sh
#   ./install-worker.sh --hub-url https://hub:8900 --api-key <key> --worker-id <id>
#
# Or interactive:
#   ./install-worker.sh
# ============================================================================

INSTALL_DIR="/opt/arcsearch-worker"
QDRANT_VERSION="v1.17.0"
PYTHON_MIN="3.12"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log()   { echo -e "${GREEN}[+]${NC} $1"; }
warn()  { echo -e "${YELLOW}[!]${NC} $1"; }
error() { echo -e "${RED}[!]${NC} $1"; exit 1; }

# --- Parse arguments ---
HUB_URL=""
API_KEY=""
WORKER_ID=""
WORKER_NAME=""
WORKER_PORT="8901"

NO_CUDA=false

while [[ $# -gt 0 ]]; do
    case $1 in
        --hub-url)    HUB_URL="$2"; shift 2;;
        --api-key)    API_KEY="$2"; shift 2;;
        --worker-id)  WORKER_ID="$2"; shift 2;;
        --name)       WORKER_NAME="$2"; shift 2;;
        --port)       WORKER_PORT="$2"; shift 2;;
        --install-dir) INSTALL_DIR="$2"; shift 2;;
        --no-cuda)    NO_CUDA=true; shift;;
        *)            error "Unknown option: $1";;
    esac
done

# --- Interactive prompts for missing values ---
if [ -z "$HUB_URL" ]; then
    read -p "Hub URL (e.g. https://hub.example.com:8900): " HUB_URL
fi
if [ -z "$API_KEY" ]; then
    read -sp "Worker API key: " API_KEY; echo
fi
if [ -z "$WORKER_ID" ]; then
    WORKER_ID=$(python3 -c "import uuid; print(uuid.uuid4())" 2>/dev/null || cat /proc/sys/kernel/random/uuid)
    log "Generated worker ID: $WORKER_ID"
fi
if [ -z "$WORKER_NAME" ]; then
    WORKER_NAME="worker-$(hostname)"
fi

[ -z "$HUB_URL" ] && error "Hub URL is required"
[ -z "$API_KEY" ] && error "API key is required"

log "ArcSearch Distributed Worker Installer"
log "======================================="
log "Install dir:  $INSTALL_DIR"
log "Hub URL:      $HUB_URL"
log "Worker ID:    $WORKER_ID"
log "Worker name:  $WORKER_NAME"
log "Worker port:  $WORKER_PORT"
echo

# --- Check prerequisites ---
log "Checking prerequisites..."

# Python
PYTHON=""
for p in python3.12 python3; do
    if command -v $p &>/dev/null; then
        ver=$($p --version 2>&1 | grep -oP '\d+\.\d+')
        if python3 -c "import sys; exit(0 if sys.version_info >= (3, 12) else 1)" 2>/dev/null; then
            PYTHON=$p
            break
        fi
    fi
done
[ -z "$PYTHON" ] && error "Python >= 3.12 required (found: $(python3 --version 2>&1))"
log "Python: $($PYTHON --version)"

# Docker (for Qdrant)
if ! command -v docker &>/dev/null; then
    warn "Docker not found — installing..."
    curl -fsSL https://get.docker.com | sh
    sudo systemctl enable docker
    sudo systemctl start docker
fi
log "Docker: $(docker --version)"

# GPU detection
GPU_AVAILABLE=false
if [ "$NO_CUDA" = true ]; then
    log "CUDA disabled by --no-cuda flag — will use CPU"
elif command -v nvidia-smi &>/dev/null; then
    GPU_COUNT=$(nvidia-smi --query-gpu=count --format=csv,noheader | head -1)
    if [ "$GPU_COUNT" -gt 0 ] 2>/dev/null; then
        GPU_AVAILABLE=true
        log "GPU: $GPU_COUNT NVIDIA GPU(s) detected"
    fi
fi
if [ "$GPU_AVAILABLE" = false ] && [ "$NO_CUDA" = false ]; then
    warn "No GPU detected — will use CPU (slower encoding)"
fi

# --- Create install directory ---
log "Creating install directory..."
sudo mkdir -p "$INSTALL_DIR"
sudo chown $(whoami):$(whoami) "$INSTALL_DIR"

# --- Start Qdrant container ---
log "Starting Qdrant container..."
sudo docker pull qdrant/qdrant:${QDRANT_VERSION} 2>&1 | tail -1
sudo docker stop arcsearch-qdrant 2>/dev/null || true
sudo docker rm arcsearch-qdrant 2>/dev/null || true
sudo docker run -d \
    --name arcsearch-qdrant \
    --restart unless-stopped \
    -v "$INSTALL_DIR/qdrant_storage:/qdrant/storage" \
    -p 127.0.0.1:6333:6333 \
    qdrant/qdrant:${QDRANT_VERSION}
log "Qdrant running on 127.0.0.1:6333"

# --- Extract source ---
log "Extracting ArcSearch source..."
MARKER="__ARCSEARCH_SOURCE_ARCHIVE__"
ARCHIVE_LINE=$(grep -n "^${MARKER}$" "$0" | tail -1 | cut -d: -f1)

if [ -n "$ARCHIVE_LINE" ]; then
    tail -n +$((ARCHIVE_LINE + 1)) "$0" | base64 -d | tar xz -C "$INSTALL_DIR"
    log "Source extracted from installer"
else
    if [ -d "/arcsearch/src" ]; then
        cp -r /arcsearch/src "$INSTALL_DIR/"
        cp /arcsearch/pyproject.toml "$INSTALL_DIR/"
        cp /arcsearch/Dockerfile.worker "$INSTALL_DIR/Dockerfile"
        cp /arcsearch/docker-compose.worker.yml "$INSTALL_DIR/docker-compose.yml"
        mkdir -p "$INSTALL_DIR/static"
        cp -r /arcsearch/static/* "$INSTALL_DIR/static/" 2>/dev/null || true
        log "Source copied from local installation"
    else
        error "No source found."
    fi
fi

# --- Write config ---
log "Writing configuration..."
cat > "$INSTALL_DIR/.env" << ENVEOF
# ArcSearch Distributed Worker
HUB_URL=$HUB_URL
WORKER_API_KEY=$API_KEY
WORKER_ID=$WORKER_ID
WORKER_PORT=$WORKER_PORT
ENVEOF

# --- Write docker-compose override if not present ---
if [ ! -f "$INSTALL_DIR/docker-compose.yml" ]; then
    cat > "$INSTALL_DIR/docker-compose.yml" << 'DCEOF'
version: '3.8'
services:
  qdrant:
    image: qdrant/qdrant:v1.17.0
    volumes:
      - qdrant_data:/qdrant/storage
    networks:
      - worker_internal
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 4G
          cpus: '2'
    security_opt:
      - no-new-privileges:true

  worker:
    build: .
    depends_on: [qdrant]
    env_file: .env
    environment:
      - DISTRIBUTED_MODE=true
      - HUB_ROLE=worker
      - QDRANT_HOST=qdrant
      - QDRANT_PORT=6333
      - CRAWL_ONLY=true
      - CRAWLER_WORKERS=50
      - RAID_PATH=/app/data
      - WEBINDEX_DIR=/app/data
    ports:
      - "${WORKER_PORT:-8901}:8901"
    volumes:
      - worker_data:/app/data
    networks:
      - worker_internal
      - worker_external
    restart: unless-stopped
    deploy:
      resources:
        limits:
          memory: 8G
          cpus: '4'
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - NET_BIND_SERVICE

networks:
  worker_internal:
    internal: true
  worker_external:

volumes:
  qdrant_data:
  worker_data:
DCEOF
fi

# Also ensure Dockerfile exists
if [ ! -f "$INSTALL_DIR/Dockerfile" ]; then
    cp "$INSTALL_DIR/Dockerfile.worker" "$INSTALL_DIR/Dockerfile" 2>/dev/null || true
fi

# --- Build and start containers ---
log "Building worker container (this takes several minutes on first run)..."
cd "$INSTALL_DIR"
sudo docker compose build 2>&1 | tail -5

log "Starting worker..."
sudo docker compose up -d 2>&1

# --- Create systemd service for docker compose ---
sudo tee /etc/systemd/system/arcsearch-worker.service > /dev/null << SVCEOF
[Unit]
Description=ArcSearch Distributed Worker
After=docker.service
Requires=docker.service

[Service]
Type=oneshot
RemainAfterExit=yes
WorkingDirectory=$INSTALL_DIR
ExecStart=/usr/bin/docker compose up -d
ExecStop=/usr/bin/docker compose down
TimeoutStartSec=120

[Install]
WantedBy=multi-user.target
SVCEOF

sudo systemctl daemon-reload
sudo systemctl enable arcsearch-worker

sleep 10
if sudo docker compose ps --status running | grep -q worker; then
    log "Worker containers are running!"
else
    warn "Containers may still be starting (model loading takes ~60s)"
    warn "Check: cd $INSTALL_DIR && docker compose logs -f worker"
fi

echo
log "============================================"
log "  ArcSearch Worker Installation Complete!"
log "============================================"
log ""
log "  Worker ID:  $WORKER_ID"
log "  Service:    sudo systemctl {start|stop|restart} arcsearch-worker"
log "  Logs:       journalctl -u arcsearch-worker -f"
log "  Config:     $INSTALL_DIR/.env"
log "  Qdrant:     http://127.0.0.1:6333"
log ""
log "  The worker will:"
log "  1. Connect to hub at $HUB_URL"
log "  2. Pull crawl jobs from the shared queue"
log "  3. Encode pages with MPNet + CLIP locally"
log "  4. Store vectors in local Qdrant shard"
log "  5. Respond to search queries from the hub"
log ""

exit 0

# Source archive is appended below this marker by the build script
__ARCSEARCH_SOURCE_ARCHIVE__
