#!/usr/bin/env bash
# ============================================================
# One-time setup: install dependencies and clone tool repos
# Run on your Linux machine with NVIDIA GPU
# ============================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$SCRIPT_DIR/config.env"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

log()   { echo -e "${GREEN}[SETUP]${NC} $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*"; exit 1; }

# ── System checks ──────────────────────────────────────────

log "Checking system requirements..."

# NVIDIA GPU / CUDA
if ! command -v nvidia-smi &>/dev/null; then
    error "nvidia-smi not found. Install NVIDIA drivers first."
fi

GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
GPU_VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
log "GPU detected: $GPU_NAME (${GPU_VRAM} MiB)"

if [[ "$GPU_VRAM" -lt 8000 ]]; then
    warn "GPU has less than 8GB VRAM. Training may fail or require --resolution 4."
fi

if ! command -v nvcc &>/dev/null; then
    warn "nvcc not found. CUDA toolkit may not be installed."
    warn "Install with: sudo apt install nvidia-cuda-toolkit"
    warn "Or install from: https://developer.nvidia.com/cuda-downloads"
fi

# Python
if ! command -v python3 &>/dev/null; then
    error "python3 not found. Install Python 3.8+ first."
fi
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
log "Python: $PYTHON_VERSION"

# Git
if ! command -v git &>/dev/null; then
    error "git not found. Install git first."
fi

# FFmpeg
if ! command -v ffmpeg &>/dev/null; then
    warn "FFmpeg not found. Installing..."
    sudo apt-get update && sudo apt-get install -y ffmpeg
fi
log "FFmpeg: $(ffmpeg -version 2>&1 | head -1)"

# COLMAP
if ! command -v colmap &>/dev/null; then
    warn "COLMAP not found. Installing..."
    sudo apt-get update && sudo apt-get install -y colmap
    if ! command -v colmap &>/dev/null; then
        warn "COLMAP not available in apt. Trying snap..."
        sudo snap install colmap
    fi
fi
if command -v colmap &>/dev/null; then
    log "COLMAP: $(colmap --version 2>&1 | head -1 || echo 'installed')"
else
    error "Could not install COLMAP. Install manually: https://colmap.github.io/install.html"
fi

# Node.js
if ! command -v node &>/dev/null; then
    warn "Node.js not found. Installing via NodeSource..."
    curl -fsSL https://deb.nodesource.com/setup_20.x | sudo -E bash -
    sudo apt-get install -y nodejs
fi
NODE_VERSION=$(node --version 2>&1)
log "Node.js: $NODE_VERSION"

# ── Clone tool repositories ───────────────────────────────

mkdir -p "$TOOLS_DIR"

# 3D Gaussian Splatting (graphdeco-inria)
if [[ ! -d "$GAUSSIAN_SPLATTING_DIR" ]]; then
    log "Cloning graphdeco-inria/gaussian-splatting..."
    git clone --recursive https://github.com/graphdeco-inria/gaussian-splatting.git "$GAUSSIAN_SPLATTING_DIR"
else
    log "gaussian-splatting already cloned."
fi

# @mkkellogg/gaussian-splats-3d (for ksplat converter)
if [[ ! -d "$GAUSSIAN_SPLATS_3D_DIR" ]]; then
    log "Cloning mkkellogg/GaussianSplats3D..."
    git clone https://github.com/mkkellogg/GaussianSplats3D.git "$GAUSSIAN_SPLATS_3D_DIR"
else
    log "GaussianSplats3D already cloned."
fi

# ── Install Python dependencies for 3DGS ──────────────────

log "Setting up Python environment for 3DGS training..."

cd "$GAUSSIAN_SPLATTING_DIR"

if [[ ! -d "venv" ]]; then
    python3 -m venv venv
fi
source venv/bin/activate

pip install --upgrade pip

# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install 3DGS dependencies
pip install plyfile tqdm

# Build CUDA submodules
pip install submodules/diff-gaussian-rasterization
pip install submodules/simple-knn

deactivate
cd "$SCRIPT_DIR"

# ── Install Node.js dependencies for ksplat converter ─────

log "Installing Node.js dependencies for ksplat converter..."

cd "$GAUSSIAN_SPLATS_3D_DIR"
npm install
cd "$SCRIPT_DIR"

# ── Create data directory ─────────────────────────────────

mkdir -p "$DATA_DIR"

# ── Summary ────────────────────────────────────────────────

echo ""
log "=========================================="
log "  Setup complete!"
log "=========================================="
echo ""
echo "  GPU:      $GPU_NAME (${GPU_VRAM} MiB)"
echo "  Python:   $PYTHON_VERSION"
echo "  Node.js:  $NODE_VERSION"
echo "  FFmpeg:   $(command -v ffmpeg)"
echo "  COLMAP:   $(command -v colmap)"
echo ""
echo "  Next: place your video in the project and run:"
echo "    ./pipeline.sh /path/to/video.mp4 my_project_name"
echo ""
