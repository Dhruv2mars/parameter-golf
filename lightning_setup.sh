#!/bin/bash
# =============================================================================
# Lightning AI - One-Time Setup Script
# =============================================================================
# Run this ONCE when you create a new Studio to set up the environment.
# =============================================================================

set -euo pipefail

echo "=========================================="
echo "Parameter Golf - Lightning AI Setup"
echo "=========================================="

# Configuration
REPO_URL="https://github.com/Dhruv2mars/parameter-golf.git"
REPO_DIR="/teamspace/studios/this_studio/parameter-golf"
VOCAB_SIZE=${1:-1024}  # Default 1024 (works), pass 8192 as arg if needed

# =============================================================================
# 1. Install CLI if not present
# =============================================================================

echo "[1/5] Checking Lightning CLI..."
if ! command -v lightning &> /dev/null; then
    echo "Installing Lightning CLI..."
    pip install lightning -q
    lightning login
else
    echo "CLI already installed"
fi

# =============================================================================
# 2. Clone repo
# =============================================================================

echo "[2/5] Setting up repository..."
cd /teamspace/studios/this_studio/parameter-golf

if [ -d "$REPO_DIR" ]; then
    echo "Repo already exists, pulling latest..."
    cd "$REPO_DIR"
    git pull origin main
else
    echo "Cloning repository..."
    git clone "$REPO_URL" "$REPO_DIR"
    cd "$REPO_DIR"
fi

# =============================================================================
# 3. Install dependencies
# =============================================================================

echo "[3/5] Installing dependencies..."
pip install --upgrade pip -q
pip install numpy sentencepiece torch torchvision torchaudio huggingface-hub tqdm -q

# Verify torch + CUDA
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"

# =============================================================================
# 4. Download FineWeb data
# =============================================================================

echo "[4/5] Downloading FineWeb dataset (vocab=$VOCAB_SIZE)..."
cd "$REPO_DIR"

# Create directories
mkdir -p data/datasets/fineweb10B_sp${VOCAB_SIZE}
mkdir -p data/tokenizers
mkdir -p checkpoints
mkdir -p logs

# Download using the existing script
python3 data/cached_challenge_fineweb.py --variant sp${VOCAB_SIZE} --train-shards 2

echo "Data download complete"

# =============================================================================
# 5. Verify setup
# =============================================================================

echo "[5/5] Verifying setup..."
echo ""
echo "=========================================="
echo "Setup Complete!"
echo "=========================================="
echo "REPO_DIR: $REPO_DIR"
echo "VOCAB_SIZE: $VOCAB_SIZE"
echo ""
echo "To run training:"
echo "  cd $REPO_DIR"
echo "  lightning run job \\"
echo "    --name 'parameter-golf-training' \\"
echo "    --command 'bash run_lightning.sh' \\"
echo "    --machine T4x2 \\"
echo "    --image python3.10"
echo ""
echo "To verify data:"
echo "  ls -la data/datasets/fineweb10B_sp${VOCAB_SIZE}/"
echo ""
