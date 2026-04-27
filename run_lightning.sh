#!/bin/bash
# =============================================================================
# Lightning AI Training Script - 4 Hour Session Wrapper
# =============================================================================
# Handles the 4-hour session reset by:
# 1. Checking for existing checkpoint on start
# 2. Running training with frequent checkpointing (every 3 hours)
# 3. Saving state before session expires
# 4. Will auto-restart from checkpoint in next session
# =============================================================================

set -euo pipefail

# =============================================================================
# CONFIGURATION - Modify these for different runs
# =============================================================================

# Model architecture (proven config from experiments)
export VOCAB_SIZE=${VOCAB_SIZE:-8192}
export NUM_LAYERS=${NUM_LAYERS:-11}
export MODEL_DIM=${MODEL_DIM:-512}
export NUM_HEADS=${NUM_HEADS:-8}
export NUM_KV_HEADS=${NUM_KV_HEADS:-4}
export MLP_MULT=${MLP_MULT:-4}

# Training settings
export ITERATIONS=${ITERATIONS:-25000}        # Extended training
export MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-14000}  # ~4 hours with buffer
export TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS:-8192}
export TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN:-512}
export GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}

# Depth recurrence (layers 3-5 loop twice after 35% of training)
export NUM_LOOPS=${NUM_LOOPS:-2}
export LOOP_START=${LOOP_START:-3}
export LOOP_END=${LOOP_END:-5}
export ENABLE_LOOPING_AT=${ENABLE_LOOPING_AT:-0.35}

# Optimizer (best config from experiments)
export MATRIX_LR=${MATRIX_LR:-0.0008}
export WEIGHT_DECAY=${WEIGHT_DECAY:-0.095}
export BETA1=${BETA1:-0.9}
export BETA2=${BETA2:-0.95}
export EMA_DECAY=${EMA_DECAY:-0.994}
export GRAD_CLIP_NORM=${GRAD_CLIP_NORM:-0.3}
export WARMDOWN_FRAC=${WARMDOWN_FRAC:-0.80}

# RoPE (proven: 32 dims is best)
export ROPE_DIMS=${ROPE_DIMS:-32}
export ROPE_BASE=${ROPE_BASE:-10000.0}
export QK_GAIN_INIT=${QK_GAIN_INIT:-4.0}

# Quantization
export MATRIX_CLIP_SIGMAS=${MATRIX_CLIP_SIGMAS:-12.85}
export EMBED_CLIP_SIGMAS=${EMBED_CLIP_SIGMAS:-20.0}

# Paths
export DATA_PATH=${DATA_PATH:-"/workspace/parameter-golf/data/datasets/fineweb10B_sp${VOCAB_SIZE}"}
export TOKENIZER_PATH=${TOKENIZER_PATH:-"/workspace/parameter-golf/data/tokenizers/fineweb_${VOCAB_SIZE}_bpe.model"}
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-"/workspace/parameter-golf/checkpoints"}
export CHECKPOINT_EVERY_SECONDS=${CHECKPOINT_EVERY_SECONDS:-10800}  # 3 hours - reduce overhead

# Run identification
export RUN_ID=${RUN_ID:-"lightning-$(date +%Y%m%d-%H%M%S)"}
export SEED=${SEED:-1337}

# =============================================================================
# DIRECTORIES
# =============================================================================

cd /workspace/parameter-golf
mkdir -p logs checkpoints

# =============================================================================
# LOGGING
# =============================================================================

LOG_FILE="logs/${RUN_ID}.log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "=========================================="
echo "Parameter Golf - Lightning AI Training"
echo "=========================================="
echo "RUN_ID: $RUN_ID"
echo "START_TIME: $(date)"
echo "ITERATIONS: $ITERATIONS"
echo "MAX_TIME: ${MAX_WALLCLOCK_SECONDS}s"
echo "EFFECTIVE_BATCH: $((TRAIN_BATCH_TOKENS * GRAD_ACCUM_STEPS)) tokens"
echo "MODEL: ${NUM_LAYERS}L x ${MODEL_DIM}d x ${NUM_HEADS}H / ${NUM_KV_HEADS}KV"
echo "SEQ_LEN: $TRAIN_SEQ_LEN"
echo "VOCAB_SIZE: $VOCAB_SIZE"
echo "ROPE_DIMS: $ROPE_DIMS"
echo "EMA_DECAY: $EMA_DECAY"
echo "=========================================="
echo ""

# =============================================================================
# CHECK FOR EXISTING CHECKPOINT (AUTO-RESUME)
# =============================================================================

LATEST_CKPT=$(ls -t ${CHECKPOINT_DIR}/*/ckpt_step_*.pt 2>/dev/null | head -1)

if [ -n "$LATEST_CKPT" ] && [ -f "$LATEST_CKPT" ]; then
    echo "Found checkpoint: $LATEST_CKPT"
    echo "Will attempt to resume from this checkpoint"
    # The train_kaggle.py handles resume automatically
else
    echo "No checkpoint found, starting fresh"
fi

# =============================================================================
# RUN TRAINING
# =============================================================================

echo "Starting training at $(date)..."
echo ""

python3 train_kaggle.py 2>&1

# =============================================================================
# DONE
# =============================================================================

echo ""
echo "=========================================="
echo "Training Complete!"
echo "END_TIME: $(date)"
echo "LOG_FILE: $LOG_FILE"
echo "=========================================="

# Extract final metrics
if [ -f "$LOG_FILE" ]; then
    echo ""
    echo "Final Metrics:"
    grep -E "final_val_bpb|best_val_bpb|total_steps" "$LOG_FILE" | tail -5
fi
