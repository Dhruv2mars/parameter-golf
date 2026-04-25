#!/bin/bash
# Parameter Golf - Kaggle 2xT4 Training Runner
# Optimized for stable, long-running sessions with checkpointing.

set -euo pipefail

# Fast syntax check
python3 -m py_compile train_kaggle.py 2>/dev/null || { 
    echo "SYNTAX ERROR in train_kaggle.py"
    exit 1
}

mkdir -p logs checkpoints

# =============================================================================
# ENVIRONMENT: All hyperparameters configurable via env vars
# =============================================================================

# Run identification
export RUN_ID=${RUN_ID:-"t4-ttt-test"}

# Training (10 min for quick validation, extend for long runs)
export ITERATIONS=${ITERATIONS:-2500}
export MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-600}
export WARMDOWN_FRAC=${WARMDOWN_FRAC:-0.72}
export MIN_LR_RATIO=${MIN_LR_RATIO:-0.1}

# Batch (effective batch = BATCH_TOKENS * GRAD_ACCUM)
export TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS:-8192}
export TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN:-512}
export GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-1}  # 1 for quick feedback, 4 for larger effective batch

# Model architecture (proven SOTA config)
export VOCAB_SIZE=${VOCAB_SIZE:-8192}
export NUM_LAYERS=${NUM_LAYERS:-11}
export MODEL_DIM=${MODEL_DIM:-512}
export NUM_HEADS=${NUM_HEADS:-8}
export NUM_KV_HEADS=${NUM_KV_HEADS:-4}
export MLP_MULT=${MLP_MULT:-4}

# Depth recurrence (layers 3-5 loop twice after 35% of training)
export NUM_LOOPS=${NUM_LOOPS:-2}
export LOOP_START=${LOOP_START:-3}
export LOOP_END=${LOOP_END:-5}
export ENABLE_LOOPING_AT=${ENABLE_LOOPING_AT:-0.35}

# QK-Gain (5.25 is proven optimal)
export QK_GAIN_INIT=${QK_GAIN_INIT:-5.25}

# RoPE
export ROPE_DIMS=${ROPE_DIMS:-16}
export ROPE_BASE=${ROPE_BASE:-10000.0}

# Optimizer (AdamW, tuned for T4)
export MATRIX_LR=${MATRIX_LR:-0.001}
export WEIGHT_DECAY=${WEIGHT_DECAY:-0.095}
export BETA1=${BETA1:-0.9}
export BETA2=${BETA2:-0.95}
export GRAD_CLIP_NORM=${GRAD_CLIP_NORM:-1.0}
export EMA_DECAY=${EMA_DECAY:-0.9965}

# Quantization (GPTQ SDClip)
export MATRIX_CLIP_SIGMAS=${MATRIX_CLIP_SIGMAS:-12.85}
export EMBED_CLIP_SIGMAS=${EMBED_CLIP_SIGMAS:-20.0}

# TTT (Test-Time Training) - enables via TTT_ENABLED=1
export TTT_ENABLED=${TTT_ENABLED:-1}
export TTT_LR=${TTT_LR:-0.005}
export TTT_EPOCHS=${TTT_EPOCHS:-3}
export TTT_WARMUP_TOKENS=${TTT_WARMUP_TOKENS:-32768}

# Checkpointing
export CHECKPOINT_DIR=${CHECKPOINT_DIR:-"checkpoints"}

# =============================================================================
# PRINT CONFIG
# =============================================================================

echo "=========================================="
echo "Parameter Golf - T4 Training"
echo "=========================================="
echo "RUN_ID: $RUN_ID"
echo "ITERATIONS: $ITERATIONS"
echo "MAX_TIME: ${MAX_WALLCLOCK_SECONDS}s"
echo "EFFECTIVE_BATCH: $((TRAIN_BATCH_TOKENS * GRAD_ACCUM_STEPS)) tokens"
echo "MODEL: ${NUM_LAYERS}L x ${MODEL_DIM}d x ${NUM_HEADS}H / ${NUM_KV_HEADS}KV"
echo "LOOP: layers ${LOOP_START}-${LOOP_END}, ${NUM_LOOPS}x"
echo "QK-GAIN: $QK_GAIN_INIT"
echo "TTT: $TTT_ENABLED (${TTT_EPOCHS} epochs)"
echo "=========================================="

# =============================================================================
# RUN
# =============================================================================

python3 train_kaggle.py 2>&1 | tee "logs/${RUN_ID}.log"

# =============================================================================
# EXTRACT METRICS
# =============================================================================

if [ -f "logs/${RUN_ID}.log" ]; then
    echo ""
    echo "=========================================="
    echo "RESULTS"
    echo "=========================================="
    
    # Extract from log
    VAL_BPB=$(grep -E "final_val_bpb|best_val_bpb" "logs/${RUN_ID}.log" | tail -1 | sed 's/.*=\([0-9.]*\).*/\1/' || echo "N/A")
    ARTIFACT=$(grep -E "Compressed artifact:|artifact_bytes" "logs/${RUN_ID}.log" | tail -1 | grep -oE '[0-9]+,' | tr -d ',' || echo "N/A")
    STEPS=$(grep -E "total_steps" "logs/${RUN_ID}.log" | tail -1 | sed 's/.*=\([0-9]*\).*/\1/' || echo "N/A")
    TIME=$(grep -E "total_time_s" "logs/${RUN_ID}.log" | tail -1 | sed 's/.*=\([0-9.]*\).*/\1/' || echo "N/A")
    
    echo "val_bpb: ${VAL_BPB}"
    echo "artifact_bytes: ${ARTIFACT}"
    echo "steps: ${STEPS}"
    echo "time_seconds: ${TIME}"
    
    # METRIC lines for auto-parsing
    grep "^METRIC " "logs/${RUN_ID}.log" | tail -10
fi