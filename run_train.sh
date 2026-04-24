#!/bin/bash
# Parameter Golf - Kaggle 2xT4 Training Runner
# Optimized for Kaggle 2xT4 GPUs (each T4: 16GB VRAM, ~65 TFLOPs)

set -euo pipefail

# Fast pre-check
python3 -m py_compile train_kaggle.py 2>/dev/null || { echo "SYNTAX ERROR"; exit 1; }

mkdir -p logs

# T4-optimized defaults. Current script is single-process; 2xT4 DDP is not wired yet.
# T4 is ~3x slower than H100 SXM, so we adjust accordingly
export RUN_ID=${RUN_ID:-"t4-exp-$(date +%s)"}
export VOCAB_SIZE=${VOCAB_SIZE:-8192}
export NUM_LAYERS=${NUM_LAYERS:-11}
export MODEL_DIM=${MODEL_DIM:-512}
export MLP_MULT=${MLP_MULT:-4}

# Training - T4 optimized (less iterations due to slower GPU)
export ITERATIONS=${ITERATIONS:-2500}
export MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-2700}
export WARMDOWN_FRAC=${WARMDOWN_FRAC:-0.72}

# TTT - essential for SOTA scores
export TTT_ENABLED=${TTT_ENABLED:-1}
export TTT_LR=${TTT_LR:-0.005}
export TTT_EPOCHS=${TTT_EPOCHS:-3}

# Hyperparameters - SOTA tuned
export QK_GAIN_INIT=${QK_GAIN_INIT:-2.0}
export WEIGHT_DECAY=${WEIGHT_DECAY:-0.095}
export MUON_MOMENTUM=${MUON_MOMENTUM:-0.99}
export MATRIX_LR=${MATRIX_LR:-0.0003}
export EMA_DECAY=${EMA_DECAY:-0.9965}

# T4 memory constraints
export TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS:-8192}
export TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN:-512}
export GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}

# Model architecture
export ROPE_DIMS=${ROPE_DIMS:-16}
export NUM_LOOPS=${NUM_LOOPS:-2}
export LOOP_START=${LOOP_START:-4}
export LOOP_END=${LOOP_END:-5}
export ENABLE_LOOPING_AT=${ENABLE_LOOPING_AT:-0.35}

echo "=========================================="
echo "Parameter Golf - Kaggle 2xT4 Training"
echo "=========================================="
echo "RUN_ID: $RUN_ID"
echo "VOCAB_SIZE: $VOCAB_SIZE"
echo "NUM_LAYERS: $NUM_LAYERS"
echo "ITERATIONS: $ITERATIONS (T4-optimized)"
echo "MAX_WALLCLOCK: $MAX_WALLCLOCK_SECONDS seconds"
echo "TTT_ENABLED: $TTT_ENABLED"
echo "QK_GAIN_INIT: $QK_GAIN_INIT"
echo "=========================================="

# Run training. Multi-GPU needs real DDP wiring before enabling torchrun.
python3 train_kaggle.py 2>&1 | tee "logs/${RUN_ID}.log"

# Extract metrics
if [ -f "logs/${RUN_ID}.log" ]; then
    VAL_BPB=$(grep -E "Best val_bpb:" "logs/${RUN_ID}.log" | tail -1 | awk '{print $NF}' || true)
    TTT_BPB=$(grep -E "TTT val_bpb:" "logs/${RUN_ID}.log" | tail -1 | awk '{print $NF}' || true)
    COMPRESSED=$(grep -E "Compressed:" "logs/${RUN_ID}.log" | tail -1 | awk '{print $2}' || true)
    PARAMS=$(grep -E "Params:" "logs/${RUN_ID}.log" | tail -1 | awk '{print $2}' || true)
    
    echo ""
    echo "=========================================="
    echo "RESULTS"
    echo "=========================================="
    echo "val_bpb: ${VAL_BPB:-N/A}"
    echo "ttt_val_bpb: ${TTT_BPB:-N/A}"
    echo "compressed_bytes: ${COMPRESSED:-N/A}"
    echo "model_params: ${PARAMS:-N/A}"
    echo "train_time: N/A"
    
    # Output for autorun parsing
    if [ -n "$VAL_BPB" ]; then
        echo ""
        echo "METRIC val_bpb=${VAL_BPB}"
        echo "METRIC ttt_val_bpb=${TTT_BPB:-0}"
        echo "METRIC compressed_bytes=${COMPRESSED:-0}"
        echo "METRIC model_params=${PARAMS:-0}"
        echo "METRIC train_time_seconds=0"
    fi
fi
