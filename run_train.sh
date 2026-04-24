#!/bin/bash
# Parameter Golf - Kaggle 2xT4 Training Runner
# Optimized for Kaggle 2xT4 GPUs (each T4: 16GB VRAM, ~65 TFLOPs)

set -euo pipefail

# Fast pre-check
python3 -m py_compile train_kaggle.py 2>/dev/null || { echo "SYNTAX ERROR"; exit 1; }

mkdir -p logs

# T4-Optimized defaults (reduced from H100 settings)
# T4 is ~3x slower than H100 SXM, so we adjust accordingly
RUN_ID=${RUN_ID:-"t4-exp-$(date +%s)"}
VOCAB_SIZE=${VOCAB_SIZE:-8192}
NUM_LAYERS=${NUM_LAYERS:-11}
MODEL_DIM=${MODEL_DIM:-512}
MLP_MULT=${MLP_MULT:-4}

# Training - T4 optimized (less iterations due to slower GPU)
ITERATIONS=${ITERATIONS:-2500}
MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-2700}
WARMDOWN_FRAC=${WARMDOWN_FRAC:-0.72}  # SOTA uses 0.72

# TTT - essential for SOTA scores
TTT_ENABLED=${TTT_ENABLED:-1}
TTT_LR=${TTT_LR:-0.005}
TTT_EPOCHS=${TTT_EPOCHS:-3}

# Hyperparameters - SOTA tuned
QK_GAIN_INIT=${QK_GAIN_INIT:-5.25}  # SOTA uses 5.25
WEIGHT_DECAY=${WEIGHT_DECAY:-0.095}  # SOTA uses 0.095
MUON_MOMENTUM=${MUON_MOMENTUM:-0.99}
MATRIX_LR=${MATRIX_LR:-0.022}  # SOTA uses 0.022
EMA_DECAY=${EMA_DECAY:-0.9965}  # SOTA uses 0.9965

# T4 memory constraints
TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS:-8192}  # Half of H100 (16K -> 8K)
TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN:-512}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}  # More accumulation

# Model architecture
ROPE_DIMS=${ROPE_DIMS:-16}
NUM_LOOPS=${NUM_LOOPS:-2}
LOOP_START=${LOOP_START:-4}
LOOP_END=${LOOP_END:-5}
ENABLE_LOOPING_AT=${ENABLE_LOOPING_AT:-0.35}  # SOTA activates at 0.35

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

# Run training with torchrun for multi-GPU support
python3 train_kaggle.py 2>&1 | tee "logs/${RUN_ID}.log"

# Extract metrics
if [ -f "logs/${RUN_ID}.log" ]; then
    VAL_BPB=$(grep -E "Best val_bpb:" "logs/${RUN_ID}.log" | tail -1 | awk '{print $3}')
    TTT_BPB=$(grep -E "TTT val_bpb:" "logs/${RUN_ID}.log" | tail -1 | awk '{print $3}')
    COMPRESSED=$(grep -E "Compressed:" "logs/${RUN_ID}.log" | tail -1 | awk '{print $2}')
    PARAMS=$(grep -E "Params:" "logs/${RUN_ID}.log" | tail -1 | awk '{print $2}')
    TRAIN_TIME=$(grep -E "Training complete" "logs/${RUN_ID}.log" | tail -1 | awk '{print $NF}')
    
    echo ""
    echo "=========================================="
    echo "RESULTS"
    echo "=========================================="
    echo "val_bpb: ${VAL_BPB:-N/A}"
    echo "ttt_val_bpb: ${TTT_BPB:-N/A}"
    echo "compressed_bytes: ${COMPRESSED:-N/A}"
    echo "model_params: ${PARAMS:-N/A}"
    echo "train_time: ${TRAIN_TIME:-N/A}"
    
    # Output for autorun parsing
    if [ ! -z "$VAL_BPB" ]; then
        echo ""
        echo "METRIC val_bpb=${VAL_BPB}"
        echo "METRIC ttt_val_bpb=${TTT_BPB:-0}"
        echo "METRIC compressed_bytes=${COMPRESSED:-0}"
        echo "METRIC model_params=${PARAMS:-0}"
        echo "METRIC train_time_seconds=${TRAIN_TIME:-0}"
    fi
fi
