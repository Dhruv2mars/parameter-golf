#!/bin/bash
# Parameter Golf - Kaggle Training Runner
# This script runs on Kaggle with GPU

# Fast pre-check
python3 -m py_compile train_kaggle.py 2>/dev/null || { echo "SYNTAX ERROR"; exit 1; }

mkdir -p logs

# SOTA defaults for first intensive run
RUN_ID=${RUN_ID:-"exp7-sota"}
VOCAB_SIZE=${VOCAB_SIZE:-8192}
NUM_LAYERS=${NUM_LAYERS:-11}
MODEL_DIM=${MODEL_DIM:-512}
MLP_MULT=${MLP_MULT:-4}
ITERATIONS=${ITERATIONS:-4000}
MAX_WALLCLOCK_SECONDS=${MAX_WALLCLOCK_SECONDS:-3600}
WARMDOWN_FRAC=${WARMDOWN_FRAC:-0.5}
TTT_ENABLED=${TTT_ENABLED:-1}
QK_GAIN_INIT=${QK_GAIN_INIT:-4.0}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.085}
MUON_MOMENTUM=${MUON_MOMENTUM:-0.99}
MATRIX_LR=${MATRIX_LR:-0.022}
TRAIN_BATCH_TOKENS=${TRAIN_BATCH_TOKENS:-16384}
TRAIN_SEQ_LEN=${TRAIN_SEQ_LEN:-512}
GRAD_ACCUM_STEPS=${GRAD_ACCUM_STEPS:-4}
ROPE_DIMS=${ROPE_DIMS:-16}
NUM_LOOPS=${NUM_LOOPS:-2}
LOOP_START=${LOOP_START:-4}
LOOP_END=${LOOP_END:-5}
ENABLE_LOOPING_AT=${ENABLE_LOOPING_AT:-0.5}

echo "=== Parameter Golf Training (SOTA Config) ==="
echo "RUN_ID: $RUN_ID"
echo "VOCAB_SIZE: $VOCAB_SIZE"
echo "NUM_LAYERS: $NUM_LAYERS"
echo "MLP_MULT: $MLP_MULT"
echo "ITERATIONS: $ITERATIONS"
echo "MAX_WALLCLOCK: $MAX_WALLCLOCK_SECONDS seconds"

# Run training
python3 train_kaggle.py 2>&1 | tee "logs/${RUN_ID}.log"

# Extract metrics
if [ -f "logs/${RUN_ID}.log" ]; then
    VAL_BPB=$(grep "Best val_bpb:" "logs/${RUN_ID}.log" | tail -1 | awk '{print $3}')
    COMPRESSED=$(grep "Compressed:" "logs/${RUN_ID}.log" | tail -1 | awk '{print $2}')
    PARAMS=$(grep "Params:" "logs/${RUN_ID}.log" | tail -1 | awk '{print $2}')
    TRAIN_TIME=$(grep "Training complete" "logs/${RUN_ID}.log" | tail -1 | awk '{print $NF}')
    
    echo ""
    echo "=== RESULTS ==="
    echo "val_bpb: $VAL_BPB"
    echo "compressed_bytes: $COMPRESSED"
    echo "model_params: $PARAMS"
    echo "train_time: $TRAIN_TIME"
    
    # Output for autorun parsing
    if [ ! -z "$VAL_BPB" ]; then
        echo "METRIC val_bpb=$VAL_BPB"
        echo "METRIC compressed_bytes=${COMPRESSED:-0}"
        echo "METRIC model_params=${PARAMS:-0}"
        echo "METRIC train_time_seconds=${TRAIN_TIME:-0}"
    fi
fi