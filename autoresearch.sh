#!/bin/bash
set -euo pipefail

# Autoresearch: Parameter Golf Training
# Runs training and extracts val_bpb metric

# Fast pre-check: syntax errors
python3 -m py_compile train_kaggle.py 2>/dev/null || { echo "SYNTAX ERROR"; exit 1; }

# Output directory
mkdir -p logs

# Run training with current hyperparameters
RUN_ID=${RUN_ID:-"auto-$(date +%s)"}
ITERATIONS=${ITERATIONS:-4000}
VOCAB_SIZE=${VOCAB_SIZE:-8192}
NUM_LAYERS=${NUM_LAYERS:-11}
MLP_MULT=${MLP_MULT:-4}
MAX_WALLCLOCK=${MAX_WALLCLOCK_SECONDS:-3600}
TTT_ENABLED=${TTT_ENABLED:-1}

echo "=== Parameter Golf Training ==="
echo "RUN_ID: $RUN_ID"
echo "VOCAB_SIZE: $VOCAB_SIZE"
echo "NUM_LAYERS: $NUM_LAYERS"
echo "ITERATIONS: $ITERATIONS"

# Run training
python3 train_kaggle.py 2>&1 | tee "logs/${RUN_ID}.log"

# Extract metrics from log
if [ -f "logs/${RUN_ID}.log" ]; then
    # Get final val_bpb
    VAL_BPB=$(grep "Best val_bpb:" "logs/${RUN_ID}.log" | tail -1 | awk '{print $3}')
    
    # Get compressed size
    COMPRESSED=$(grep "Compressed:" "logs/${RUN_ID}.log" | tail -1 | awk '{print $2}')
    
    # Get model params
    PARAMS=$(grep "Params:" "logs/${RUN_ID}.log" | tail -1 | awk '{print $2}')
    
    # Get train time
    TRAIN_TIME=$(grep "Training complete" "logs/${RUN_ID}.log" | tail -1 | awk '{print $NF}')
    
    echo ""
    echo "=== METRICS ==="
    echo "METRIC val_bpb=${VAL_BPB:-0}"
    echo "METRIC compressed_bytes=${COMPRESSED:-0}"
    echo "METRIC model_params=${PARAMS:-0}"
    echo "METRIC train_time_seconds=${TRAIN_TIME:-0}"
    
    # Save metrics for reference
    echo "val_bpb=$VAL_BPB" > "logs/${RUN_ID}.metrics"
else
    echo "METRIC val_bpb=0"
    echo "ERROR: No log file generated"
    exit 1
fi