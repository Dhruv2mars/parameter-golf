#!/bin/bash
# =============================================================================
# Lightning AI Auto-Start Script - Run on Session Launch
# =============================================================================
# Put this in your Studio's ~/.studiorc to auto-start training on session start.
# Also handles auto-resume from checkpoints.
# =============================================================================

# Exit if not in the right directory
cd /teamspace/studios/this_studio/parameter-golf 2>/dev/null || { echo "Repo not found at /teamspace/studios/this_studio/parameter-golf"; exit 1; }

echo "=========================================="
echo "Auto-Start: Parameter Golf Training"
echo "START_TIME: $(date)"
echo "=========================================="

# Check for checkpoint
LATEST_CKPT=$(ls -t checkpoints/*/ckpt_step_*.pt 2>/dev/null | head -1)

if [ -n "$LATEST_CKPT" ] && [ -f "$LATEST_CKPT" ]; then
    echo "Found checkpoint: $LATEST_CKPT"
    echo "Auto-resuming training..."
    echo ""
    bash run_lightning.sh
else
    echo "No checkpoint found"
    echo "Starting fresh training run..."
    echo ""
    bash run_lightning.sh
fi

echo ""
echo "=========================================="
echo "Session training complete"
echo "END_TIME: $(date)"
echo "=========================================="
