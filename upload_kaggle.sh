#!/bin/bash
set -euo pipefail

echo "Uploading stable training pipeline to Kaggle..."

kaggle kernels push -p /Users/dhruv2mars/dev/github/parameter-golf \
    --accelerator NvidiaTeslaT4

echo ""
echo "Done! Key features:"
echo "- Checkpointing every 5 minutes with resume"
echo "- NaN/Inf detection with graceful recovery"
echo "- 10-minute validation cycles for incremental progress"
echo "- Standard proven architecture (11L x 512d, SP8192)"
echo "- TTT enabled via TTT_ENABLED=1"
echo ""
echo "To run a quick validation (10 min):"
echo "  bash run_train.sh"
echo ""
echo "To run extended training (hours):"
echo "  MAX_WALLCLOCK_SECONDS=36000 ITERATIONS=50000 bash run_train.sh"