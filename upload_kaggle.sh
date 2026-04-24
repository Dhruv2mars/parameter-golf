#!/bin/bash
set -euo pipefail

# Upload the script kernel described by kernel-metadata.json.

echo "Uploading to Kaggle..."
kaggle kernels push -p /Users/dhruv2mars/dev/github/parameter-golf \
    --accelerator NvidiaTeslaT4

echo ""
echo "Done! Check Kaggle for the new kernel version."
echo ""
echo "Key T4 optimizations:"
echo "- Batch tokens: 8192"
echo "- Grad accum: 4 (effective batch = 32K)"
echo "- Iterations: 2500 (reduced for slower GPU)"
echo "- Wallclock: 2700s (with buffer for TTT)"
echo "- Hyperparameters: AdamW LR=0.0003, WD=0.095, EMA=0.9965"
