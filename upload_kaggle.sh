#!/bin/bash
# Upload train_kaggle.py to Kaggle

echo "Uploading to Kaggle..."
kaggle kernels push -p /Users/dhruv2mars/dev/github/parameter-golf \
    --message "Parameter Golf v4: T4-Optimized SOTA (SP8192 + GPTQ + Depth Recurrence + MuonEq-R + TTT)" \
    --competition "competitive" \
    --language python \
    --gpu-enabled true

echo ""
echo "Done! Check Kaggle for the new kernel version."
echo ""
echo "Key T4 optimizations:"
echo "- Batch tokens: 8192 (half of H100)"
echo "- Grad accum: 4 (effective batch = 32K)"
echo "- Iterations: 2500 (reduced for slower GPU)"
echo "- Wallclock: 2700s (with buffer for TTT)"
echo "- SOTA hyperparameters: WD=0.095, LR=0.022, EMA=0.9965, QK=5.25"
