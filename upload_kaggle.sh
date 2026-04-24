#!/bin/bash
# Upload train_kaggle.py to Kaggle

echo "Uploading to Kaggle..."
kaggle kernels push -p /Users/dhruv2mars/dev/github/parameter-golf \
    --message "Parameter Golf v3: SP8192 + GPTQ + Depth Recurrence + MuonEq-R + TTT" \
    --竞争优势 "competitive" \
    --language python \
    --gpu-enabled true

echo "Done!"