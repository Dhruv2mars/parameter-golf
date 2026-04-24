# Parameter Golf - Dhruv2Mars Progress

## Current Status

**Last Run (exp6_baseline):**
- val_bpb: **2.60** (target: <1.22 baseline, goal: ~1.08 SOTA)
- model: 6L×384 GQA, 5.3M params, Muon+WD
- compressed: 2.48MB
- Status: Still improving at step 1170

## Leaderboard

| Rank | Score | Author | Key Techniques |
|------|-------|--------|----------------|
| #1 | **1.0810** | bigbag | SP8192 + 3-Layer Recurrence + Parallel Residuals + Legal TTT |
| #2 | 1.0822 | aryanbhosale | Parallel Residuals + Score-First TTT |
| #3 | 1.0828 | dexhunter | QK-Gain 5 + Legal TTT |
| Baseline | 1.2244 | OpenAI | 9L 512dim 1024vocab |
| **Us** | **2.60** | dhruv2mars | 6L×384 GQA + Muon |

## Key Gaps

1. **Vocabulary** - We're at 1024, SOTA uses 8192
2. **Quantization** - We have none, SOTA uses GPTQ int6/int8 with SDClip
3. **Depth Recurrence** - Not implemented
4. **Training iterations** - Only 1170 steps, model still improving
5. **Parallel residuals** - Not implemented

## Our Setup

- **Kaggle 2xT4**: 30hrs/week quota, ~2xP100 GPU power
- **M1 Mac Mini**: 16GB, for development
- **No time limit** on Kaggle (unlike 10min on 8×H100)

## Next Steps

### Phase 1: Quick Improvements
1. Increase vocab to 4096/8192
2. Extend training iterations to 4000+
3. Add int8 quantization

### Phase 2: Catch Baseline
4. Implement depth recurrence (loop layers 4-5)
5. Add GPTQ-SDClip quantization
6. Implement parallel residuals

### Phase 3: SOTA
7. Implement TTT (Test-Time Training)
8. Multi-seed averaging
9. Hyperparameter tuning

## Files

- `train_kaggle.py` - Main training script for Kaggle
- `upload_kaggle.sh` - Script to upload to Kaggle
- `kaggle_notebook.json` - Kaggle notebook metadata

## Running

```bash
# Upload to Kaggle
bash upload_kaggle.sh

# Or manually
kaggle kernels push -p . --message "Parameter Golf Training"
```

## Environment Variables

```bash
# Key hyperparameters
VOCAB_SIZE=8192          # SOTA uses 8192
NUM_LAYERS=11            # SOTA uses 11
MODEL_DIM=512
MLP_MULT=4              # SOTA uses 4x expansion
MATRIX_LR=0.022         # Tuned learning rate
WEIGHT_DECAY=0.085
MUON_MOMENTUM=0.99
ITERATIONS=4000
MAX_WALLCLOCK_SECONDS=3000  # 50 minutes on Kaggle
TTT_ENABLED=1
```