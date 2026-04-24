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

## Project Setup (Autoresearch-Ready)

```
parameter-golf/
├── train_kaggle.py      # SOTA training script (685 lines)
├── run_train.sh         # Training runner for Kaggle
├── autoresearch.sh      # Autoresearch experiment runner
├── autoresearch.md      # Experiment documentation
├── autoresearch.ideas.md # Ideas backlog
├── autoresearch.config.json # Config (working dir, max iterations)
├── PROGRESS.md          # This file
├── upload_kaggle.sh     # Upload script
└── kaggle_notebook.json # Notebook metadata
```

## Next Action

**Push to Kaggle and run the first intensive training:**

```bash
# Upload
kaggle kernels push -p . --message "Parameter Golf v3: SP8192 + Depth Recurrence + MuonEq-R"

# Or use the upload script
bash upload_kaggle.sh
```

## Environment Variables for SOTA Run

```bash
VOCAB_SIZE=8192          # SOTA uses 8192
NUM_LAYERS=11            # SOTA uses 11
MODEL_DIM=512
MLP_MULT=4              # SOTA uses 4x expansion
MATRIX_LR=0.022
WEIGHT_DECAY=0.085
MUON_MOMENTUM=0.99
ITERATIONS=4000
MAX_WALLCLOCK_SECONDS=3600
TTT_ENABLED=1
QK_GAIN_INIT=4.0
```

## Files in Scope (Don't Modify)

- `train_gpt.py` - Original OpenAI reference (read-only)
- `data/` - FineWeb dataset (read-only)