# Autoresearch: Parameter Golf

## Objective
Train the best language model under 16MB constraint. Goal: reach top 5 on the OpenAI parameter-golf leaderboard (~1.09 BPB or better), then iterate to beat #1 (currently 1.0810 BPB).

Metric: **val_bpb** (bits per byte, lower is better)

## Metrics
- **Primary**: val_bpb (BPB, lower is better) — compression-based evaluation on FineWeb validation set
- **Secondary**: 
  - compressed_size_bytes — must stay under 16,000,000
  - train_time_seconds — for iteration speed
  - model_params — parameter count

## How to Run
```bash
# Phase 1: Intensive baseline run (reaches top 5)
./autoresearch.sh

# Or manually:
RUN_ID=exp7 VOCAB_SIZE=8192 NUM_LAYERS=11 MLP_MULT=4 \
  ITERATIONS=4000 MAX_WALLCLOCK_SECONDS=3600 \
  python train_kaggle.py
```

## Files in Scope
| File | Purpose |
|------|---------|
| `train_kaggle.py` | Main training script with all SOTA techniques |
| `train_gpt.py` | Original OpenAI reference baseline |
| `data/` | FineWeb dataset and tokenizer |
| `records/` | Submission records from community |

## Key Techniques (SOTA)
1. **SP8192 vocabulary** — better tokenization than 1024
2. **GPTQ-SDClip quantization** — int6 matrices, int8 embeddings
3. **Depth recurrence** — loop layers 4-5 for virtual depth
4. **MuonEq-R optimizer** — row-normalized Muon
5. **Parallel residuals** — from layer 7+
6. **LeakyReLU² activation** — from SOTA submissions
7. **TTT (Test-Time Training)** — score-first adaptation at eval

## Architecture (Current)
- 11L × 512dim × 8H / 4KV (GQA)
- MLP 4x expansion
- Tied embeddings
- RoPE (partial, 16/64 dims)
- QK-Gain per-head scaling
- Logit softcap 30.0

## Constraints
1. **16MB artifact cap** — model weights + code must fit
2. **Training must complete** — no crashes allowed
3. **val_bpb calculation** — must use correct FineWeb BPB formula
4. **No external data at eval** — all data must be in artifact

## Off Limits
- `train_gpt.py` — reference only, don't modify
- `data/` contents — don't re-download

## What's Been Tried

### exp1-6 (local baseline)
| Exp | Model | Opt | Steps | val_bpb | Notes |
|-----|-------|-----|-------|---------|-------|
| 1 | 3L×256 | AdamW | 20 | 3.79 | Too small |
| 2 | 5L×384 GQA | AdamW | 100 | 3.48 | |
| 3 | 5L×384 GQA | AdamW | 992 | 3.36 | |
| 4 | 5L×384 GQA | AdamW | 1013 | 3.35 | |
| 5 | 5L×384 GQA | Muon+WD | 290 | 2.71 | Muon breakthrough! |
| 6 | 6L×384 GQA | Muon+WD | 1170 | 2.60 | Still improving |

### Key Findings
- Muon + weight decay = massive improvement (3.35 → 2.71)
- 6 layers slightly better than 5 (2.71 → 2.60)
- Model still improving at step 1170 — needs more training
- 1024 vocab is a bottleneck

### Critical Gaps vs SOTA
1. **Vocabulary** — 1024 vs 8192 (huge gap)
2. **No quantization** — storing fp16, SOTA uses int6/int8
3. **No depth recurrence** — SOTA loops layers 4-5
4. **Training cut short** — wallclock cap hit at 818s
5. **No parallel residuals**

## Current Status
- **Current**: 2.60 BPB (way behind baseline 1.22)
- **Target**: <1.09 BPB (top 5)
- **SOTA**: 1.0810 BPB

## Priority
1. First run: SP8192 + 11L + MuonEq-R + depth recurrence (should reach ~1.15-1.20)
2. Second run: Add GPTQ-SDClip quantization
3. Third run: Add TTT + parallel residuals
4. Iterate to beat SOTA