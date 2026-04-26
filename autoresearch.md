# Autoresearch: Parameter Golf - Find Best Config for Sub-1 BPB

## Objective

Optimize a language model under 16MB to achieve the lowest bits-per-byte (BPB) on FineWeb validation.
Each iteration = one 30-minute Kaggle training run.
Winner will be used for extended training (6-12hr) to hit sub-1 BPB.

## Metrics

- **Primary**: `val_bpb` (BPB, lower is better) — the optimization target
- **Secondary**: `best_val_bpb`, `ttt_val_bpb`, `steps`, `compressed_bytes`, `runtime_seconds`

## How to Run

```bash
./autoresearch.sh
```

Outputs `METRIC val_bpb=X` lines. Each run takes ~40 min (30 min training + buffer).

## Files in Scope

- `train_kaggle.py` — main training pipeline (all hyperparameters configurable via env vars)
- `run_train.sh` — runner script (reference, not used by autoresearch)
- `experiments/pg_autoresearch.py` — Kaggle kernel launcher/collector
- `autoresearch.sh` — this loop's driver

## Off Limits

- Do NOT modify the data pipeline (FineWeb downloads, tokenizer)
- Do NOT change vocab_size below 8192 (tokenizer is fixed)
- Do NOT increase ITERATIONS above 7000 for the 30-min loop

## Constraints

- Model must fit under 16MB (checked at submission)
- 30-min wallclock cap per run
- Training must be stable (no NaN/Inf)

## Search Space

### Architecture (env vars in train_kaggle.py)

| Parameter | Current | Search Range | Notes |
|-----------|---------|--------------|-------|
| NUM_LAYERS | 11 | 8-14 | More layers = more capacity but slower |
| MODEL_DIM | 512 | 384-640 | Width of model |
| NUM_HEADS | 8 | 4-16 | Attention heads |
| NUM_KV_HEADS | 4 | 2-8 | KV heads for GQA |
| MLP_MULT | 4 | 2-6 | MLP hidden dim = dim * mult |
| ROPE_DIMS | 16 | 8-32 | RoPE dimensions |
| QK_GAIN_INIT | 5.25 | 3.0-8.0 | QK scaling per head |
| NUM_LOOPS | 2 | 1-4 | Depth recurrence loops |
| LOOP_START | 3 | 2-6 | First layer to loop |
| LOOP_END | 5 | 4-8 | Last layer to loop |

### Hyperparameters

| Parameter | Current | Search Range |
|-----------|---------|--------------|
| MATRIX_LR | 0.001 | 0.0001-0.005 |
| WEIGHT_DECAY | 0.095 | 0.01-0.2 |
| EMA_DECAY | 0.9965 | 0.99-0.999 |
| WARMDOWN_FRAC | 0.72 | 0.6-0.85 |
| GRAD_CLIP_NORM | 1.0 | 0.3-2.0 |
| TIE_EMBED_LR_RATIO | 1.0 | 0.1-2.0 |

### TTT (Test-Time Training)

| Parameter | Current | Search Range |
|-----------|---------|--------------|
| TTT_ENABLED | 0 | 0 or 1 |
| TTT_LR | 0.005 | 0.001-0.02 |
| TTT_EPOCHS | 3 | 1-5 |
| TTT_WARMUP_TOKENS | 32768 | 8192-131072 |

### Quantization

| Parameter | Current | Search Range |
|-----------|---------|--------------|
| MATRIX_CLIP_SIGMAS | 12.85 | 8-20 |
| EMBED_CLIP_SIGMAS | 20.0 | 10-30 |

### Training

| Parameter | Current | Search Range |
|-----------|---------|--------------|
| TRAIN_BATCH_TOKENS | 8192 | 4096-32768 |
| TRAIN_SEQ_LEN | 512 | 256-1024 |
| GRAD_ACCUM_STEPS | 1 | 1-4 |

## What's Been Tried

### Key Discoveries (32+ experiments)

1. **ROPE_DIMS 32 = MAJOR WIN** (+2% over baseline)
   - Sweet spot is 32 dims (tested 8, 16, 24, 28, 32, 48)
   - rope8=2.49, rope24=2.36, rope32=2.33, rope48=2.35
   
2. **EMA_DECAY 0.994 > 0.9965** (+0.2%)
   - Lower decay = faster EMA update helps convergence
   - ema0.994 → 2.326, ema0.9965 → 2.331

3. **GRAD_CLIP_NORM 0.3 = SWEET SPOT** (+0.6%)
   - clip0.2→2.312, clip0.3→2.309, clip0.4→2.313, clip1.0→2.331
   - Tighter clipping helps training stability

4. **QK_GAIN 4.5 > 5.25** (+0.1%)
   - qk4.5 → 2.310, qk5.0 → 2.318, qk6.0 → 2.319
   - Original 5.25 was good but 4.5 is better

5. **Current best config**: rope32 + ema0.994 + clip0.3 + qk4.5 = 2.310 BPB
   - 2.9% better than baseline

### Failed Experiments
- **TTT**: broken (OOM + dtype issues) - needs fix before use
- Wider architecture (640d): too slow, hit wallclock early
- Deeper (13L-480d): underfits
- Seq 1024: fewer steps per time = worse
- More depth loops (3x): slower = worse
- Higher LR (0.002): worse convergence
- Lower WD (0.05): worse
- ROPE 48 dims: worse than rope32
- EMA 0.999: too slow convergence

### Architecture exploration
- 11L x 512d x 8H / 4KV is optimal for time budget
- More layers or width = fewer steps = worse
- GQA (4 KV heads) is optimal

## Current Best Config (30-min runs)
```
ROPE_DIMS=32          # +2% - BIGGEST WIN
EMA_DECAY=0.994       # +0.2%
GRAD_CLIP_NORM=0.3    # +0.6%
QK_GAIN_INIT=4.5      # +0.1%
NUM_LAYERS=11          # keep
MODEL_DIM=512          # keep
```
**Result: 2.310 val_bpb in 30 min (1100 steps)**

## Loop Strategy

1. Start with 30-min baseline (RUNNING NOW)
2. Establish baseline BPB
3. Try architecture changes first (layers, dims, heads) — highest impact
4. Then tune hyperparameters (LR, WD, EMA) — medium impact
5. Enable TTT if disabled — easy win if it works
6. Explore PPM integration if time permits
7. Keep iterating until confident in best config
8. Run extended training (6-12hr) on winning config

## Notes

- T4 is ~3x slower than H100, so 30-min on T4 ≈ 10-min on H100
- Sub-1 BPB likely requires: extended training (6hr+) + TTT + good architecture
- Each run takes ~40 min (30 min training + 10 min queue/overhead)
- Loop runs indefinitely until manually stopped
