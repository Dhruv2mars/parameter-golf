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

### Baseline
- 30-min run (ITERATIONS=7000, MAX_WALLCLOCK=1800) — RUNNING NOW
- Expected ~1.0-1.5 BPB based on 10-min runs at ~2.65 BPB

### Known from prior experiments
- **SOTA architecture**: 11L x 512d x 8H / 4KV with tied embeddings
- **SP8192 tokenizer** is fixed (cannot change)
- **Depth recurrence**: layers 3-5, loop 2x, activates at 35% training
- **QK-Gain 5.25** is proven optimal
- **MLP 4x** with squared LeakyReLU
- **RoPE 16 dims** at base 10000
- **Best 10-min BPB**: ~2.65 (current), SOTA ~2.2-2.3 expected
- **Best Kaggle leaderboard**: ~1.08-1.12 BPB (10-min on H100 equivalent)

### Ideas to Try
- **TTT**: Enable TTT (currently disabled) — may add ~0.05-0.1 BPB
- **PPM**: Not yet implemented — hybrid byte-level compression
- **Architecture exploration**: Wider/shallower vs narrower/deeper
- **Extended warmdown**: Longer learning rate decay
- **Lower learning rate**: May improve convergence

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
