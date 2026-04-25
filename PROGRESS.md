# Parameter Golf Progress

## Goal

Train a language model achieving sub-1 BPB on 2xT4 GPUs with extended training.
Use incremental 10-minute validation cycles for hill climbing toward the target.

## Constraints

- **Hardware:** 2x Tesla T4 (16GB each)
- **Artifact:** Under 16,000,000 bytes
- **Training:** 30 hours/week quota, 12 hours per session
- **Strategy:** Incremental 10-min runs → validate → improve → repeat

## Architecture (Proven SOTA Config)

```
- Model: 11L x 512d x 8H / 4KV, tied embeddings
- Tokenizer: SP8192 (SentencePiece 8K)
- Depth Recurrence: layers 3-5, loop 2x, activates at 35% of training
- QK-Gain: 5.25 (learnable per-head scaling)
- MLP: 4x dim, squared LeakyReLU
- RoPE: 16 dims, base 10000
- Logit Softcap: 30.0
```

## Training Pipeline (Stable)

### Features
- Checkpointing every 5 minutes with resume capability
- NaN/Inf detection with graceful recovery
- Signal handling for clean shutdown
- 10-minute validation cycles
- EMA model averaging

### Hyperparameters
```
Iterations: 2500 (quick) / 50000+ (extended)
Effective Batch: 32K tokens (8K x 4 accum)
Matrix LR: 0.001 (AdamW)
Weight Decay: 0.095
EMA: 0.9965
Warmdown: 72% of training
```

## Current Status

### Phase 1: Stability (COMPLETED)
- Rewrote train_kaggle.py from scratch
- Removed dead code (MuonEqR, dead compression functions)
- Fixed TTT implementation (was calling model twice)
- Added checkpointing with resume
- Added signal handling for graceful shutdown
- Added NaN/Inf detection

### Phase 2: Baseline (IN PROGRESS)
- Run 10-min baseline to establish current BPB
- Expected: ~2.3-2.5 BPB with 2500 steps

### Phase 3: Hybrid Architecture (PENDING)
- Add PPM (Prediction by Partial Match) for byte-level patterns
- Add adaptive cache for document repetition
- Target: ~0.1-0.15 BPB improvement

### Phase 4: Extended Training (PENDING)
- Scale to 50K-100K steps with extended sessions
- Target: ~1.4-1.6 BPB (neural only)
- With hybrid: sub-1 achievable

## Key Files

- `train_kaggle.py`: Clean, stable training pipeline
- `run_train.sh`: Runner with all hyperparameters
- `checkpoints/`: Saved states for resume
- `logs/`: Training logs

## Next Steps

1. Run 10-min baseline on Kaggle
2. Verify BPB is reasonable (~2.3-2.5)
3. Begin incremental optimization experiments
4. Add PPM integration for hybrid approach