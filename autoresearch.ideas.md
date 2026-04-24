# Ideas Backlog - Parameter Golf

## High Priority (Next Iteration)

### 1. Larger Vocabulary Impact
- **Idea**: Jump from 1024 to 8192 vocab
- **Expected gain**: ~0.3-0.5 BPB improvement
- **Risk**: Low
- **Status**: TODO

### 2. Depth Recurrence
- **Idea**: Loop layers 4-5 twice (like SOTA)
- **Expected gain**: ~0.1 BPB improvement
- **Risk**: Medium
- **Status**: TODO

### 3. GPTQ-SDClip Quantization
- **Idea**: int6 matrices with std-based clipping (k=12.85)
- **Expected gain**: Fits more params in 16MB
- **Risk**: Low
- **Status**: TODO

## Medium Priority

### 4. Parallel Residuals
- **Idea**: From layer 7+, attention and MLP read from same pre-residual input
- **Expected gain**: ~0.05 BPB
- **Risk**: Medium
- **Status**: TODO

### 5. TTT (Test-Time Training)
- **Idea**: SGD adaptation on val tokens at eval time
- **Expected gain**: ~0.03 BPB
- **Risk**: Medium
- **Status**: TODO

### 6. QK-Gain Tuning
- **Idea**: Increase from 1.5 to 4.0-5.25
- **Expected gain**: ~0.02 BPB
- **Risk**: Low
- **Status**: TODO

## Lower Priority

### 7. EMA (Exponential Moving Average)
- **Idea**: Track EMA of weights during training
- **Expected gain**: ~0.01 BPB
- **Risk**: Low

### 8. Warmdown Schedule
- **Idea**: Linear LR decay over final 50% of training
- **Expected gain**: Stabilizes training
- **Risk**: Low

### 9. Layerwise LN Scale
- **Idea**: Scale each layer by 1/sqrt(layer_idx+1)
- **Expected gain**: ~0.01 BPB
- **Risk**: Low

## Dead Ends (Don't Try)

- ❌ Ternary quantization (< int4) — hurts too much
- ❌ Flash Attention 3 — T4 doesn't support Hopper
- ❌ Coprime-stride loader — marginal gains
- ❌ Very long context (2048+) — too slow on T4

## SOTA Reference

Top leaderboard scores:
1. 1.0810 — SP8192 + 3-Layer Recurrence + Parallel Residuals + Legal TTT
2. 1.0822 — Parallel Residuals + Score-First TTT  
3. 1.0828 — QK-Gain 5 + Legal TTT
4. 1.0835 — Hessian SDClip + Progressive Recurrence
5. 1.0856 — SP8192 + GPTQ Embeddings + Depth Recurrence + SDClip

Baseline: 1.2244 BPB (OpenAI naive baseline)