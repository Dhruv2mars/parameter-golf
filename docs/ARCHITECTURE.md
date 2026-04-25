# Parameter Golf - Architecture Blueprint

## Ultimate Goal: Sub-1 BPB (Target: 0.5-0.6 BPB)

---

## Final Agreed Architecture: Multi-Level Hybrid Compressor

```
┌─────────────────────────────────────────────────────────────┐
│              MULTI-LEVEL HYBRID COMPRESSOR                  │
│                                                             │
│  final_prediction = gate × neural + (1-gate) × statistical  │
└─────────────────────────────────────────────────────────────┘
```

---

## Component Breakdown

### 1. NEURAL COMPONENT

| Aspect | Value |
|--------|-------|
| Architecture | 11L × 512d, 8H / 4KV |
| Tokenizer | SP8192 (SentencePiece 8K) |
| Depth Recurrence | Layers 3-5, loop 2×, activates at 35% |
| QK-Gain | 5.25 (learnable per-head scaling) |
| Parallel Residuals | From layer 7+ |
| MLP | 4× dim, squared LeakyReLU |
| RoPE | 16 dims, base 10000 |
| Logit Softcap | 30.0 |
| Tied Embeddings | Yes |
| Quantization | GPTQ int6/int8 + Brotli-11 |
| Expected BPB | ~1.4-1.5 (alone) |

### 2. STATISTICAL COMPONENTS

#### 2a. PPM-D (Prediction by Partial Match)
- Order: 5 (byte-level context)
- Runs on CPU (doesn't compete with GPU)
- Handles: URLs, code, exact repeats, boilerplate
- Budget: ~2-4 MB
- Expected gain: **+0.10-0.15 BPB**

#### 2b. N-gram Table
- Context: last 1-5 tokens
- Top-k next tokens per context
- Compressed hash table
- Budget: ~2-4 MB
- Handles: common phrases, local patterns
- Expected gain: **+0.05-0.10 BPB**

#### 2c. Adaptive Cache
- Document-local repetition
- Score-first update (legal)
- Hash(last n tokens) → recent counts
- Budget: runtime only (not stored)
- Expected gain: **+0.02-0.05 BPB**

### 3. MIXTURE GATE

| Aspect | Value |
|--------|-------|
| Type | Learned per-context |
| Update | Score-first (legal) |
| Features | n-gram confidence, entropy, context type |
| Output | λ = sigmoid(W × features) |
| Formula | final = λ × neural + (1-λ) × statistical |

### 4. TEST-TIME TRAINING (TTT)

| Aspect | Value |
|--------|-------|
| Type | Legal score-first SGD |
| Epochs | 3-4 |
| LR | 0.005, momentum 0.9 |
| Tokens/chunk | 32K |
| Expected gain | **+0.05-0.10 BPB** |

---

## BPB Breakdown (How We Get Sub-1)

| Component | BPB | Cumulative |
|-----------|-----|------------|
| Neural alone | ~1.4-1.5 | 1.4-1.5 |
| + PPM | ~1.25-1.35 | 1.25-1.35 |
| + N-gram | ~1.15-1.25 | 1.15-1.25 |
| + Cache | ~1.10-1.20 | 1.10-1.20 |
| + TTT | ~1.00-1.10 | 1.00-1.10 |
| **Sub-1 Target** | **0.95-1.00** | |

For **0.5-0.6 BPB**:
- Need Scylla tokenizer (998 tokens, proven 0.94 neural)
- More training time (12+ hours)
- Higher-order PPM (order 6-8)
- Possibly multi-level retrieval

---

## Training Strategy

### Phase 1: Stability (DONE)
- [x] Clean, stable pipeline
- [x] Checkpointing every 5 min
- [x] NaN/Inf detection
- [x] 10-min validation cycles

### Phase 2: Incremental Validation
- [ ] Build neural baseline first
- [ ] Run 10-min test → verify ~2.5-2.7 BPB
- [ ] Add PPM → run 10-min test → verify improvement
- [ ] Add cache → run 10-min test → verify improvement
- [ ] Add TTT → run 10-min test → verify improvement
- [ ] Add mixture gate → run 10-min test → verify improvement

### Phase 3: Scale
- [ ] Once hybrid works in 10 min → extend to 2+ hours
- [ ] Target: sub-1.2 BPB minimum
- [ ] With extended time + Scylla tokenizer → sub-1 achievable

---

## Implementation Priority

1. **PPM Integration** (highest ROI, lowest complexity)
2. **Learned Mixture Gate**
3. **Adaptive Cache**
4. **Scylla Tokenizer** (for sub-1 push)
5. **Extended Training** (2+ hours)

---

## Why Hybrid > Pure Neural

| Problem | Pure Neural | Hybrid |
|---------|-------------|--------|
| Local patterns | Wastes params | PPM handles |
| URLs/code | Wastes params | PPM handles |
| Exact repeats | Wastes params | Cache handles |
| Semantic patterns | Learns | Learns |
| **Total params** | **36M** | **36M + 4MB tables** |

**Hybrid = same neural params + statistical compression**

---

## Artifact Budget

| Component | Size |
|-----------|------|
| Neural weights | ~10-12 MB |
| PPM tables | ~2-4 MB |
| N-gram table | ~2-4 MB |
| Code | ~0.5 MB |
| **Total** | **<16 MB** ✓ |

---

## Legal Considerations

- **Score-first update**: PPM/cache update AFTER scoring token
- **No future info**: Only use past tokens for prediction
- **Artifact only**: No external data at eval time

---

*Last updated: 2026-04-25*
*Based on grill-me session with rigorous design review*