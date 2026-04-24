# Experiment 6: 6 Layers + Muon + Weight Decay (Baseline)

**val_bpb = 2.60** (single run) | **~2.48 MB** | Kaggle 2xT4 GPU

## Results

| Run | val_bpb | Compressed | Params | Train Time |
|-----|---------|-----------|--------|------------|
| exp6 | 2.60 | 2.48MB | 5.3M | 818s |

## Progress Summary

| Exp | Model | Opt | Steps | val_bpb | Size |
|-----|-------|-----|-------|---------|------|
| 1 | 3L×256 | AdamW | 20 | 3.79 | 1.9MB |
| 2 | 5L×384 GQA | AdamW | 100 | 3.48 | 4.8MB |
| 3 | 5L×384 GQA | AdamW | 992 | 3.36 | 4.7MB |
| 4 | 5L×384 GQA | AdamW | 1013 | 3.35 | 4.7MB |
| **5** | **5L×384 GQA** | **Muon+WD** | **290** | **2.71** | **2.52MB** |
| **6** | **6L×384 GQA** | **Muon+WD** | **1170** | **2.60** | **2.48MB** |

## Key Findings

- **Muon+weight decay**: Massive improvement (3.35→2.71)
- **6 layers**: Slightly better than 5 layers (2.71→2.60)
- Model still improving at step 1170 — needs more iterations

## Architecture

- 6L × 384 dim × 8 heads / 4 KV heads (GQA)
- MLP 2x expansion
- Tied embeddings
- 1024 vocab (SentencePiece BPE)
- RoPE + QK-Gain init 1.5

## Training

- Optimizer: Muon (2D params) + AdamW (scalars)
- Weight decay: 0.1
- Batch tokens: 16,384
- Seq len: 512
- Wallclock cap: 700s

## Known Issues

- PyTorch reinstall on every run (~165s wasted)
- Need more training iterations
- Val evaluation limited to 32K tokens (OOM prevention)

## Next Steps

1. Fix PyTorch reinstall issue
2. Increase wallclock to allow more steps
3. Upgrade vocab to 4096/8192
4. Add depth recurrence
5. Switch to brotli compression
