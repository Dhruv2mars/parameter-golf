# Lightning AI Setup Guide

## Overview

This guide sets up your Parameter Golf training on Lightning AI with:
- 2x T4 GPU (16GB each)
- Persistent storage across 4-hour session resets
- Auto-checkpoint and resume

---

## Step 1: Create a New Studio

1. Go to [lightning.ai](https://lightning.ai)
2. Click **Create Studio**
3. Select **T4x2** (2x T4 GPUs)
4. Name it `parameter-golf`
5. Wait for it to start

---

## Step 2: One-Time Setup (Run Once)

Open the Studio terminal and run:

```bash
# Download setup script
wget https://raw.githubusercontent.com/Dhruv2mars/parameter-golf/main/lightning_setup.sh

# Run setup (this installs deps, clones repo, downloads data)
bash lightning_setup.sh
```

This takes ~10-15 minutes. You'll see:
- CLI installation
- Repo clone
- Dependencies installed
- FineWeb data downloaded (2 shards for testing)

---

## Step 3: Configure Auto-Start (Important!)

To handle 4-hour session resets, set up auto-start:

```bash
# Copy the autostart script to your home directory
cp /workspace/parameter-golf/lightning_autostart.sh ~/.studiorc
```

This ensures training auto-resumes when you restart your session.

---

## Step 4: Start Training

**Option A: Interactive (keeps terminal active)**

```bash
cd /workspace/parameter-golf
bash run_lightning.sh
```

**Option B: As batch job (runs in background)**

```bash
cd /workspace/parameter-golf
lightning run job \
  --name "parameter-golf-training" \
  --command "bash run_lightning.sh" \
  --machine T4x2 \
  --image python3.10
```

---

## Step 5: Monitor Progress

```bash
# Watch logs
tail -f logs/*.log

# List checkpoints
ls -la checkpoints/*/ckpt_step_*.pt

# Check GPU usage
nvidia-smi
```

---

## Session Reset Handling

When your 4-hour session expires:

1. **Session auto-saves checkpoint** (every 3 hours)
2. **You restart Studio** (click Start on lightning.ai)
3. **Auto-start kicks in** (`~/.studiorc` runs `run_lightning.sh`)
4. **Training resumes from checkpoint** (automatic)

No manual intervention needed!

---

## Customizing Training

Edit environment variables in `run_lightning.sh`:

```bash
# Quick 30-min test
export ITERATIONS=2000
export MAX_WALLCLOCK_SECONDS=1800

# Extended 4-hour run
export ITERATIONS=25000
export MAX_WALLCLOCK_SECONDS=14000

# Different architecture
export NUM_LAYERS=11
export MODEL_DIM=512
export ROPE_DIMS=32
```

---

## Quick Reference

| Command | Description |
|---------|-------------|
| `bash run_lightning.sh` | Start training |
| `tail -f logs/*.log` | Watch logs |
| `ls checkpoints/` | List checkpoints |
| `nvidia-smi` | GPU status |

---

## Troubleshooting

**Session keeps dying without saving:**
- Check disk space: `df -h`
- Ensure persistent storage is enabled

**Checkpoint not found on resume:**
- Check: `ls -la checkpoints/`
- Verify run_id matches

**Out of memory:**
- Reduce `TRAIN_BATCH_TOKENS` to 4096 or 2048
- Reduce `MODEL_DIM` to 480

---

## GPU Hours Tracking

You have **80 GPU hours/month** on free tier.

| Run Type | Est. Time | GPU Hours Used |
|----------|-----------|----------------|
| 30-min test | 30 min | ~1 hour |
| 4-hour run | 4 hours | ~8 hours |
| 12-hour (3 sessions) | 12 hours | ~24 hours |

---

## Quick Start Checklist

- [ ] Create Studio with T4x2
- [ ] Run `lightning_setup.sh`
- [ ] Configure `~/.studiorc`
- [ ] Run `bash run_lightning.sh`
- [ ] Verify training starts (check logs)
