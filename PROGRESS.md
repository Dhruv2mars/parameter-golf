# Parameter Golf Progress

## Goal

Train a submission-ready language model under the 16,000,000-byte artifact cap. Do not submit until results are comparable to top 5 leaderboard scores.

Current top-5 target from upstream README: 1.0810-1.0856 val_bpb.

## Current Local Result

`exp6_baseline`: 2.60 val_bpb, 6L x 384 GQA, Muon + weight decay, 2.48MB compressed. Useful only as a local experiment; not a valid record submission.

## Current Kaggle Baseline

`parameter-golf-t4x2-v5` v3: 2.4912 val_bpb at step 500 on 2x Tesla T4, 36.0M params, SP8192, 11L x 512d, AdamW lr=0.0003. Training reached wallclock cap at step 522. Run errored during post-train quantization after saving `best_model.pt`; score is from rank0 validation log, not a submission-ready artifact.

## Active Files

- `train_kaggle.py`: Kaggle training script.
- `run_train.sh`: local/Kaggle runner for `train_kaggle.py`.
- `upload_kaggle.sh`: pushes the script kernel using `kernel-metadata.json`.
- `kernel-metadata.json`: Kaggle script-kernel metadata.
- `train_gpt.py`, `train_gpt_mlx.py`, `records/`, `data/`: upstream reference/evidence.

## Next Work

1. Restore stable high-performance optimizer path; current Muon variant NaNs after first step.
2. Reduce validation overhead; duplicate validation at each grad-accum microstep was fixed after v3.
3. Re-run Kaggle baseline after quantization/save fix.
4. Only create a new `records/` folder after a reproducible, compliant result exists.
