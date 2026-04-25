#!/usr/bin/env python3
"""
Architecture Verification Experiment Runner
==========================================
Runs multiple 10-minute experiments in parallel to verify architecture choices.

Usage:
    python3 experiments/arch_verifier.py --run-all
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from dataclasses import dataclass

REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "experiments" / "arch_work"
OUT = REPO / "experiments" / "arch_results"

@dataclass
class Experiment:
    name: str
    env: dict
    description: str

# =============================================================================
# EXPERIMENTS TO RUN
# =============================================================================

EXPERIMENTS = [
    # Reference (our current design)
    Experiment(
        name="a1_baseline",
        env={
            "RUN_ID": "a1_baseline",
            "NUM_LAYERS": "11",
            "MODEL_DIM": "512",
            "VOCAB_SIZE": "8192",
            "QK_GAIN_INIT": "5.25",
            "NUM_LOOPS": "2",
            "LOOP_START": "3",
            "LOOP_END": "5",
            "ENABLE_LOOPING_AT": "0.35",
        },
        description="Reference: 11L×512d, SP8192, QK=5.25, depth recurrence"
    ),
    # Model size variants
    Experiment(
        name="a2_smaller_dim",
        env={
            "RUN_ID": "a2_smaller_dim",
            "NUM_LAYERS": "11",
            "MODEL_DIM": "384",
            "VOCAB_SIZE": "8192",
            "QK_GAIN_INIT": "5.25",
            "NUM_LOOPS": "2",
            "LOOP_START": "3",
            "LOOP_END": "5",
            "ENABLE_LOOPING_AT": "0.35",
        },
        description="Smaller dim: 11L×384d (cheaper, can train more steps)"
    ),
    Experiment(
        name="a3_larger_dim",
        env={
            "RUN_ID": "a3_larger_dim",
            "NUM_LAYERS": "11",
            "MODEL_DIM": "640",
            "VOCAB_SIZE": "8192",
            "QK_GAIN_INIT": "5.25",
            "NUM_LOOPS": "2",
            "LOOP_START": "3",
            "LOOP_END": "5",
            "ENABLE_LOOPING_AT": "0.35",
        },
        description="Larger dim: 11L×640d (more capacity)"
    ),
    # Layer count variants
    Experiment(
        name="a4_fewer_layers",
        env={
            "RUN_ID": "a4_fewer_layers",
            "NUM_LAYERS": "8",
            "MODEL_DIM": "512",
            "VOCAB_SIZE": "8192",
            "QK_GAIN_INIT": "5.25",
            "NUM_LOOPS": "2",
            "LOOP_START": "2",
            "LOOP_END": "4",
            "ENABLE_LOOPING_AT": "0.35",
        },
        description="Fewer layers: 8L×512d"
    ),
    Experiment(
        name="a5_more_layers",
        env={
            "RUN_ID": "a5_more_layers",
            "NUM_LAYERS": "15",
            "MODEL_DIM": "512",
            "VOCAB_SIZE": "8192",
            "QK_GAIN_INIT": "5.25",
            "NUM_LOOPS": "2",
            "LOOP_START": "4",
            "LOOP_END": "6",
            "ENABLE_LOOPING_AT": "0.35",
        },
        description="More layers: 15L×512d"
    ),
    # Tokenizer variants
    Experiment(
        name="a6_vocab4096",
        env={
            "RUN_ID": "a6_vocab4096",
            "NUM_LAYERS": "11",
            "MODEL_DIM": "512",
            "VOCAB_SIZE": "4096",
            "QK_GAIN_INIT": "5.25",
            "NUM_LOOPS": "2",
            "LOOP_START": "3",
            "LOOP_END": "5",
            "ENABLE_LOOPING_AT": "0.35",
        },
        description="Smaller vocab: SP4096"
    ),
    Experiment(
        name="a7_vocab1024",
        env={
            "RUN_ID": "a7_vocab1024",
            "NUM_LAYERS": "11",
            "MODEL_DIM": "512",
            "VOCAB_SIZE": "1024",
            "QK_GAIN_INIT": "5.25",
            "NUM_LOOPS": "2",
            "LOOP_START": "3",
            "LOOP_END": "5",
            "ENABLE_LOOPING_AT": "0.35",
        },
        description="Small vocab: SP1024 (like Karpathy's nanochat)"
    ),
    # Architecture tests
    Experiment(
        name="a8_no_recurrence",
        env={
            "RUN_ID": "a8_no_recurrence",
            "NUM_LAYERS": "11",
            "MODEL_DIM": "512",
            "VOCAB_SIZE": "8192",
            "QK_GAIN_INIT": "5.25",
            "NUM_LOOPS": "1",
            "LOOP_START": "3",
            "LOOP_END": "5",
            "ENABLE_LOOPING_AT": "1.0",  # Never activate
        },
        description="No depth recurrence (tests if recurrence helps)"
    ),
    # QK-Gain variants
    Experiment(
        name="a9_qk_lower",
        env={
            "RUN_ID": "a9_qk_lower",
            "NUM_LAYERS": "11",
            "MODEL_DIM": "512",
            "VOCAB_SIZE": "8192",
            "QK_GAIN_INIT": "4.0",
            "NUM_LOOPS": "2",
            "LOOP_START": "3",
            "LOOP_END": "5",
            "ENABLE_LOOPING_AT": "0.35",
        },
        description="Lower QK-Gain: 4.0"
    ),
    Experiment(
        name="a10_qk_higher",
        env={
            "RUN_ID": "a10_qk_higher",
            "NUM_LAYERS": "11",
            "MODEL_DIM": "512",
            "VOCAB_SIZE": "8192",
            "QK_GAIN_INIT": "6.0",
            "NUM_LOOPS": "2",
            "LOOP_START": "3",
            "LOOP_END": "5",
            "ENABLE_LOOPING_AT": "0.35",
        },
        description="Higher QK-Gain: 6.0"
    ),
]


def run(cmd, cwd=REPO, check=True):
    return subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check)


def prepare_kernel(exp):
    """Create a kernel directory for this experiment."""
    target = WORK / exp.name
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)
    
    # Copy requirements
    shutil.copy2(REPO / "requirements.txt", target / "requirements.txt")
    
    # Create wrapper that sets env vars
    env_lines = "\n".join(f"os.environ[{k!r}] = {v!r}" for k, v in sorted(exp.env.items()))
    train_source = (REPO / "train_kaggle.py").read_text()
    
    wrapper = target / "run_kernel.py"
    wrapper.write_text(
        "import os\n"
        f"{env_lines}\n"
        f"{train_source}\n"
    )
    
    # Kernel metadata
    slug = f"pg-arch-{exp.name}-{int(time.time()) % 100000}"
    (target / "kernel-metadata.json").write_text(json.dumps({
        "id": f"dhruv2mars/{slug}",
        "title": f"PG Arch: {exp.name}",
        "code_file": "run_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": True,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
        "dataset_sources": [],
        "competition_sources": [],
        "kernel_sources": [],
        "model_sources": [],
    }, indent=2) + "\n")
    
    return target, f"dhruv2mars/{slug}"


def push_kernel(path):
    """Push kernel to Kaggle."""
    cp = run(["kaggle", "kernels", "push", "-p", str(path), "--accelerator", "NvidiaTeslaT4"], check=False)
    if cp.returncode != 0:
        raise RuntimeError(f"Push failed: {cp.stdout}")
    return cp.stdout


def wait_for_completion(ref, timeout=900, poll=30):
    """Wait for kernel to complete."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        cp = run(["kaggle", "kernels", "status", ref], check=False)
        m = re.search(r'status "([^"]+)"', cp.stdout)
        status = m.group(1) if m else "UNKNOWN"
        
        if "COMPLETE" in status or "ERROR" in status:
            return status
        
        time.sleep(poll)
    
    return "TIMEOUT"


def pull_output(ref, dest):
    """Pull kernel output."""
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    run(["kaggle", "kernels", "output", ref, "-p", str(dest), "--force"], check=False)


def parse_results(dest):
    """Parse results from kernel output."""
    # Find log files
    log_files = list(dest.glob("logs/*.txt")) + list(dest.glob("logs/*.log"))
    
    text = ""
    for f in log_files:
        text += f.read_text(errors="replace")
    
    # Extract metrics
    vals = [float(x) for x in re.findall(r"val_bpb[=:]\s*([0-9.]+)", text, re.I)]
    bpb_vals = [float(x) for x in re.findall(r"final_val_bpb[=:\s]*([0-9.]+)", text)]
    steps = [int(x) for x in re.findall(r"total_steps[=:\s]*(\d+)", text)]
    times = [float(x) for x in re.findall(r"total_time_s[=:\s]*([0-9.]+)", text)]
    
    # Get final BPB
    final_bpb = min(bpb_vals) if bpb_vals else (min(vals) if vals else None)
    total_steps = max(steps) if steps else 0
    total_time = max(times) if times else 0
    
    return {
        "val_bpb": final_bpb,
        "steps": total_steps,
        "time_s": total_time,
        "status": "COMPLETE" if final_bpb else "FAILED"
    }


def print_results_table(results):
    """Print formatted results table."""
    print("\n" + "=" * 80)
    print("ARCHITECTURE VERIFICATION RESULTS")
    print("=" * 80)
    print(f"{'Rank':<5} {'Name':<20} {'val_bpb':<10} {'Steps':<8} {'Time':<8} {'Status':<10}")
    print("-" * 80)
    
    # Sort by val_bpb
    valid = [r for r in results if r["val_bpb"] is not None]
    valid.sort(key=lambda x: x["val_bpb"])
    
    for i, r in enumerate(valid, 1):
        print(f"{i:<5} {r['name']:<20} {r['val_bpb']:.4f}    {r['steps']:<8} {r['time_s']:<8.0f} {r['status']:<10}")
    
    print("-" * 80)
    
    if valid:
        best = valid[0]
        worst = valid[-1]
        print(f"Best:  {best['name']} ({best['val_bpb']:.4f} BPB)")
        print(f"Worst: {worst['name']} ({worst['val_bpb']:.4f} BPB)")
        print(f"Delta: {worst['val_bpb'] - best['val_bpb']:.4f} BPB")
    
    print("=" * 80)


def analyze_results(results):
    """Analyze and provide insights."""
    valid = [r for r in results if r["val_bpb"] is not None]
    if not valid:
        print("No valid results to analyze")
        return
    
    valid.sort(key=lambda x: x["val_bpb"])
    best = valid[0]
    
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)
    
    # Baseline (a1)
    baseline = next((r for r in valid if "baseline" in r["name"]), None)
    
    if baseline:
        print(f"\nBaseline (a1): {baseline['val_bpb']:.4f} BPB")
        
        for r in valid:
            if r["name"] != "a1_baseline":
                delta = r["val_bpb"] - baseline["val_bpb"]
                direction = "↓" if delta < 0 else "↑"
                print(f"  {r['name']}: {r['val_bpb']:.4f} BPB ({delta:+.4f} {direction})")
    
    print("\n" + "-" * 80)
    print("VERDICT")
    print("-" * 80)
    
    # Check if our architecture is good
    ref_bpb = baseline["val_bpb"] if baseline else 3.0
    
    if best["val_bpb"] < ref_bpb:
        improvement = ref_bpb - best["val_bpb"]
        winner = next(r for r in valid if r["name"] == best["name"])
        
        print(f"\n✓ Architecture verdict: {'BETTER' if winner['name'] != 'a1_baseline' else 'CONFIRMED'}")
        print(f"  Best variant: {winner['name']} ({winner['val_bpb']:.4f} BPB)")
        print(f"  Improvement over baseline: {improvement:.4f} BPB")
        
        if winner["name"] == "a1_baseline":
            print("\n  → Our current design (11L×512d, SP8192, QK=5.25) is confirmed as best")
        else:
            print(f"\n  → Better architecture found: {winner['name']}")
    
    print("=" * 80)


def run_all_experiments():
    """Run all architecture experiments."""
    print("=" * 80)
    print("ARCHITECTURE VERIFICATION - PARALLEL EXPERIMENTS")
    print("=" * 80)
    print(f"Running {len(EXPERIMENTS)} experiments in parallel...")
    print(f"Each experiment: 10 minutes")
    print(f"Total time (parallel): ~10 minutes")
    print("=" * 80)
    
    WORK.mkdir(parents=True, exist_ok=True)
    OUT.mkdir(parents=True, exist_ok=True)
    
    # Prepare and push all kernels
    refs = []
    for exp in EXPERIMENTS:
        print(f"\nPreparing: {exp.name}")
        path, ref = prepare_kernel(exp)
        print(f"  Pushing to Kaggle: {ref}")
        push_kernel(path)
        refs.append((exp, ref))
        print(f"  ✓ Pushed")
    
    # Wait for all to complete
    print("\n" + "=" * 80)
    print("WAITING FOR COMPLETION...")
    print("=" * 80)
    
    results = []
    for exp, ref in refs:
        print(f"Waiting for {exp.name}...", end=" ", flush=True)
        status = wait_for_completion(ref, timeout=900)
        print(f"{status}")
        
        dest = OUT / exp.name
        pull_output(ref, dest)
        
        result = parse_results(dest)
        result["name"] = exp.name
        result["ref"] = ref
        result["description"] = exp.description
        results.append(result)
    
    # Print and analyze
    print_results_table(results)
    analyze_results(results)
    
    # Save results
    results_file = OUT / "results.json"
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    return results


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)
    
    run_all = sub.add_parser("run-all", help="Run all architecture experiments")
    run_all.set_defaults(func=lambda args: run_all_experiments())
    
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()