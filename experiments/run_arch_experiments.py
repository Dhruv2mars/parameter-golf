#!/usr/bin/env python3
"""Architecture Verification - Corrected parallel execution."""

import os
import json
import time
import subprocess
import shutil
import re
import threading
from pathlib import Path

REPO = Path("/Users/dhruv2mars/dev/github/parameter-golf")
OUT = REPO / "experiments" / "arch_results"

EXPERIMENTS = [
    ("a1_baseline", {"NUM_LAYERS": "11", "MODEL_DIM": "512", "VOCAB_SIZE": "8192", "QK_GAIN_INIT": "5.25", "NUM_LOOPS": "2", "LOOP_START": "3", "LOOP_END": "5", "ENABLE_LOOPING_AT": "0.35"}, "Reference"),
    ("a2_small", {"NUM_LAYERS": "11", "MODEL_DIM": "384", "VOCAB_SIZE": "8192", "QK_GAIN_INIT": "5.25", "NUM_LOOPS": "2", "LOOP_START": "3", "LOOP_END": "5", "ENABLE_LOOPING_AT": "0.35"}, "Smaller dim"),
    ("a3_large", {"NUM_LAYERS": "11", "MODEL_DIM": "640", "VOCAB_SIZE": "8192", "QK_GAIN_INIT": "5.25", "NUM_LOOPS": "2", "LOOP_START": "3", "LOOP_END": "5", "ENABLE_LOOPING_AT": "0.35"}, "Larger dim"),
    ("a4_8l", {"NUM_LAYERS": "8", "MODEL_DIM": "512", "VOCAB_SIZE": "8192", "QK_GAIN_INIT": "5.25", "NUM_LOOPS": "2", "LOOP_START": "2", "LOOP_END": "4", "ENABLE_LOOPING_AT": "0.35"}, "Fewer layers"),
    ("a5_15l", {"NUM_LAYERS": "15", "MODEL_DIM": "512", "VOCAB_SIZE": "8192", "QK_GAIN_INIT": "5.25", "NUM_LOOPS": "2", "LOOP_START": "4", "LOOP_END": "6", "ENABLE_LOOPING_AT": "0.35"}, "More layers"),
    ("a6_vocab4k", {"NUM_LAYERS": "11", "MODEL_DIM": "512", "VOCAB_SIZE": "4096", "QK_GAIN_INIT": "5.25", "NUM_LOOPS": "2", "LOOP_START": "3", "LOOP_END": "5", "ENABLE_LOOPING_AT": "0.35"}, "Vocab 4K"),
    ("a7_vocab1k", {"NUM_LAYERS": "11", "MODEL_DIM": "512", "VOCAB_SIZE": "1024", "QK_GAIN_INIT": "5.25", "NUM_LOOPS": "2", "LOOP_START": "3", "LOOP_END": "5", "ENABLE_LOOPING_AT": "0.35"}, "Vocab 1K"),
    ("a8_no_rec", {"NUM_LAYERS": "11", "MODEL_DIM": "512", "VOCAB_SIZE": "8192", "QK_GAIN_INIT": "5.25", "NUM_LOOPS": "1", "LOOP_START": "3", "LOOP_END": "5", "ENABLE_LOOPING_AT": "1.0"}, "No recurrence"),
    ("a9_qk4", {"NUM_LAYERS": "11", "MODEL_DIM": "512", "VOCAB_SIZE": "8192", "QK_GAIN_INIT": "4.0", "NUM_LOOPS": "2", "LOOP_START": "3", "LOOP_END": "5", "ENABLE_LOOPING_AT": "0.35"}, "QK=4"),
    ("a10_qk6", {"NUM_LAYERS": "11", "MODEL_DIM": "512", "VOCAB_SIZE": "8192", "QK_GAIN_INIT": "6.0", "NUM_LOOPS": "2", "LOOP_START": "3", "LOOP_END": "5", "ENABLE_LOOPING_AT": "0.35"}, "QK=6"),
]

# Global state
kernel_refs = {}
results_lock = threading.Lock()
all_statuses = {}


def push_kernel(name, env, desc):
    """Push a single kernel."""
    target = f"/tmp/pg-arch-{name}"
    os.makedirs(target, exist_ok=True)
    for f in Path(target).glob("*"):
        f.unlink()
    
    env_lines = "\n".join(f"os.environ[{k!r}] = {v!r}" for k, v in sorted(env.items()))
    train_source = (REPO / "train_kaggle.py").read_text()
    
    with open(f"{target}/run_kernel.py", "w") as f:
        f.write(f"import os\n{env_lines}\n{train_source}\n")
    
    metadata = {
        "title": f"PG {name}",
        "code_file": "run_kernel.py",
        "language": "python",
        "kernel_type": "script",
        "is_private": False,
        "enable_gpu": True,
        "enable_tpu": False,
        "enable_internet": True,
    }
    with open(f"{target}/kernel-metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    
    for attempt in range(3):
        result = subprocess.run(
            ["kaggle", "kernels", "push", "-p", target, "--accelerator", "NvidiaTeslaT4"],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            # Find the actual slug from the output
            match = re.search(r"kaggle\.com/code/([^/\s]+)", result.stdout)
            if match:
                slug = match.group(1)
                kernel_refs[name] = f"dhruv2mars/{slug}"
                return True
        
        time.sleep(5)
    
    return False


def wait_for_kernel(name, kernel_ref):
    """Wait for a single kernel to complete."""
    for _ in range(60):  # 30 min timeout
        time.sleep(30)
        result = subprocess.run(
            ["kaggle", "kernels", "status", kernel_ref],
            capture_output=True, text=True
        )
        
        if "COMPLETE" in result.stdout:
            all_statuses[name] = "COMPLETE"
            return
        elif "ERROR" in result.stdout:
            all_statuses[name] = "ERROR"
            return
    
    all_statuses[name] = "TIMEOUT"


def collect_result(name, kernel_ref):
    """Collect results for a single kernel."""
    dest = OUT / name
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    
    subprocess.run(
        ["kaggle", "kernels", "output", kernel_ref, "-p", str(dest), "--force"],
        capture_output=True
    )
    
    bpb = None
    steps = 0
    
    for f in dest.glob("logs/*.txt"):
        text = f.read_text(errors="replace")
        m = re.search(r"final_val_bpb[=:\s]*([0-9.]+)", text)
        if m:
            bpb = float(m.group(1))
        m = re.search(r"total_steps[=:\s]*(\d+)", text)
        if m:
            steps = int(m.group(1))
    
    return {"name": name, "bpb": bpb, "steps": steps}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("ARCHITECTURE VERIFICATION (PARALLEL)")
    print("="*60)
    
    # Step 1: Push all kernels (sequential to avoid conflicts)
    print(f"\nStep 1: Pushing {len(EXPERIMENTS)} kernels (sequential)...")
    for name, env, desc in EXPERIMENTS:
        print(f"  Pushing {name}...", end=" ", flush=True)
        
        # Create and push
        target = f"/tmp/pg-arch-{name}"
        os.makedirs(target, exist_ok=True)
        for f in Path(target).glob("*"):
            f.unlink()
        
        env_lines = "\n".join(f"os.environ[{k!r}] = {v!r}" for k, v in sorted(env.items()))
        train_source = (REPO / "train_kaggle.py").read_text()
        
        with open(f"{target}/run_kernel.py", "w") as f:
            f.write(f"import os\n{env_lines}\n{train_source}\n")
        
        metadata = {
            "title": f"PG {name}",
            "code_file": "run_kernel.py",
            "language": "python",
            "kernel_type": "script",
            "is_private": False,
            "enable_gpu": True,
            "enable_tpu": False,
            "enable_internet": True,
        }
        with open(f"{target}/kernel-metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        result = subprocess.run(
            ["kaggle", "kernels", "push", "-p", target, "--accelerator", "NvidiaTeslaT4"],
            capture_output=True, text=True
        )
        
        if result.returncode == 0:
            match = re.search(r"kaggle\.com/code/([^/\s]+)", result.stdout)
            if match:
                slug = match.group(1)
                kernel_refs[name] = f"dhruv2mars/{slug}"
                print("✓")
            else:
                print("✗ (no slug found)")
        else:
            print(f"✗ ({result.stdout[:50]}...)")
        
        time.sleep(3)  # Small delay between pushes
    
    print(f"\n✓ Pushed {len(kernel_refs)}/{len(EXPERIMENTS)} kernels")
    print(f"  Kernel refs: {kernel_refs}")
    
    # Step 2: Wait for all in parallel
    print(f"\nStep 2: Waiting for completion (parallel)...")
    wait_threads = []
    
    for name, _, _ in EXPERIMENTS:
        if name in kernel_refs:
            t = threading.Thread(target=wait_for_kernel, args=(name, kernel_refs[name]))
            t.start()
            wait_threads.append(t)
            print(f"  {name}: waiting...", end=" ", flush=True)
    
    # Monitor
    last_count = 0
    while any(t.is_alive() for t in wait_threads):
        time.sleep(60)
        complete = sum(1 for s in all_statuses.values() if s in ("COMPLETE", "ERROR"))
        if complete != last_count:
            print(f"\n  {complete}/{len(wait_threads)} complete: {dict(all_statuses)}")
            last_count = complete
    
    for t in wait_threads:
        t.join()
    
    print(f"\n✓ All complete: {all_statuses}")
    
    # Step 3: Collect results in parallel
    print(f"\nStep 3: Collecting results...")
    result_threads = []
    all_results = {}
    
    def collect(name, ref):
        result = collect_result(name, ref)
        with results_lock:
            all_results[name] = result
    
    for name, _, _ in EXPERIMENTS:
        if name in kernel_refs:
            t = threading.Thread(target=collect, args=(name, kernel_refs[name]))
            t.start()
            result_threads.append(t)
    
    for t in result_threads:
        t.join()
    
    # Build results list
    results = []
    for name, env, desc in EXPERIMENTS:
        r = all_results.get(name, {"name": name, "bpb": None, "steps": 0})
        r["desc"] = desc
        r["status"] = all_statuses.get(name, "UNKNOWN")
        results.append(r)
    
    # Summary
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    valid = [r for r in results if r.get("bpb") is not None]
    valid.sort(key=lambda x: x["bpb"])
    
    print(f"\n{'Rank':<5} {'Name':<12} {'BPB':<10} {'Steps':<8} {'Status':<10} {'Description'}")
    print("-"*70)
    for i, r in enumerate(valid, 1):
        print(f"{i:<5} {r['name']:<12} {r['bpb']:.4f}    {r['steps']:<8} {r.get('status', ''):<10} {r.get('desc', '')}")
    
    if valid:
        print("-"*70)
        best = valid[0]
        print(f"\n★ Best: {best['name']} ({best['bpb']:.4f} BPB)")
        
        baseline = next((r for r in valid if "baseline" in r["name"]), best)
        print(f"\n  vs baseline ({baseline['bpb']:.4f} BPB):")
        for r in valid:
            if r["name"] != baseline["name"]:
                delta = r["bpb"] - baseline["bpb"]
                symbol = "↓" if delta < 0 else "↑"
                print(f"    {r['name']}: {r['bpb']:.4f} ({delta:+.4f} {symbol})")
        
        print("\n" + "="*60)
        print("VERDICT")
        print("="*60)
        
        if best["name"] == "a1_baseline":
            print("✓ Our architecture (11L×512d, SP8192, QK=5.25) is confirmed best")
        else:
            print(f"✓ Better architecture found: {best['name']}")
        
        if best["bpb"] < 2.7:
            print(f"\n✓ {best['name']}: {best['bpb']:.4f} BPB (good foundation)")
        else:
            print(f"\n⚠ {best['name']}: {best['bpb']:.4f} BPB (high)")
    
    with open(OUT / "results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*60)


if __name__ == "__main__":
    main()