#!/usr/bin/env python3
"""Kaggle-backed Parameter Golf experiment helper."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


REPO = Path(__file__).resolve().parents[1]
WORK = REPO / "experiments" / "kaggle_work"
OUT = REPO / "experiments" / "kaggle_results"
STATE = REPO / "experiments" / "kaggle_state.json"
DEFAULT_REF = "dhruv2mars/parameter-golf-t4-stable"


@dataclass
class Metrics:
    ref: str
    status: str
    best_val_bpb: float | None
    ttt_val_bpb: float | None
    steps: int
    compressed_bytes: int | None
    runtime_seconds: float | None
    non_finite: bool
    error: str | None

    @property
    def metric(self) -> float:
        return self.ttt_val_bpb or self.best_val_bpb or 999.0


def run(cmd: list[str], cwd: Path = REPO, check: bool = True) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=check)


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")
    return s[:45] or "trial"


def load_state() -> dict:
    if STATE.exists():
        return json.loads(STATE.read_text())
    return {
        "best": {
            "ref": DEFAULT_REF,
            "best_val_bpb": 2.4912,
            "metric": 2.4912,
            "description": "Kaggle v3 AdamW baseline, 2xT4, step 500",
        },
        "runs": [],
    }


def save_state(state: dict) -> None:
    STATE.parent.mkdir(parents=True, exist_ok=True)
    tmp = STATE.with_suffix(".tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n")
    tmp.replace(STATE)


def prepare_kernel(name: str, env: dict[str, str]) -> tuple[Path, str]:
    slug = slugify(name)
    ref = f"dhruv2mars/pg-auto-{slug}-{int(time.time()) % 100000}"
    target = WORK / slug
    if target.exists():
        shutil.rmtree(target)
    target.mkdir(parents=True)

    for rel in ["requirements.txt"]:
        shutil.copy2(REPO / rel, target / rel)

    wrapper = target / "run_kernel.py"
    env_lines = "\n".join(f"os.environ[{k!r}] = {v!r}" for k, v in sorted(env.items()))
    train_source = (REPO / "train_kaggle.py").read_text()
    wrapper.write_text(
        "import os\n"
        f"{env_lines}\n"
        "# Inlined because Kaggle executes only code_file as /kaggle/src/script.py.\n"
        f"{train_source}\n"
    )
    (target / "kernel-metadata.json").write_text(
        json.dumps(
            {
                "id": ref,
                "title": ref.split("/", 1)[1].replace("-", " ").title(),
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
            },
            indent=2,
        )
        + "\n"
    )
    return target, ref


def push_kernel(path: Path) -> str:
    cp = run(["kaggle", "kernels", "push", "-p", str(path), "--accelerator", "NvidiaTeslaT4"], check=False)
    if cp.returncode != 0:
        raise RuntimeError(cp.stdout)
    return cp.stdout


def status(ref: str) -> str:
    cp = run(["kaggle", "kernels", "status", ref], check=False)
    m = re.search(r'status "([^"]+)"', cp.stdout)
    return m.group(1) if m else cp.stdout.strip()


def wait(ref: str, timeout: int, poll: int) -> str:
    deadline = time.time() + timeout
    last = ""
    while time.time() < deadline:
        last = status(ref)
        if last.endswith(".COMPLETE") or last.endswith(".ERROR") or last in {"COMPLETE", "ERROR"}:
            return last
        time.sleep(poll)
    return last or "TIMEOUT"


def pull(ref: str, dest: Path) -> None:
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True)
    run(["kaggle", "kernels", "output", ref, "-p", str(dest), "--force"], check=False)


def parse_logs(ref: str, dest: Path, status_value: str) -> Metrics:
    text = "\n".join(p.read_text(errors="replace") for p in dest.rglob("*.txt"))
    text += "\n".join(p.read_text(errors="replace") for p in dest.rglob("*.log"))

    vals = [float(x) for x in re.findall(r"Best val_bpb:\s*([0-9.]+)", text)]
    ttt = [float(x) for x in re.findall(r"TTT val_bpb:\s*([0-9.]+)", text)]
    steps = [int(x) for x in re.findall(r"step:(\d+)/", text)]
    compressed = [int(x) for x in re.findall(r"Compressed:\s*(\d+)\s*bytes", text)]
    times = [float(x) / 1000.0 for x in re.findall(r"time:([0-9.]+)ms", text)]
    err_match = re.search(r"(Traceback .*|RuntimeError: .*|IndexError: .*|CUDA out of memory.*)", text, re.S)
    return Metrics(
        ref=ref,
        status=status_value,
        best_val_bpb=min(vals) if vals else None,
        ttt_val_bpb=min(ttt) if ttt else None,
        steps=max(steps) if steps else 0,
        compressed_bytes=compressed[-1] if compressed else None,
        runtime_seconds=max(times) if times else None,
        non_finite="Non-finite loss" in text or "train_loss:nan" in text,
        error=(err_match.group(1)[:500] if err_match else None),
    )


def record(metrics: Metrics, desc: str) -> None:
    state = load_state()
    row = {
        "ref": metrics.ref,
        "description": desc,
        "status": metrics.status,
        "best_val_bpb": metrics.best_val_bpb,
        "ttt_val_bpb": metrics.ttt_val_bpb,
        "metric": metrics.metric,
        "steps": metrics.steps,
        "compressed_bytes": metrics.compressed_bytes,
        "runtime_seconds": metrics.runtime_seconds,
        "non_finite": metrics.non_finite,
        "error": metrics.error,
        "ts": int(time.time()),
    }
    state.setdefault("runs", []).append(row)
    best_metric = float(state.get("best", {}).get("metric", 999.0))
    if metrics.metric < best_metric and not metrics.non_finite:
        state["best"] = row
    save_state(state)


def cmd_baseline(_args: argparse.Namespace) -> int:
    state = load_state()
    print(json.dumps({"val_bpb": state["best"]["metric"], "best": state["best"]}, sort_keys=True))
    return 0


def cmd_launch(args: argparse.Namespace) -> int:
    env = dict(item.split("=", 1) for item in args.env)
    env.setdefault("RUN_ID", slugify(args.name))
    path, ref = prepare_kernel(args.name, env)
    push_kernel(path)
    print(json.dumps({"ref": ref, "path": str(path), "env": env}, sort_keys=True))
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    env = dict(item.split("=", 1) for item in args.env)
    env.setdefault("RUN_ID", slugify(args.name))
    path, ref = prepare_kernel(args.name, env)
    push_kernel(path)
    st = wait(ref, args.timeout, args.poll)
    dest = OUT / ref.split("/", 1)[1]
    pull(ref, dest)
    metrics = parse_logs(ref, dest, st)
    record(metrics, args.name)
    print(json.dumps(metrics.__dict__ | {"metric": metrics.metric, "val_bpb": metrics.metric, "output": str(dest)}, sort_keys=True))
    return 0 if metrics.best_val_bpb is not None and not metrics.non_finite else 1


def cmd_collect(args: argparse.Namespace) -> int:
    st = status(args.ref)
    dest = OUT / args.ref.split("/", 1)[1]
    pull(args.ref, dest)
    metrics = parse_logs(args.ref, dest, st)
    record(metrics, args.name or args.ref)
    print(json.dumps(metrics.__dict__ | {"metric": metrics.metric, "val_bpb": metrics.metric, "output": str(dest)}, sort_keys=True))
    return 0 if metrics.best_val_bpb is not None and not metrics.non_finite else 1


def cmd_status(args: argparse.Namespace) -> int:
    print(status(args.ref))
    return 0


def main() -> int:
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("baseline").set_defaults(func=cmd_baseline)

    launch = sub.add_parser("launch")
    launch.add_argument("--name", required=True)
    launch.add_argument("--env", action="append", default=[])
    launch.set_defaults(func=cmd_launch)

    runp = sub.add_parser("run")
    runp.add_argument("--name", required=True)
    runp.add_argument("--env", action="append", default=[])
    runp.add_argument("--timeout", type=int, default=5400)
    runp.add_argument("--poll", type=int, default=120)
    runp.set_defaults(func=cmd_run)

    collect = sub.add_parser("collect")
    collect.add_argument("--ref", required=True)
    collect.add_argument("--name")
    collect.set_defaults(func=cmd_collect)

    stat = sub.add_parser("status")
    stat.add_argument("--ref", required=True)
    stat.set_defaults(func=cmd_status)

    args = p.parse_args()
    try:
        return args.func(args)
    except Exception as exc:
        print(json.dumps({"error": str(exc), "metric": 999.0}, sort_keys=True))
        return 1


if __name__ == "__main__":
    sys.exit(main())
