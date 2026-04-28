#!/usr/bin/env python3
"""
HuggingFace checkpoint sync for Parameter Golf.
Uploads/downloads run artifacts for cross-platform training.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import torch


HF_REPO_ID = "Dhruv2mars/parameter-golf-runs"


def get_hf_repo():
    """Get HF repository instance."""
    from huggingface_hub import HfApi
    api = HfApi()
    # Ensure repo exists
    try:
        api.repo_info(HF_REPO_ID)
    except Exception:
        api.create_repo(HF_REPO_ID, repo_type="model", exist_ok=True)
    return api


def run_exists(run_id: str, repo_id: str = HF_REPO_ID) -> bool:
    """Check if a run has been uploaded to HF."""
    try:
        api = get_hf_repo()
        api.hf_hub_download(
            repo_id=repo_id,
            filename=f"runs/{run_id}/config.json",
            repo_type="model",
        )
        return True
    except Exception:
        return False


def upload_run(
    run_id: str,
    output_dir: str,
    config: Optional[dict] = None,
    repo_id: str = HF_REPO_ID,
    log_path: Optional[str] = None,
) -> dict[str, str]:
    """
    Upload run artifacts to HuggingFace.
    
    Uploads:
    - runs/{run_id}/best_model.pt
    - runs/{run_id}/config.json
    - runs/{run_id}/train.log (if provided)
    
    Returns dict of uploaded paths.
    """
    from huggingface_hub import HfApi

    api = get_hf_repo()
    output_dir = Path(output_dir)
    uploaded = {}

    # best_model.pt
    model_path = output_dir / "best_model.pt"
    if model_path.exists():
        remote_path = f"runs/{run_id}/best_model.pt"
        api.upload_file(
            path_or_fileobj=str(model_path),
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="model",
        )
        uploaded["model"] = remote_path
        print(f"Uploaded: {remote_path}")

    # config.json
    if config is not None:
        config_path = output_dir / "config.json"
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        remote_path = f"runs/{run_id}/config.json"
        api.upload_file(
            path_or_fileobj=str(config_path),
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="model",
        )
        uploaded["config"] = remote_path
        print(f"Uploaded: {remote_path}")

    # train.log
    if log_path and os.path.exists(log_path):
        remote_path = f"runs/{run_id}/train.log"
        api.upload_file(
            path_or_fileobj=log_path,
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="model",
        )
        uploaded["log"] = remote_path
        print(f"Uploaded: {remote_path}")

    # best_model.int8.br (compressed artifact)
    artifact_path = output_dir / "best_model.int8.br"
    if artifact_path.exists():
        remote_path = f"runs/{run_id}/best_model.int8.br"
        api.upload_file(
            path_or_fileobj=str(artifact_path),
            path_in_repo=remote_path,
            repo_id=repo_id,
            repo_type="model",
        )
        uploaded["artifact"] = remote_path
        print(f"Uploaded: {remote_path}")

    print(f"HF upload complete for run_id={run_id}")
    return uploaded


def download_run(
    run_id: str,
    output_dir: str,
    repo_id: str = HF_REPO_ID,
    files: Optional[list[str]] = None,
) -> dict[str, str]:
    """
    Download run artifacts from HuggingFace.
    
    Args:
        run_id: The run identifier
        output_dir: Local directory to save files
        repo_id: HF repo ID
        files: List of files to download. If None, downloads all.
    
    Returns dict mapping remote_path -> local_path.
    """
    from huggingface_hub import hf_hub_download

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    default_files = ["best_model.pt", "config.json", "train.log", "best_model.int8.br"]
    files = files or default_files

    downloaded = {}

    for fname in files:
        remote_path = f"runs/{run_id}/{fname}"
        local_path = output_dir / fname
        try:
            cached = hf_hub_download(
                repo_id=repo_id,
                filename=remote_path,
                repo_type="model",
            )
            import shutil
            shutil.copy(cached, local_path)
            downloaded[remote_path] = str(local_path)
            print(f"Downloaded: {remote_path} -> {local_path}")
        except Exception as e:
            print(f"Skipped {remote_path}: {e}")

    return downloaded


def download_config(run_id: str, repo_id: str = HF_REPO_ID) -> Optional[dict]:
    """Download and parse config.json for a run."""
    try:
        downloaded = download_run(run_id, "/tmp/pg-hf-config-temp", repo_id, files=["config.json"])
        config_path = downloaded.get(f"runs/{run_id}/config.json")
        if config_path and os.path.exists(config_path):
            with open(config_path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return None


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="HF checkpoint sync")
    parser.add_argument("--action", choices=["upload", "download", "exists"], required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--output_dir", default="/tmp/pg-runs")
    parser.add_argument("--config_json", default="{}")
    args = parser.parse_args()

    if args.action == "upload":
        config = json.loads(args.config_json) if args.config_json else None
        result = upload_run(args.run_id, args.output_dir, config=config)
        print(json.dumps(result, indent=2))

    elif args.action == "download":
        result = download_run(args.run_id, args.output_dir)
        print(json.dumps(result, indent=2))

    elif args.action == "exists":
        exists = run_exists(args.run_id)
        print(f"Run exists: {exists}")