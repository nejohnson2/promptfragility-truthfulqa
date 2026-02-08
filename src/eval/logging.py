"""Run logging and metadata utilities."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


def get_git_hash() -> str:
    """Return current git commit hash, or 'unknown' if not in a repo."""
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
            )
            .decode()
            .strip()
        )
    except Exception:
        return "unknown"


def get_gpu_info() -> str:
    """Return accelerator name: CUDA GPU, MPS, or CPU."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    if torch.backends.mps.is_available():
        return "Apple MPS"
    return "cpu"


def get_package_versions() -> dict[str, str]:
    """Return versions of key packages."""
    import transformers
    import datasets
    import numpy
    import pandas

    return {
        "python": sys.version,
        "torch": torch.__version__,
        "transformers": transformers.__version__,
        "datasets": datasets.__version__,
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
    }


def build_metadata(
    run_id: str,
    model_id: str,
    split: str,
    decoding_params: dict[str, Any],
    conditions_file: str,
    dataset_file: str,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build metadata dict for a run."""
    meta = {
        "run_id": run_id,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "model_id": model_id,
        "split": split,
        "decoding_params": decoding_params,
        "conditions_file": conditions_file,
        "dataset_file": dataset_file,
        "git_hash": get_git_hash(),
        "gpu": get_gpu_info(),
        "platform": platform.platform(),
        "packages": get_package_versions(),
    }
    if extra:
        meta.update(extra)
    return meta


def save_metadata(metadata: dict[str, Any], output_dir: Path) -> None:
    """Write metadata.json to the output directory."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "metadata.json"
    with open(path, "w") as f:
        json.dump(metadata, f, indent=2, default=str)


def append_prediction(record: dict[str, Any], output_dir: Path) -> None:
    """Append a single prediction record to predictions.jsonl."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "predictions.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
