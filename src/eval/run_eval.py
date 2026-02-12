#!/usr/bin/env python3
"""Evaluation runner — runs the full model x condition matrix.

Usage:
    python -m src.eval.run_eval \
        --model_id mistralai/Mistral-7B-Instruct-v0.2 \
        --dataset data/truthfulqa_cache.jsonl \
        --conditions prompts/conditions.yaml \
        --split dev \
        --output_dir results/runs/dev_run_001 \
        [--quantize_4bit]
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
import traceback
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
from tqdm import tqdm

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import DecodingParams
from src.eval.logging import append_prediction, build_metadata, save_metadata
from src.eval.parse_answer import parse_mc_answer
from src.eval.scoring import score_prediction
from src.models.hf_model import HFModelRunner
from src.prompts.render import load_conditions, render_prompt

logger = logging.getLogger("promptfragility")


def setup_logging(run_dir: Path) -> None:
    """Configure logging to both console and file."""
    log_file = run_dir / "run.log"
    fmt = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(fmt)
    file_handler.setLevel(logging.DEBUG)

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(fmt)
    console_handler.setLevel(logging.INFO)

    root = logging.getLogger("promptfragility")
    root.setLevel(logging.DEBUG)
    root.addHandler(file_handler)
    root.addHandler(console_handler)


def log_gpu_memory() -> None:
    """Log current GPU memory usage if CUDA is available."""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            logger.debug(
                "GPU %d: %.1f GB allocated, %.1f GB reserved, %.1f GB total",
                i, allocated, reserved, total,
            )


def load_dataset(path: Path, split: str) -> list[dict[str, Any]]:
    """Load JSONL dataset and filter by split."""
    records = []
    with open(path) as f:
        for line in f:
            rec = json.loads(line)
            if rec["split"] == split:
                records.append(rec)
    if not records:
        raise ValueError(f"No records found for split '{split}' in {path}")
    return records


def run_evaluation(
    model_id: str,
    dataset_path: Path,
    conditions_path: Path,
    split: str,
    output_dir: Path,
    quantize_4bit: bool = False,
    decoding: DecodingParams | None = None,
) -> None:
    """Run full evaluation for one model across all conditions."""
    decoding = decoding or DecodingParams()
    run_id = f"{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:8]}"

    run_dir = output_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    setup_logging(run_dir)

    # Load data and conditions
    records = load_dataset(dataset_path, split)
    conditions = load_conditions(conditions_path)

    total_inferences = len(records) * len(conditions)

    logger.info("Run ID:      %s", run_id)
    logger.info("Model:       %s", model_id)
    logger.info("Split:       %s (%d questions)", split, len(records))
    logger.info("Conditions:  %d", len(conditions))
    logger.info("Total inferences: %d", total_inferences)
    logger.info("Output:      %s", run_dir)
    logger.info("Quantize:    %s", quantize_4bit)
    logger.info("Decoding:    %s", asdict(decoding))

    # Save metadata
    meta = build_metadata(
        run_id=run_id,
        model_id=model_id,
        split=split,
        decoding_params=asdict(decoding),
        conditions_file=str(conditions_path),
        dataset_file=str(dataset_path),
    )
    save_metadata(meta, run_dir)

    # Load model
    logger.info("Loading model: %s", model_id)
    load_start = time.perf_counter()
    runner = HFModelRunner(model_id, decoding=decoding, quantize_4bit=quantize_4bit)
    load_time = time.perf_counter() - load_start
    logger.info("Model loaded in %.1f seconds on device: %s", load_time, runner.device)
    log_gpu_memory()

    # Run matrix
    logger.info("Starting inference (%d total)", total_inferences)
    run_start = time.perf_counter()
    completed = 0
    errors = 0

    from src.config import TEMPLATE_PATH

    for cond_idx, condition in enumerate(conditions, 1):
        cond_id = condition["id"]
        cond_start = time.perf_counter()
        cond_correct = 0
        cond_invalid = 0

        logger.info(
            "Condition [%d/%d]: %s", cond_idx, len(conditions), cond_id
        )

        for rec in tqdm(records, desc=f"  {cond_id}", leave=False):
            try:
                prompt_text, label_map = render_prompt(rec, condition, TEMPLATE_PATH)
                result = runner.generate(prompt_text)
                parsed_label, is_invalid = parse_mc_answer(result.output_text, label_map)
                is_correct, _ = score_prediction(parsed_label, rec["correct_label"])

                if is_invalid:
                    cond_invalid += 1
                if is_correct:
                    cond_correct += 1

                prediction = {
                    "run_id": run_id,
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "model_id": model_id,
                    "condition_id": cond_id,
                    "question_id": rec["question_id"],
                    "split": split,
                    "prompt_text": prompt_text,
                    "output_text": result.output_text,
                    "parsed_label": parsed_label,
                    "correct_label": rec["correct_label"],
                    "is_correct": is_correct,
                    "is_invalid": is_invalid,
                    "token_count": result.token_count,
                    "latency_ms": round(result.latency_ms, 2),
                    "seed": decoding.seed,
                    "decoding_params": asdict(decoding),
                }
                append_prediction(prediction, run_dir)
                completed += 1

            except Exception:
                errors += 1
                logger.error(
                    "Error on question %s, condition %s:\n%s",
                    rec.get("question_id", "?"),
                    cond_id,
                    traceback.format_exc(),
                )

        cond_elapsed = time.perf_counter() - cond_start
        cond_acc = cond_correct / len(records) * 100 if records else 0
        logger.info(
            "  %s: acc=%.1f%%, invalid=%d, time=%.1fs (%.2fs/q)",
            cond_id,
            cond_acc,
            cond_invalid,
            cond_elapsed,
            cond_elapsed / len(records) if records else 0,
        )
        log_gpu_memory()

    total_elapsed = time.perf_counter() - run_start
    logger.info("Inference complete: %d/%d succeeded, %d errors", completed, total_inferences, errors)
    logger.info("Total inference time: %.1f seconds (%.1f min)", total_elapsed, total_elapsed / 60)
    logger.info("Predictions saved to %s", run_dir / "predictions.jsonl")

    if errors > 0:
        logger.warning("There were %d errors during inference. Check run.log for details.", errors)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run TruthfulQA MC evaluation.")
    parser.add_argument("--model_id", required=True, help="HuggingFace model identifier.")
    parser.add_argument(
        "--dataset", type=Path, default=Path("data/truthfulqa_cache.jsonl")
    )
    parser.add_argument(
        "--conditions", type=Path, default=Path("prompts/conditions.yaml")
    )
    parser.add_argument("--split", choices=["dev", "final"], default="dev")
    parser.add_argument("--output_dir", type=Path, default=Path("results/runs"))
    parser.add_argument("--quantize_4bit", action="store_true")
    parser.add_argument("--max_new_tokens", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    decoding = DecodingParams(
        max_new_tokens=args.max_new_tokens,
        seed=args.seed,
    )

    run_evaluation(
        model_id=args.model_id,
        dataset_path=args.dataset,
        conditions_path=args.conditions,
        split=args.split,
        output_dir=args.output_dir,
        quantize_4bit=args.quantize_4bit,
        decoding=decoding,
    )


if __name__ == "__main__":
    main()
