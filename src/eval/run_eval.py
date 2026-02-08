#!/usr/bin/env python3
"""Evaluation runner — runs the full model × condition matrix.

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
import sys
import uuid
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from tqdm import tqdm

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from src.config import DecodingParams
from src.eval.logging import append_prediction, build_metadata, save_metadata
from src.eval.parse_answer import parse_mc_answer
from src.eval.scoring import score_prediction
from src.models.hf_model import HFModelRunner
from src.prompts.render import load_conditions, render_prompt


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

    # Load data and conditions
    records = load_dataset(dataset_path, split)
    conditions = load_conditions(conditions_path)

    print(f"Run ID:     {run_id}")
    print(f"Model:      {model_id}")
    print(f"Split:      {split} ({len(records)} questions)")
    print(f"Conditions: {len(conditions)}")
    print(f"Output:     {run_dir}")

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
    print(f"\nLoading model: {model_id} …")
    runner = HFModelRunner(model_id, decoding=decoding, quantize_4bit=quantize_4bit)

    # Run matrix
    total = len(records) * len(conditions)
    print(f"\nRunning {total} inferences …\n")

    from src.config import TEMPLATE_PATH

    for condition in conditions:
        cond_id = condition["id"]
        label_style = condition["template_vars"]["label_style"]
        print(f"  Condition: {cond_id}")

        for rec in tqdm(records, desc=f"    {cond_id}", leave=False):
            prompt_text, label_map = render_prompt(rec, condition, TEMPLATE_PATH)

            result = runner.generate(prompt_text)

            parsed_label, is_invalid = parse_mc_answer(result.output_text, label_map)
            is_correct, _ = score_prediction(parsed_label, rec["correct_label"])

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

    print(f"\nDone. Predictions saved to {run_dir / 'predictions.jsonl'}")


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
