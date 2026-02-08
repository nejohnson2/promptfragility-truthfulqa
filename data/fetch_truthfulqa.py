#!/usr/bin/env python3
"""Download TruthfulQA MC and save as normalized JSONL cache.

Usage:
    python data/fetch_truthfulqa.py --out data/truthfulqa_cache.jsonl --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Any


def load_truthfulqa() -> list[dict[str, Any]]:
    """Load TruthfulQA multiple_choice split from HuggingFace datasets."""
    from datasets import load_dataset

    ds = load_dataset("truthful_qa", "multiple_choice", split="validation")
    return list(ds)


def normalize_record(idx: int, raw: dict[str, Any]) -> dict[str, Any]:
    """Convert a raw HF record into our canonical schema.

    TruthfulQA MC format:
      - question: str
      - mc1_targets: {"choices": [...], "labels": [0|1, ...]}
      - mc2_targets: {"choices": [...], "labels": [0|1, ...]}
      - category: str (optional)

    We use mc1_targets (single correct answer) for MC1 evaluation.
    """
    question = raw["question"]
    category = raw.get("category", "")

    mc1 = raw["mc1_targets"]
    choices_text = mc1["choices"]
    labels = mc1["labels"]

    letter_labels = [chr(ord("A") + i) for i in range(len(choices_text))]
    choices = [{"label": lbl, "text": txt} for lbl, txt in zip(letter_labels, choices_text)]

    correct_idx = labels.index(1)
    correct_label = letter_labels[correct_idx]

    return {
        "question_id": idx,
        "question": question,
        "choices": choices,
        "correct_label": correct_label,
        "category": category,
    }


def assign_splits(
    records: list[dict[str, Any]], dev_fraction: float, seed: int
) -> list[dict[str, Any]]:
    """Deterministically assign each record to 'dev' or 'final' split."""
    rng = random.Random(seed)
    indices = list(range(len(records)))
    rng.shuffle(indices)

    n_dev = max(1, int(len(records) * dev_fraction))
    dev_set = set(indices[:n_dev])

    for i, rec in enumerate(records):
        rec["split"] = "dev" if i in dev_set else "final"
    return records


def save_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"Saved {len(records)} records to {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch and cache TruthfulQA MC data.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/truthfulqa_cache.jsonl"),
        help="Output JSONL path.",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed for split.")
    parser.add_argument(
        "--dev-fraction", type=float, default=0.10, help="Fraction for dev split."
    )
    args = parser.parse_args()

    print("Loading TruthfulQA from HuggingFace datasets …")
    raw_data = load_truthfulqa()
    print(f"  Loaded {len(raw_data)} questions.")

    records = [normalize_record(i, r) for i, r in enumerate(raw_data)]
    records = assign_splits(records, args.dev_fraction, args.seed)

    n_dev = sum(1 for r in records if r["split"] == "dev")
    n_final = sum(1 for r in records if r["split"] == "final")
    print(f"  Split: {n_dev} dev, {n_final} final")

    save_jsonl(records, args.out)


if __name__ == "__main__":
    main()
