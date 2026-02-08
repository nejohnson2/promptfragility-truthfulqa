#!/usr/bin/env python3
"""Paired bootstrap confidence intervals for condition deltas.

Usage:
    python -m src.analysis.stats \
        --predictions results/runs/RUN_ID/predictions.jsonl \
        --output_dir results/runs/RUN_ID
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import BOOTSTRAP_N, BOOTSTRAP_SEED


def load_predictions(path: Path) -> pd.DataFrame:
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def paired_bootstrap_delta(
    baseline_correct: np.ndarray,
    condition_correct: np.ndarray,
    n_bootstrap: int = BOOTSTRAP_N,
    seed: int = BOOTSTRAP_SEED,
    ci: float = 0.95,
) -> dict[str, float]:
    """Compute paired bootstrap CI for accuracy delta (condition - baseline).

    Args:
        baseline_correct: Boolean array of baseline correctness per question.
        condition_correct: Boolean array of condition correctness per question.
        n_bootstrap: Number of bootstrap resamples.
        seed: Random seed.
        ci: Confidence interval width.

    Returns:
        Dict with keys: observed_delta, median_delta, ci_low, ci_high.
    """
    assert len(baseline_correct) == len(condition_correct)
    n = len(baseline_correct)

    deltas = condition_correct.astype(float) - baseline_correct.astype(float)
    observed_delta = deltas.mean()

    rng = np.random.RandomState(seed)
    bootstrap_deltas = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.randint(0, n, size=n)
        bootstrap_deltas[i] = deltas[idx].mean()

    alpha = (1 - ci) / 2
    ci_low = np.percentile(bootstrap_deltas, alpha * 100)
    ci_high = np.percentile(bootstrap_deltas, (1 - alpha) * 100)
    median_delta = np.median(bootstrap_deltas)

    return {
        "observed_delta": round(observed_delta, 6),
        "median_delta": round(median_delta, 6),
        "ci_low": round(ci_low, 6),
        "ci_high": round(ci_high, 6),
    }


def compute_all_deltas(df: pd.DataFrame) -> pd.DataFrame:
    """Compute paired bootstrap deltas for each (model, condition) vs baseline."""
    results = []
    models = df["model_id"].unique()
    conditions = [c for c in df["condition_id"].unique() if c != "baseline"]

    for model_id in models:
        model_df = df[df["model_id"] == model_id]
        baseline_df = model_df[model_df["condition_id"] == "baseline"].sort_values("question_id")
        baseline_correct = baseline_df["is_correct"].values.astype(int)
        baseline_qids = baseline_df["question_id"].values

        for cond_id in conditions:
            cond_df = model_df[model_df["condition_id"] == cond_id].sort_values("question_id")
            cond_correct = cond_df["is_correct"].values.astype(int)
            cond_qids = cond_df["question_id"].values

            # Ensure same questions in same order
            if not np.array_equal(baseline_qids, cond_qids):
                # Align on shared questions
                shared = set(baseline_qids) & set(cond_qids)
                mask_b = np.isin(baseline_qids, list(shared))
                mask_c = np.isin(cond_qids, list(shared))
                baseline_correct_aligned = baseline_correct[mask_b]
                cond_correct_aligned = cond_correct[mask_c]
            else:
                baseline_correct_aligned = baseline_correct
                cond_correct_aligned = cond_correct

            if len(baseline_correct_aligned) == 0:
                continue

            stats = paired_bootstrap_delta(baseline_correct_aligned, cond_correct_aligned)
            stats["model_id"] = model_id
            stats["condition_id"] = cond_id
            stats["n_questions"] = len(baseline_correct_aligned)
            results.append(stats)

    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute paired bootstrap CIs.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading predictions from {args.predictions} …")
    df = load_predictions(args.predictions)

    print("Computing paired bootstrap deltas (this may take a minute) …")
    deltas_df = compute_all_deltas(df)

    out_path = args.output_dir / "bootstrap_deltas.csv"
    deltas_df.to_csv(out_path, index=False)
    print(f"\nSaved bootstrap deltas to {out_path}")
    print(f"  {len(deltas_df)} model × condition comparisons")

    # Print summary
    print("\n=== Bootstrap Delta Summary ===")
    for model_id in deltas_df["model_id"].unique():
        model_deltas = deltas_df[deltas_df["model_id"] == model_id]
        print(f"\n  {model_id}:")
        for _, row in model_deltas.iterrows():
            sig = "*" if row["ci_low"] > 0 or row["ci_high"] < 0 else " "
            print(
                f"    {row['condition_id']:25s}  "
                f"Δ={row['observed_delta']:+.4f}  "
                f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}] {sig}"
            )


if __name__ == "__main__":
    main()
