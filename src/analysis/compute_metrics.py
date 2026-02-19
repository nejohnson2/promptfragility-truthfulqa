#!/usr/bin/env python3
"""Aggregate predictions into accuracy metrics and ranking stability measures.

Usage:
    python -m src.analysis.compute_metrics \
        --predictions results/runs/RUN_ID/predictions.jsonl \
        --output_dir results/runs/RUN_ID
"""

from __future__ import annotations

import argparse
import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau


def load_predictions(path: Path) -> pd.DataFrame:
    """Load predictions JSONL into a DataFrame."""
    records = []
    with open(path) as f:
        for line in f:
            records.append(json.loads(line))
    return pd.DataFrame(records)


def compute_accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    """Compute per model × condition accuracy and invalid rate."""
    grouped = df.groupby(["model_id", "condition_id"]).agg(
        accuracy=("is_correct", "mean"),
        invalid_rate=("is_invalid", "mean"),
        n=("is_correct", "count"),
    ).reset_index()
    return grouped


def compute_model_summary(accuracy_table: pd.DataFrame) -> pd.DataFrame:
    """Compute per-model summary statistics across conditions."""
    rows = []
    for model_id, grp in accuracy_table.groupby("model_id"):
        accs = grp["accuracy"].values
        baseline_row = grp[grp["condition_id"] == "baseline"]
        baseline_acc = baseline_row["accuracy"].values[0] if len(baseline_row) > 0 else np.nan

        # IQR and MAD — auxiliary robustness metrics less sensitive to outliers
        q75, q25 = np.percentile(accs, [75, 25])
        median_acc = np.median(accs)
        mad = np.median(np.abs(accs - median_acc))

        rows.append({
            "model_id": model_id,
            "baseline_acc": baseline_acc,
            "mean_acc": accs.mean(),
            "median_acc": median_acc,
            "worst_acc": accs.min(),
            "best_acc": accs.max(),
            "std_acc": accs.std(),
            "range_acc": accs.max() - accs.min(),
            "PSI_drop": baseline_acc - accs.min(),
            "PSI_iqr": q75 - q25,
            "PSI_mad": mad,
            "mean_invalid_rate": grp["invalid_rate"].mean(),
        })
    return pd.DataFrame(rows)


def compute_ranking_metrics(accuracy_table: pd.DataFrame) -> dict:
    """Compute ranking stability metrics across conditions.

    Returns dict with:
        kendall_tau: per-condition Kendall τ vs baseline ranking
        rank_flip_rate: fraction of model pairs whose order changes
        top1_instability: count of conditions where top model differs from baseline
    """
    # Get baseline ranking
    baseline = accuracy_table[accuracy_table["condition_id"] == "baseline"].copy()
    baseline = baseline.sort_values("accuracy", ascending=False)
    baseline_ranking = {row["model_id"]: rank for rank, (_, row) in enumerate(baseline.iterrows())}
    baseline_order = list(baseline_ranking.keys())
    baseline_top1 = baseline_order[0] if baseline_order else None

    conditions = accuracy_table["condition_id"].unique()
    # Exclude baseline when computing flip rate and top-1 instability
    # (flips are measured "compared to baseline", so baseline vs itself is excluded)
    non_baseline_conditions = [c for c in conditions if c != "baseline"]
    models = accuracy_table["model_id"].unique()

    kendall_taus = {}
    top1_changes = 0
    all_pair_flips = []

    for cond_id in conditions:
        cond_df = accuracy_table[accuracy_table["condition_id"] == cond_id].copy()
        cond_df = cond_df.sort_values("accuracy", ascending=False)
        cond_ranking = {row["model_id"]: rank for rank, (_, row) in enumerate(cond_df.iterrows())}

        # Kendall tau (compute for all conditions including baseline for completeness)
        if len(cond_ranking) >= 2:
            baseline_ranks = [baseline_ranking.get(m, 0) for m in models]
            cond_ranks = [cond_ranking.get(m, 0) for m in models]
            tau, _ = kendalltau(baseline_ranks, cond_ranks)
            kendall_taus[cond_id] = tau
        else:
            kendall_taus[cond_id] = np.nan

        # Skip baseline for flip rate and top-1 instability
        if cond_id == "baseline":
            continue

        # Top-1 instability
        cond_top1 = list(cond_ranking.keys())[0] if cond_ranking else None
        if cond_top1 != baseline_top1:
            top1_changes += 1

        # Pairwise rank flips
        for m1, m2 in combinations(models, 2):
            b1 = baseline_ranking.get(m1, 0)
            b2 = baseline_ranking.get(m2, 0)
            c1 = cond_ranking.get(m1, 0)
            c2 = cond_ranking.get(m2, 0)
            baseline_order_pair = b1 < b2
            cond_order_pair = c1 < c2
            if baseline_order_pair != cond_order_pair:
                all_pair_flips.append(1)
            else:
                all_pair_flips.append(0)

    n_pairs = len(list(combinations(models, 2)))
    n_non_baseline = len(non_baseline_conditions)
    total_comparisons = n_pairs * n_non_baseline
    rank_flip_rate = sum(all_pair_flips) / total_comparisons if total_comparisons > 0 else 0.0

    return {
        "kendall_tau": kendall_taus,
        "rank_flip_rate": rank_flip_rate,
        "top1_instability": top1_changes,
        "n_conditions": len(conditions),
        "n_non_baseline_conditions": n_non_baseline,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate predictions into metrics.")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading predictions from {args.predictions} …")
    df = load_predictions(args.predictions)
    print(f"  {len(df)} prediction records")

    # Sanity checks
    conditions_present = df["condition_id"].unique()
    assert "baseline" in conditions_present, "baseline condition missing!"
    for cond_id in conditions_present:
        n = len(df[df["condition_id"] == cond_id])
        print(f"  Condition '{cond_id}': {n} predictions")

    # Accuracy table
    accuracy_table = compute_accuracy_table(df)
    out_path = args.output_dir / "aggregated.csv"
    accuracy_table.to_csv(out_path, index=False)
    print(f"\nSaved aggregated metrics to {out_path}")

    # Model summary
    summary = compute_model_summary(accuracy_table)
    summary_path = args.output_dir / "model_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Saved model summary to {summary_path}")

    # Ranking metrics
    ranking = compute_ranking_metrics(accuracy_table)
    ranking_path = args.output_dir / "ranking_metrics.json"
    with open(ranking_path, "w") as f:
        json.dump(ranking, f, indent=2, default=str)
    print(f"Saved ranking metrics to {ranking_path}")

    # Print summary
    print("\n=== Model Summary ===")
    print(summary.to_string(index=False))
    print(f"\nRank flip rate: {ranking['rank_flip_rate']:.3f}")
    print(f"Top-1 instability: {ranking['top1_instability']} / {ranking['n_non_baseline_conditions']} non-baseline conditions")


if __name__ == "__main__":
    main()
