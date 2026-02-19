#!/usr/bin/env python3
"""PSI sensitivity analysis: how stable are PSI estimates to perturbation subset size?

For each model, samples random subsets of k perturbations (k = 3, 5, 7, 9, 12)
from the 12 non-baseline conditions, computes PSI (baseline - worst in subset),
and reports the distribution. This directly addresses reviewer Question 1.

Usage:
    python -m src.analysis.psi_sensitivity \
        --aggregated results/runs/RUN_ID/aggregated.csv \
        --output_dir results/runs/RUN_ID
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Subset sizes to evaluate
K_VALUES = [3, 5, 7, 9, 12]
N_DRAWS = 1000
SEED = 42


def compute_psi_sensitivity(
    aggregated_path: Path,
    k_values: list[int] | None = None,
    n_draws: int = N_DRAWS,
    seed: int = SEED,
) -> pd.DataFrame:
    """Compute PSI sensitivity to perturbation subset size.

    Returns DataFrame with columns:
        model_id, k, psi_mean, psi_std, psi_p5, psi_p25, psi_median, psi_p75, psi_p95, psi_full
    """
    if k_values is None:
        k_values = K_VALUES

    agg_df = pd.read_csv(aggregated_path)
    rng = np.random.RandomState(seed)

    results = []
    for model_id in agg_df["model_id"].unique():
        mdf = agg_df[agg_df["model_id"] == model_id]

        # Get baseline accuracy
        baseline_row = mdf[mdf["condition_id"] == "baseline"]
        if baseline_row.empty:
            logger.warning(f"No baseline for {model_id}, skipping")
            continue
        baseline_acc = baseline_row["accuracy"].values[0]

        # Get non-baseline condition accuracies
        non_baseline = mdf[mdf["condition_id"] != "baseline"]
        condition_ids = non_baseline["condition_id"].values
        condition_accs = non_baseline["accuracy"].values
        n_conditions = len(condition_ids)

        # Full PSI (all perturbations)
        psi_full = baseline_acc - condition_accs.min()

        for k in k_values:
            if k > n_conditions:
                continue

            psi_samples = np.empty(n_draws)
            for i in range(n_draws):
                # Sample k conditions without replacement
                idx = rng.choice(n_conditions, size=k, replace=False)
                subset_accs = condition_accs[idx]
                psi_samples[i] = baseline_acc - subset_accs.min()

            results.append({
                "model_id": model_id,
                "k": k,
                "psi_mean": round(psi_samples.mean(), 6),
                "psi_std": round(psi_samples.std(), 6),
                "psi_p5": round(np.percentile(psi_samples, 5), 6),
                "psi_p25": round(np.percentile(psi_samples, 25), 6),
                "psi_median": round(np.median(psi_samples), 6),
                "psi_p75": round(np.percentile(psi_samples, 75), 6),
                "psi_p95": round(np.percentile(psi_samples, 95), 6),
                "psi_full": round(psi_full, 6),
            })

    return pd.DataFrame(results)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="PSI sensitivity to perturbation subset size."
    )
    parser.add_argument("--aggregated", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--n_draws", type=int, default=N_DRAWS)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Computing PSI sensitivity (n_draws={args.n_draws}, seed={args.seed}) ...")
    result_df = compute_psi_sensitivity(
        args.aggregated, n_draws=args.n_draws, seed=args.seed
    )

    out_path = args.output_dir / "psi_sensitivity.csv"
    result_df.to_csv(out_path, index=False)
    print(f"Saved PSI sensitivity to {out_path} ({len(result_df)} rows)")

    # Print summary
    print("\n=== PSI Sensitivity Summary ===")
    for model_id in result_df["model_id"].unique():
        mdf = result_df[result_df["model_id"] == model_id]
        full_psi = mdf["psi_full"].iloc[0]
        short = model_id.split("/")[-1]
        print(f"\n  {short} (full PSI = {full_psi * 100:.1f} pp):")
        for _, row in mdf.iterrows():
            print(
                f"    k={row['k']:2d}: "
                f"mean={row['psi_mean'] * 100:5.1f}  "
                f"std={row['psi_std'] * 100:4.1f}  "
                f"[p25={row['psi_p25'] * 100:5.1f}, p75={row['psi_p75'] * 100:5.1f}]"
            )


if __name__ == "__main__":
    main()
