#!/usr/bin/env python3
"""Decision robustness analysis: connecting rank flips to statistical uncertainty.

Computes how often model ranking changes correspond to statistically significant
differences vs. differences within noise. This addresses reviewer Weakness 4.

Usage:
    python -m src.analysis.decision_robustness \
        --aggregated results/runs/RUN_ID/aggregated.csv \
        --bootstrap results/runs/RUN_ID/bootstrap_deltas.csv \
        --output_dir results/runs/RUN_ID
"""

from __future__ import annotations

import argparse
import json
import logging
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def compute_decision_robustness(
    aggregated_path: Path,
    bootstrap_path: Path,
) -> dict:
    """Compute decision robustness metrics.

    Returns a dict with:
        - n_top1_changes: number of conditions where top-1 model differs from baseline
        - n_top1_significant: among those, how many have the new top-1 significantly
          better than the baseline's top-1 (CI of their delta excludes zero)
        - total_flips: total pairwise rank flips across all conditions
        - flips_within_ci: flips where the accuracy gap is within CI bounds
        - frac_top1_significant: n_top1_significant / n_top1_changes
        - frac_flips_within_ci: flips_within_ci / total_flips
        - per_pair_details: list of per-pair flip info
    """
    agg_df = pd.read_csv(aggregated_path)
    boot_df = pd.read_csv(bootstrap_path)

    # Baseline ranking
    baseline = agg_df[agg_df["condition_id"] == "baseline"].copy()
    baseline = baseline.sort_values("accuracy", ascending=False)
    baseline_ranking = {row["model_id"]: row["accuracy"] for _, row in baseline.iterrows()}
    baseline_order = list(baseline_ranking.keys())
    baseline_top1 = baseline_order[0]

    models = agg_df["model_id"].unique()
    non_baseline_conditions = [c for c in agg_df["condition_id"].unique() if c != "baseline"]

    # Build a lookup for bootstrap CIs: (model_id, condition_id) -> (ci_low, ci_high, delta)
    ci_lookup = {}
    for _, row in boot_df.iterrows():
        ci_lookup[(row["model_id"], row["condition_id"])] = {
            "delta": row["observed_delta"],
            "ci_low": row["ci_low"],
            "ci_high": row["ci_high"],
        }

    # Track top-1 changes and their significance
    n_top1_changes = 0
    n_top1_significant = 0
    top1_details = []

    # Track pairwise flips
    total_flips = 0
    flips_within_ci = 0
    per_pair_flips = {}

    for cond_id in non_baseline_conditions:
        cond_df = agg_df[agg_df["condition_id"] == cond_id].copy()
        cond_ranking = {row["model_id"]: row["accuracy"] for _, row in cond_df.iterrows()}

        # Top-1 under this condition
        cond_top1 = max(cond_ranking, key=cond_ranking.get)

        if cond_top1 != baseline_top1:
            n_top1_changes += 1

            # Is the new top-1 significantly better than baseline's top-1 under this condition?
            # We check if the new top-1's delta (vs baseline prompt) is significantly different
            # from the old top-1's delta. Since we don't have pairwise CIs between models,
            # we check: is the new top-1's accuracy under this condition minus the old top-1's
            # accuracy under this condition greater than the CI width would suggest?
            #
            # Simpler approach: check if both models' CIs for this condition overlap
            # in a way that their accuracy difference is within noise.
            new_top1_acc = cond_ranking.get(cond_top1, 0)
            old_top1_acc = cond_ranking.get(baseline_top1, 0)
            gap = new_top1_acc - old_top1_acc

            # Get CI widths for both models under this condition
            ci_new = ci_lookup.get((cond_top1, cond_id), {})
            ci_old = ci_lookup.get((baseline_top1, cond_id), {})

            # Conservative: the gap is "significant" if it exceeds the sum of
            # both models' CI half-widths (a rough Bonferroni-like criterion)
            if ci_new and ci_old:
                ci_new_width = (ci_new["ci_high"] - ci_new["ci_low"]) / 2
                ci_old_width = (ci_old["ci_high"] - ci_old["ci_low"]) / 2
                threshold = ci_new_width + ci_old_width
                is_sig = gap > threshold
            else:
                is_sig = False

            if is_sig:
                n_top1_significant += 1

            top1_details.append({
                "condition_id": cond_id,
                "baseline_top1": baseline_top1.split("/")[-1],
                "condition_top1": cond_top1.split("/")[-1],
                "accuracy_gap": round(gap, 4),
                "significant": is_sig,
            })

        # Pairwise flip analysis
        for m1, m2 in combinations(models, 2):
            b1 = baseline_ranking.get(m1, 0)
            b2 = baseline_ranking.get(m2, 0)
            c1 = cond_ranking.get(m1, 0)
            c2 = cond_ranking.get(m2, 0)

            baseline_order_pair = b1 > b2  # m1 ranked higher in baseline
            cond_order_pair = c1 > c2      # m1 ranked higher in condition

            if baseline_order_pair != cond_order_pair:
                total_flips += 1

                # Is this flip within CI bounds?
                # The accuracy gap between the two models under this condition
                acc_gap = abs(c1 - c2)

                # Get CI widths for both
                ci_m1 = ci_lookup.get((m1, cond_id), {})
                ci_m2 = ci_lookup.get((m2, cond_id), {})

                if ci_m1 and ci_m2:
                    ci_m1_width = (ci_m1["ci_high"] - ci_m1["ci_low"]) / 2
                    ci_m2_width = (ci_m2["ci_high"] - ci_m2["ci_low"]) / 2
                    combined_uncertainty = ci_m1_width + ci_m2_width
                    within_ci = acc_gap < combined_uncertainty
                else:
                    within_ci = True  # conservative: assume within CI if no data

                if within_ci:
                    flips_within_ci += 1

                pair_key = f"{m1.split('/')[-1]} vs {m2.split('/')[-1]}"
                if pair_key not in per_pair_flips:
                    per_pair_flips[pair_key] = {"total": 0, "within_ci": 0}
                per_pair_flips[pair_key]["total"] += 1
                if within_ci:
                    per_pair_flips[pair_key]["within_ci"] += 1

    result = {
        "n_top1_changes": n_top1_changes,
        "n_top1_significant": n_top1_significant,
        "frac_top1_significant": round(n_top1_significant / n_top1_changes, 3) if n_top1_changes > 0 else 0.0,
        "total_flips": total_flips,
        "flips_within_ci": flips_within_ci,
        "frac_flips_within_ci": round(flips_within_ci / total_flips, 3) if total_flips > 0 else 0.0,
        "top1_details": top1_details,
        "per_pair_flips": per_pair_flips,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decision robustness analysis."
    )
    parser.add_argument("--aggregated", type=Path, required=True)
    parser.add_argument("--bootstrap", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print("Computing decision robustness metrics ...")
    result = compute_decision_robustness(args.aggregated, args.bootstrap)

    out_path = args.output_dir / "decision_robustness.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    print(f"Saved decision robustness to {out_path}")

    # Print summary
    print(f"\n=== Decision Robustness Summary ===")
    print(f"Top-1 changes: {result['n_top1_changes']} conditions")
    print(f"  Significant: {result['n_top1_significant']} ({result['frac_top1_significant'] * 100:.0f}%)")
    print(f"Total rank flips: {result['total_flips']}")
    print(f"  Within CI: {result['flips_within_ci']} ({result['frac_flips_within_ci'] * 100:.0f}%)")

    if result["top1_details"]:
        print("\nTop-1 changes detail:")
        for d in result["top1_details"]:
            sig = "SIG" if d["significant"] else "n.s."
            print(f"  {d['condition_id']:25s}: {d['baseline_top1']} -> {d['condition_top1']} "
                  f"(gap={d['accuracy_gap']:+.4f}, {sig})")


if __name__ == "__main__":
    main()
