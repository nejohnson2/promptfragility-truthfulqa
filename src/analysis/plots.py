#!/usr/bin/env python3
"""Generate figures for the prompt fragility analysis.

Usage:
    python -m src.analysis.plots \
        --aggregated results/runs/RUN_ID/aggregated.csv \
        --ranking_metrics results/runs/RUN_ID/ranking_metrics.json \
        --model_summary results/runs/RUN_ID/model_summary.csv \
        --output_dir results/runs/RUN_ID/figures
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


def plot_accuracy_heatmap(df: pd.DataFrame, output_dir: Path) -> None:
    """Heatmap of accuracy: models × conditions."""
    pivot = df.pivot(index="model_id", columns="condition_id", values="accuracy")

    # Shorten model names for readability
    pivot.index = [m.split("/")[-1] for m in pivot.index]

    fig, ax = plt.subplots(figsize=(14, max(4, len(pivot) * 0.8)))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".3f",
        cmap="RdYlGn",
        ax=ax,
        vmin=0,
        vmax=1,
        linewidths=0.5,
    )
    ax.set_title("Accuracy by Model × Condition")
    ax.set_ylabel("Model")
    ax.set_xlabel("Condition")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_dir / "accuracy_heatmap.png", dpi=150)
    plt.close(fig)
    print(f"  Saved accuracy_heatmap.png")


def plot_accuracy_bars(df: pd.DataFrame, output_dir: Path) -> None:
    """Bar chart per model showing accuracy across conditions."""
    models = df["model_id"].unique()
    conditions = df["condition_id"].unique()

    n_models = len(models)
    fig, axes = plt.subplots(n_models, 1, figsize=(12, 3 * n_models), sharex=True)
    if n_models == 1:
        axes = [axes]

    for ax, model_id in zip(axes, models):
        model_df = df[df["model_id"] == model_id].copy()
        model_df = model_df.sort_values("condition_id")

        colors = ["steelblue" if c != "baseline" else "darkorange" for c in model_df["condition_id"]]
        ax.bar(model_df["condition_id"], model_df["accuracy"], color=colors)
        ax.set_ylabel("Accuracy")
        ax.set_title(model_id.split("/")[-1])
        ax.set_ylim(0, 1)
        ax.axhline(y=model_df[model_df["condition_id"] == "baseline"]["accuracy"].values[0],
                    color="darkorange", linestyle="--", alpha=0.7, label="baseline")

    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_dir / "accuracy_bars.png", dpi=150)
    plt.close(fig)
    print(f"  Saved accuracy_bars.png")


def plot_kendall_tau(ranking_metrics: dict, output_dir: Path) -> None:
    """Bar chart of Kendall τ per condition."""
    taus = ranking_metrics.get("kendall_tau", {})
    if not taus:
        print("  Skipping Kendall τ plot — no data.")
        return

    conditions = list(taus.keys())
    values = [taus[c] for c in conditions]

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = ["steelblue" if c != "baseline" else "darkorange" for c in conditions]
    ax.bar(conditions, values, color=colors)
    ax.set_ylabel("Kendall τ (vs baseline ranking)")
    ax.set_xlabel("Condition")
    ax.set_title("Ranking Stability: Kendall τ per Condition")
    ax.set_ylim(-1, 1)
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(output_dir / "kendall_tau.png", dpi=150)
    plt.close(fig)
    print(f"  Saved kendall_tau.png")


def plot_baseline_vs_worst(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """Scatter plot of baseline accuracy vs worst-case accuracy per model."""
    fig, ax = plt.subplots(figsize=(7, 7))

    short_names = [m.split("/")[-1] for m in summary_df["model_id"]]

    ax.scatter(summary_df["baseline_acc"], summary_df["worst_acc"], s=100, zorder=3)

    for i, name in enumerate(short_names):
        ax.annotate(
            name,
            (summary_df["baseline_acc"].iloc[i], summary_df["worst_acc"].iloc[i]),
            textcoords="offset points",
            xytext=(8, 4),
            fontsize=8,
        )

    # Diagonal line (perfect stability)
    lims = [0, max(summary_df["baseline_acc"].max(), summary_df["worst_acc"].max()) + 0.05]
    ax.plot(lims, lims, "k--", alpha=0.3, label="y=x (no drop)")
    ax.set_xlabel("Baseline Accuracy")
    ax.set_ylabel("Worst-Case Accuracy")
    ax.set_title("Baseline vs Worst-Case Accuracy")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend()
    plt.tight_layout()
    fig.savefig(output_dir / "baseline_vs_worst.png", dpi=150)
    plt.close(fig)
    print(f"  Saved baseline_vs_worst.png")


def plot_psi_sensitivity(psi_sensitivity_path: Path, output_dir: Path) -> None:
    """Line plot showing PSI convergence as perturbation subset size increases."""
    df = pd.read_csv(psi_sensitivity_path)
    if df.empty:
        print("  Skipping PSI sensitivity plot — no data.")
        return

    models = df["model_id"].unique()
    fig, ax = plt.subplots(figsize=(10, 6))

    cmap = plt.cm.tab10
    for i, model_id in enumerate(sorted(models)):
        mdf = df[df["model_id"] == model_id].sort_values("k")
        short = model_id.split("/")[-1]
        color = cmap(i / max(len(models) - 1, 1))

        ax.plot(mdf["k"], mdf["psi_mean"] * 100, "o-", color=color, label=short, linewidth=1.5, markersize=5)
        ax.fill_between(
            mdf["k"],
            mdf["psi_p25"] * 100,
            mdf["psi_p75"] * 100,
            alpha=0.15,
            color=color,
        )

    ax.set_xlabel("Number of perturbation conditions sampled ($k$)", fontsize=12)
    ax.set_ylabel("PSI (percentage points)", fontsize=12)
    ax.set_title("PSI Sensitivity to Perturbation Subset Size", fontsize=13)
    ax.legend(fontsize=8, loc="lower right", ncol=2)
    ax.set_xticks([3, 5, 7, 9, 12])
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    fig.savefig(output_dir / "psi_subset_sensitivity.png", dpi=150)
    plt.close(fig)
    print(f"  Saved psi_subset_sensitivity.png")


def plot_parse_stage_distribution(parse_ablation_path: Path, output_dir: Path) -> None:
    """Stacked bar chart showing parse stage distribution by condition."""
    df = pd.read_csv(parse_ablation_path)
    if df.empty:
        print("  Skipping parse stage plot — no data.")
        return

    # Aggregate across models: total counts per (condition, stage)
    agg = df.groupby(["condition_id", "stage"])["count"].sum().reset_index()
    totals = agg.groupby("condition_id")["count"].sum().reset_index()
    totals.columns = ["condition_id", "total"]
    agg = agg.merge(totals, on="condition_id")
    agg["pct"] = agg["count"] / agg["total"] * 100

    # Pivot for stacked bar
    stages_order = ["answer_pattern", "exact_token", "label_punctuation", "word_boundary", "invalid"]
    stage_labels = {
        "answer_pattern": "Answer: X",
        "exact_token": "Exact token",
        "label_punctuation": "Label+punct",
        "word_boundary": "Word boundary",
        "invalid": "Invalid",
    }
    stage_colors = {
        "answer_pattern": "#2ecc71",
        "exact_token": "#3498db",
        "label_punctuation": "#9b59b6",
        "word_boundary": "#e67e22",
        "invalid": "#e74c3c",
    }

    conditions = sorted(agg["condition_id"].unique())
    fig, ax = plt.subplots(figsize=(12, 6))

    bottom = np.zeros(len(conditions))
    for stage in stages_order:
        vals = []
        for cond in conditions:
            row = agg[(agg["condition_id"] == cond) & (agg["stage"] == stage)]
            vals.append(row["pct"].values[0] if len(row) > 0 else 0.0)
        ax.bar(conditions, vals, bottom=bottom, label=stage_labels[stage],
               color=stage_colors[stage], edgecolor="white", linewidth=0.5)
        bottom += np.array(vals)

    ax.set_ylabel("Percentage of outputs (%)", fontsize=12)
    ax.set_xlabel("Prompt condition", fontsize=12)
    ax.set_title("Parse Stage Distribution by Condition (All Models)", fontsize=13)
    ax.legend(fontsize=9, loc="upper right")
    plt.xticks(rotation=45, ha="right")
    ax.set_ylim(0, 105)
    plt.tight_layout()
    fig.savefig(output_dir / "parse_stage_distribution.png", dpi=150)
    plt.close(fig)
    print(f"  Saved parse_stage_distribution.png")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate analysis plots.")
    parser.add_argument("--aggregated", type=Path, required=True)
    parser.add_argument("--ranking_metrics", type=Path, required=True)
    parser.add_argument("--model_summary", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--psi_sensitivity", type=Path, default=None,
                        help="Path to psi_sensitivity.csv (optional)")
    parser.add_argument("--parse_ablation", type=Path, default=None,
                        help="Path to parse_ablation.csv (optional)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.aggregated)
    summary_df = pd.read_csv(args.model_summary)
    with open(args.ranking_metrics) as f:
        ranking_metrics = json.load(f)

    print("Generating figures …")
    plot_accuracy_heatmap(df, args.output_dir)
    plot_accuracy_bars(df, args.output_dir)
    plot_kendall_tau(ranking_metrics, args.output_dir)
    plot_baseline_vs_worst(summary_df, args.output_dir)

    # New figures (optional — only if analysis outputs exist)
    if args.psi_sensitivity and args.psi_sensitivity.exists():
        plot_psi_sensitivity(args.psi_sensitivity, args.output_dir)
    if args.parse_ablation and args.parse_ablation.exists():
        plot_parse_stage_distribution(args.parse_ablation, args.output_dir)

    print(f"\nAll figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
