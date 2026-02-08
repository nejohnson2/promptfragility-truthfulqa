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


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate analysis plots.")
    parser.add_argument("--aggregated", type=Path, required=True)
    parser.add_argument("--ranking_metrics", type=Path, required=True)
    parser.add_argument("--model_summary", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
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
    print(f"\nAll figures saved to {args.output_dir}")


if __name__ == "__main__":
    main()
