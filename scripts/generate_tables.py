#!/usr/bin/env python3
"""Generate LaTeX table fragments and metric macros from analysis outputs.

Reads the CSV/JSON files produced by compute_metrics.py and stats.py,
then writes .tex fragments into paper/generated/ for \\input{} inclusion.

Usage:
    python scripts/generate_tables.py \
        --run_dir results/runs/merged_final_20260218_120244_job14605 \
        --output_dir paper/generated
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SHORT_NAMES = {
    "Qwen/Qwen2.5-3B-Instruct": "Qwen2.5-3B",
    "Qwen/Qwen2.5-7B-Instruct": "Qwen2.5-7B",
    "Qwen/Qwen2.5-72B-Instruct": "Qwen2.5-72B",
    "meta-llama/Llama-3.1-8B-Instruct": "Llama-3.1-8B",
    "meta-llama/Llama-3.1-70B-Instruct": "Llama-3.1-70B",
    "mistralai/Mistral-7B-Instruct-v0.3": "Mistral-7B-v0.3",
    "google/gemma-3-12b-it": "Gemma-3-12B",
    "microsoft/Phi-4-mini-instruct": "Phi-4-mini",
}

# Order models by descending baseline accuracy for consistent presentation
MODEL_ORDER = [
    "Qwen/Qwen2.5-72B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    "google/gemma-3-12b-it",
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct",
    "mistralai/Mistral-7B-Instruct-v0.3",
    "microsoft/Phi-4-mini-instruct",
    "Qwen/Qwen2.5-3B-Instruct",
]

CONDITION_ORDER = [
    "baseline",
    "expert",
    "cautious",
    "concise",
    "avoid_misconceptions",
    "numeric_labels",
    "single_line",
    "code_block",
    "distractor_sentence",
    "distractor_paragraph",
    "polite",
    "direct",
    "cot_constrained",
]

CONDITION_SHORT = {
    "baseline": "baseline",
    "expert": "expert",
    "cautious": "cautious",
    "concise": "concise",
    "avoid_misconceptions": "avoid\\_misc.",
    "numeric_labels": "numeric",
    "single_line": "single\\_line",
    "code_block": "code\\_block",
    "distractor_sentence": "dist\\_sent",
    "distractor_paragraph": "dist\\_para",
    "polite": "polite",
    "direct": "direct",
    "cot_constrained": "cot\\_constr.",
}


def short_name(model_id: str) -> str:
    return SHORT_NAMES.get(model_id, model_id.split("/")[-1])


def fmt_acc(v: float) -> str:
    """Format accuracy as .XXX (3 decimal places)."""
    return f".{int(round(v * 1000)):03d}"


def fmt_pct(v: float) -> str:
    """Format as X.X% (one decimal, no percent sign)."""
    return f"{v * 100:.1f}"


def fmt_delta(v: float) -> str:
    """Format delta as +.XXX or -.XXX."""
    sign = "+" if v >= 0 else "$-$"
    abs_v = abs(v)
    formatted = f".{int(round(abs_v * 1000)):03d}"
    if v >= 0:
        return f"+{formatted}"
    else:
        return f"$-${formatted}"


# ---------------------------------------------------------------------------
# Table generators
# ---------------------------------------------------------------------------

def generate_model_summary_table(summary_df: pd.DataFrame, output_dir: Path) -> None:
    """Generate the per-model summary table (Table 3 in paper)."""
    # Sort by baseline accuracy descending
    summary_df = summary_df.copy()
    summary_df["_order"] = summary_df["model_id"].map(
        {m: i for i, m in enumerate(MODEL_ORDER)}
    )
    summary_df = summary_df.sort_values("_order")

    lines = []
    for _, row in summary_df.iterrows():
        name = short_name(row["model_id"])
        line = (
            f"{name:20s} & {fmt_acc(row['baseline_acc'])} "
            f"& {fmt_acc(row['mean_acc'])} "
            f"& {fmt_acc(row['worst_acc'])} "
            f"& {fmt_acc(row['best_acc'])} "
            f"& {fmt_acc(row['PSI_drop'])} "
            f"& {fmt_pct(row['mean_invalid_rate'])} \\\\"
        )
        lines.append(line)

    # Include \bottomrule inside the generated file because \input{}
    # inside a tabular environment doesn't allow \bottomrule to follow
    # on the next line in the parent file (LaTeX scoping issue).
    lines.append("\\bottomrule")
    content = "\n".join(lines)
    (output_dir / "table_model_summary_rows.tex").write_text(content + "\n")
    print(f"  Wrote table_model_summary_rows.tex")


def generate_bootstrap_summary_table(
    bootstrap_df: pd.DataFrame, output_dir: Path
) -> None:
    """Generate the bootstrap summary table (Table 4 in paper)."""
    lines = []
    for model_id in MODEL_ORDER:
        mdf = bootstrap_df[bootstrap_df["model_id"] == model_id].copy()
        if mdf.empty:
            continue

        # Count significant conditions (CI excludes zero)
        mdf["sig"] = (mdf["ci_low"] > 0) | (mdf["ci_high"] < 0)
        n_sig = mdf["sig"].sum()
        n_total = len(mdf)

        # Worst delta (most negative)
        worst_row = mdf.loc[mdf["observed_delta"].idxmin()]
        worst_delta = worst_row["observed_delta"]
        worst_cond = CONDITION_SHORT.get(
            worst_row["condition_id"], worst_row["condition_id"]
        )

        # Best delta (most positive)
        best_row = mdf.loc[mdf["observed_delta"].idxmax()]
        best_delta = best_row["observed_delta"]
        best_cond = CONDITION_SHORT.get(
            best_row["condition_id"], best_row["condition_id"]
        )

        name = short_name(model_id)
        line = (
            f"{name:20s} & {n_sig}/{n_total} "
            f" & {fmt_delta(worst_delta)} (\\texttt{{{worst_cond}}})"
            f" & {fmt_delta(best_delta)} (\\texttt{{{best_cond}}}) \\\\"
        )
        lines.append(line)

    lines.append("\\bottomrule")
    content = "\n".join(lines)
    (output_dir / "table_bootstrap_summary_rows.tex").write_text(content + "\n")
    print(f"  Wrote table_bootstrap_summary_rows.tex")


def generate_full_bootstrap_table(
    bootstrap_df: pd.DataFrame, output_dir: Path
) -> None:
    """Generate the full bootstrap delta table (Appendix C)."""
    lines = []
    for model_id in MODEL_ORDER:
        mdf = bootstrap_df[bootstrap_df["model_id"] == model_id].copy()
        if mdf.empty:
            continue

        mdf["sig"] = (mdf["ci_low"] > 0) | (mdf["ci_high"] < 0)
        # Sort by delta descending
        mdf = mdf.sort_values("observed_delta", ascending=False)

        name = short_name(model_id)
        first = True
        for _, row in mdf.iterrows():
            cond = CONDITION_SHORT.get(row["condition_id"], row["condition_id"])
            sig_mark = "*" if row["sig"] else ""

            delta_str = fmt_delta(row["observed_delta"]) + sig_mark

            ci_low_sign = "+" if row["ci_low"] >= 0 else "$-$"
            ci_low_abs = f".{int(round(abs(row['ci_low']) * 1000)):03d}"
            ci_low_str = f"{ci_low_sign}{ci_low_abs}" if row["ci_low"] >= 0 else f"{ci_low_sign}{ci_low_abs}"

            ci_high_sign = "+" if row["ci_high"] >= 0 else "$-$"
            ci_high_abs = f".{int(round(abs(row['ci_high']) * 1000)):03d}"
            ci_high_str = f"{ci_high_sign}{ci_high_abs}" if row["ci_high"] >= 0 else f"{ci_high_sign}{ci_high_abs}"

            if first:
                model_col = f"\\multirow{{{len(mdf)}}}{{*}}{{{name}}}"
                first = False
            else:
                model_col = ""

            line = (
                f" {model_col} & \\texttt{{{cond}}} "
                f"& {delta_str} & {ci_low_str} & {ci_high_str} \\\\"
            )
            lines.append(line)

        lines.append("\\midrule")

    # Remove trailing \midrule
    if lines and lines[-1] == "\\midrule":
        lines[-1] = "\\bottomrule"

    content = "\n".join(lines)
    (output_dir / "table_full_bootstrap_rows.tex").write_text(content + "\n")
    print(f"  Wrote table_full_bootstrap_rows.tex")


def generate_metrics_macros(
    summary_df: pd.DataFrame,
    ranking_metrics: dict,
    bootstrap_df: pd.DataFrame,
    aggregated_df: pd.DataFrame,
    output_dir: Path,
    decision_robustness: dict | None = None,
) -> None:
    """Generate LaTeX macro definitions for inline numeric citations.

    This is the key file: every number cited in prose comes from here,
    so re-running this script keeps the paper in sync with results.
    """
    macros = []

    def add(name: str, value: str):
        macros.append(f"\\newcommand{{\\{name}}}{{{value}}}")

    # ── Dataset ──
    n_questions = int(aggregated_df["n"].iloc[0]) if "n" in aggregated_df.columns else 736
    add("nQuestions", str(n_questions))
    add("nModels", str(len(summary_df)))
    n_conditions = ranking_metrics.get("n_conditions", 13)
    add("nConditions", str(n_conditions))
    n_non_baseline = ranking_metrics.get("n_non_baseline_conditions", 12)
    add("nPerturbations", str(n_non_baseline))

    # ── Global accuracy range ──
    acc_min = aggregated_df["accuracy"].min()
    acc_max = aggregated_df["accuracy"].max()
    add("globalAccMin", f"{acc_min * 100:.1f}")
    add("globalAccMax", f"{acc_max * 100:.1f}")

    # Find model+condition for min and max
    min_row = aggregated_df.loc[aggregated_df["accuracy"].idxmin()]
    max_row = aggregated_df.loc[aggregated_df["accuracy"].idxmax()]
    add("globalAccMinModel", short_name(min_row["model_id"]))
    add("globalAccMinCond", CONDITION_SHORT.get(min_row["condition_id"], min_row["condition_id"]))
    add("globalAccMaxModel", short_name(max_row["model_id"]))
    add("globalAccMaxCond", CONDITION_SHORT.get(max_row["condition_id"], max_row["condition_id"]))

    # ── PSI range ──
    psi_min = summary_df["PSI_drop"].min()
    psi_max = summary_df["PSI_drop"].max()
    psi_mean = summary_df["PSI_drop"].mean()
    add("PSImin", f"{psi_min * 100:.1f}")
    add("PSImax", f"{psi_max * 100:.1f}")
    add("PSImean", f"{psi_mean * 100:.1f}")

    psi_min_model = short_name(summary_df.loc[summary_df["PSI_drop"].idxmin(), "model_id"])
    psi_max_model = short_name(summary_df.loc[summary_df["PSI_drop"].idxmax(), "model_id"])
    add("PSIminModel", psi_min_model)
    add("PSImaxModel", psi_max_model)

    # PSI for specific models (for prose)
    # Use "val" prefix to avoid collisions with LaTeX builtins like \psi, \phi
    # LaTeX macro names CANNOT contain digits — use manually curated short keys
    model_keys = {
        "Qwen2.5-3B": "QwenS",
        "Qwen2.5-7B": "QwenM",
        "Qwen2.5-72B": "QwenL",
        "Llama-3.1-8B": "LlamaM",
        "Llama-3.1-70B": "LlamaL",
        "Mistral-7B-v0.3": "Mistral",
        "Gemma-3-12B": "Gemma",
        "Phi-4-mini": "Phi",
    }
    for _, row in summary_df.iterrows():
        sname = short_name(row["model_id"])
        key = model_keys.get(sname, sname.replace("-", "").replace(".", "").replace("_", ""))
        add(f"valPSI{key}", f"{row['PSI_drop'] * 100:.1f}")
        add(f"valBaseline{key}", f"{row['baseline_acc'] * 100:.1f}")
        add(f"valWorst{key}", f"{row['worst_acc'] * 100:.1f}")
        add(f"valBest{key}", f"{row['best_acc'] * 100:.1f}")
        add(f"valMean{key}", f"{row['mean_acc'] * 100:.1f}")
        add(f"valInvalid{key}", f"{row['mean_invalid_rate'] * 100:.1f}")

    # ── Accuracy range (mean across models) ──
    mean_range = summary_df["range_acc"].mean()
    add("meanAccRange", f"{mean_range * 100:.1f}")

    # ── Ranking metrics ──
    taus = ranking_metrics.get("kendall_tau", {})
    non_baseline_taus = {k: v for k, v in taus.items() if k != "baseline"}
    if non_baseline_taus:
        mean_tau = np.mean(list(non_baseline_taus.values()))
        min_tau = min(non_baseline_taus.values())
        max_tau = max(non_baseline_taus.values())
        add("meanKendallTau", f"{mean_tau:.3f}")
        add("minKendallTau", f"{min_tau:.3f}")
        add("maxKendallTau", f"{max_tau:.3f}")

    rank_flip_rate = ranking_metrics.get("rank_flip_rate", 0.0)
    add("rankFlipRate", f"{rank_flip_rate * 100:.1f}")

    top1_instability = ranking_metrics.get("top1_instability", 0)
    add("topOneInstability", str(top1_instability))

    # ── Bootstrap significance counts ──
    bootstrap_df = bootstrap_df.copy()
    bootstrap_df["sig"] = (bootstrap_df["ci_low"] > 0) | (bootstrap_df["ci_high"] < 0)
    n_sig = int(bootstrap_df["sig"].sum())
    n_total = len(bootstrap_df)
    sig_pct = n_sig / n_total * 100 if n_total > 0 else 0
    add("nSigDeltas", str(n_sig))
    add("nTotalDeltas", str(n_total))
    add("sigDeltaPct", f"{sig_pct:.1f}")

    # ── Per-condition significance counts ──
    for cond_id in bootstrap_df["condition_id"].unique():
        cdf = bootstrap_df[bootstrap_df["condition_id"] == cond_id]
        n_sig_cond = int(cdf["sig"].sum())
        safe_cond = cond_id.replace("_", "")
        add(f"nSig{safe_cond}", str(n_sig_cond))

    # ── Avoid-misconceptions effect ──
    am_df = bootstrap_df[bootstrap_df["condition_id"] == "avoid_misconceptions"]
    am_min_delta = am_df["observed_delta"].min()
    am_max_delta = am_df["observed_delta"].max()
    add("avoidMiscMinGain", f"{am_min_delta * 100:.1f}")
    add("avoidMiscMaxGain", f"{am_max_delta * 100:.1f}")

    # ── Largest single delta ──
    worst_overall = bootstrap_df.loc[bootstrap_df["observed_delta"].idxmin()]
    add("worstSingleDelta", f"{abs(worst_overall['observed_delta']) * 100:.1f}")
    add("worstSingleModel", short_name(worst_overall["model_id"]))
    add("worstSingleCond", CONDITION_SHORT.get(worst_overall["condition_id"], worst_overall["condition_id"]))

    best_overall = bootstrap_df.loc[bootstrap_df["observed_delta"].idxmax()]
    add("bestSingleDelta", f"{best_overall['observed_delta'] * 100:.1f}")
    add("bestSingleModel", short_name(best_overall["model_id"]))
    add("bestSingleCond", CONDITION_SHORT.get(best_overall["condition_id"], best_overall["condition_id"]))

    # ── Category-level mean deltas ──
    category_map = {
        "expert": "instruction_wording",
        "cautious": "instruction_wording",
        "concise": "instruction_wording",
        "avoid_misconceptions": "instruction_wording",
        "numeric_labels": "formatting",
        "single_line": "formatting",
        "code_block": "formatting",
        "distractor_sentence": "benign_distractor",
        "distractor_paragraph": "benign_distractor",
        "polite": "social_style",
        "direct": "social_style",
        "cot_constrained": "stress",
    }
    bootstrap_df["category"] = bootstrap_df["condition_id"].map(category_map)
    for cat in ["instruction_wording", "formatting", "benign_distractor", "social_style", "stress"]:
        cat_mean = bootstrap_df[bootstrap_df["category"] == cat]["observed_delta"].mean()
        safe_cat = cat.replace("_", "")
        add(f"catMeanDelta{safe_cat}", f"{cat_mean * 100:+.1f}")

    # ── IQR and MAD auxiliary robustness metrics ──
    if "PSI_iqr" in summary_df.columns:
        iqr_min = summary_df["PSI_iqr"].min()
        iqr_max = summary_df["PSI_iqr"].max()
        iqr_mean = summary_df["PSI_iqr"].mean()
        add("PSIiqrMin", f"{iqr_min * 100:.1f}")
        add("PSIiqrMax", f"{iqr_max * 100:.1f}")
        add("PSIiqrMean", f"{iqr_mean * 100:.1f}")

        mad_min = summary_df["PSI_mad"].min()
        mad_max = summary_df["PSI_mad"].max()
        mad_mean = summary_df["PSI_mad"].mean()
        add("PSImadMin", f"{mad_min * 100:.1f}")
        add("PSImadMax", f"{mad_max * 100:.1f}")
        add("PSImadMean", f"{mad_mean * 100:.1f}")

    # ── Decision robustness metrics ──
    if decision_robustness:
        add("nTopOneChanges", str(decision_robustness.get("n_top1_changes", 0)))
        add("nTopOneSig", str(decision_robustness.get("n_top1_significant", 0)))
        frac_top1 = decision_robustness.get("frac_top1_significant", 0.0)
        add("fracTopOneSig", f"{frac_top1 * 100:.0f}")
        add("totalFlips", str(decision_robustness.get("total_flips", 0)))
        add("flipsWithinCI", str(decision_robustness.get("flips_within_ci", 0)))
        frac_within = decision_robustness.get("frac_flips_within_ci", 0.0)
        add("fracFlipsWithinCI", f"{frac_within * 100:.0f}")

    content = "% Auto-generated by scripts/generate_tables.py — DO NOT EDIT\n"
    content += "% Re-run: make paper\n"
    content += "\n".join(macros)
    content += "\n"

    (output_dir / "metrics_macros.tex").write_text(content)
    print(f"  Wrote metrics_macros.tex ({len(macros)} macros)")


def generate_category_analysis_table(
    bootstrap_df: pd.DataFrame, aggregated_df: pd.DataFrame, output_dir: Path
) -> None:
    """Generate a per-category mean delta table (new Table 5)."""
    category_map = {
        "expert": "Instruction wording",
        "cautious": "Instruction wording",
        "concise": "Instruction wording",
        "avoid_misconceptions": "Instruction wording",
        "numeric_labels": "Formatting",
        "single_line": "Formatting",
        "code_block": "Formatting",
        "distractor_sentence": "Benign distractor",
        "distractor_paragraph": "Benign distractor",
        "polite": "Social style",
        "direct": "Social style",
        "cot_constrained": "Stress",
    }

    bdf = bootstrap_df.copy()
    bdf["category"] = bdf["condition_id"].map(category_map)
    bdf["sig"] = (bdf["ci_low"] > 0) | (bdf["ci_high"] < 0)

    lines = []
    cat_order = [
        "Instruction wording",
        "Formatting",
        "Benign distractor",
        "Social style",
        "Stress",
    ]
    for cat in cat_order:
        cdf = bdf[bdf["category"] == cat]
        n_conds = cdf["condition_id"].nunique()
        mean_delta = cdf["observed_delta"].mean()
        min_delta = cdf["observed_delta"].min()
        max_delta = cdf["observed_delta"].max()
        n_sig = int(cdf["sig"].sum())
        n_total = len(cdf)

        line = (
            f"{cat:22s} & {n_conds} "
            f"& {mean_delta * 100:+.1f} "
            f"& {min_delta * 100:+.1f} "
            f"& {max_delta * 100:+.1f} "
            f"& {n_sig}/{n_total} \\\\"
        )
        lines.append(line)

    lines.append("\\bottomrule")
    content = "\n".join(lines)
    (output_dir / "table_category_analysis_rows.tex").write_text(content + "\n")
    print(f"  Wrote table_category_analysis_rows.tex")


def generate_parse_ablation_table(
    parse_ablation_df: pd.DataFrame, output_dir: Path
) -> None:
    """Generate the parse stage ablation table (Appendix)."""
    # Aggregate across models: per condition, per stage
    agg = parse_ablation_df.groupby(["condition_id", "stage"])["count"].sum().reset_index()
    totals = agg.groupby("condition_id")["count"].sum().reset_index()
    totals.columns = ["condition_id", "total"]
    agg = agg.merge(totals, on="condition_id")
    agg["pct"] = agg["count"] / agg["total"] * 100

    stages = ["answer_pattern", "exact_token", "label_punctuation", "word_boundary", "invalid"]

    lines = []
    for cond_id in CONDITION_ORDER:
        cdf = agg[agg["condition_id"] == cond_id]
        if cdf.empty:
            continue
        cond_short = CONDITION_SHORT.get(cond_id, cond_id)
        vals = []
        for stage in stages:
            row = cdf[cdf["stage"] == stage]
            pct = row["pct"].values[0] if len(row) > 0 else 0.0
            vals.append(f"{pct:.1f}")
        line = f"\\texttt{{{cond_short}}} & " + " & ".join(vals) + " \\\\"
        lines.append(line)

    lines.append("\\bottomrule")
    content = "\n".join(lines)
    (output_dir / "table_parse_ablation_rows.tex").write_text(content + "\n")
    print(f"  Wrote table_parse_ablation_rows.tex")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate LaTeX tables and macros from analysis outputs."
    )
    parser.add_argument(
        "--run_dir",
        type=Path,
        required=True,
        help="Path to merged run directory (e.g., results/runs/merged_final_...)",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("paper/generated"),
        help="Output directory for .tex fragments (default: paper/generated)",
    )
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load all analysis outputs
    aggregated_path = args.run_dir / "aggregated.csv"
    summary_path = args.run_dir / "model_summary.csv"
    ranking_path = args.run_dir / "ranking_metrics.json"
    bootstrap_path = args.run_dir / "bootstrap_deltas.csv"

    for p in [aggregated_path, summary_path, ranking_path, bootstrap_path]:
        if not p.exists():
            print(f"ERROR: Missing {p}")
            raise SystemExit(1)

    aggregated_df = pd.read_csv(aggregated_path)
    summary_df = pd.read_csv(summary_path)
    with open(ranking_path) as f:
        ranking_metrics = json.load(f)
    bootstrap_df = pd.read_csv(bootstrap_path)

    # Optional new analysis outputs
    decision_robustness_path = args.run_dir / "decision_robustness.json"
    decision_robustness = None
    if decision_robustness_path.exists():
        with open(decision_robustness_path) as f:
            decision_robustness = json.load(f)
        print(f"  Loaded decision_robustness.json")

    parse_ablation_path = args.run_dir / "parse_ablation.csv"
    parse_ablation_df = None
    if parse_ablation_path.exists():
        parse_ablation_df = pd.read_csv(parse_ablation_path)
        print(f"  Loaded parse_ablation.csv ({len(parse_ablation_df)} rows)")

    print(f"Generating LaTeX fragments from {args.run_dir} ...")
    generate_model_summary_table(summary_df, args.output_dir)
    generate_bootstrap_summary_table(bootstrap_df, args.output_dir)
    generate_full_bootstrap_table(bootstrap_df, args.output_dir)
    generate_metrics_macros(
        summary_df, ranking_metrics, bootstrap_df, aggregated_df, args.output_dir,
        decision_robustness=decision_robustness,
    )
    generate_category_analysis_table(bootstrap_df, aggregated_df, args.output_dir)

    if parse_ablation_df is not None:
        generate_parse_ablation_table(parse_ablation_df, args.output_dir)

    print(f"\nAll fragments written to {args.output_dir}/")


if __name__ == "__main__":
    main()
