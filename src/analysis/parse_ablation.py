#!/usr/bin/env python3
"""Retroactive parse stage ablation analysis.

Re-parses every prediction's output_text through the staged parser to
determine which regex stage matched, revealing whether conditions primarily
affect format compliance (parseability) vs. model beliefs (accuracy).

Usage:
    python -m src.analysis.parse_ablation \
        --predictions results/runs/RUN_ID/predictions.jsonl \
        --output_dir results/runs/RUN_ID
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import pandas as pd

from src.eval.parse_answer import parse_mc_answer_staged
from src.prompts.render import _format_options, load_conditions

logger = logging.getLogger(__name__)


def _build_label_maps(conditions: list[dict]) -> dict[str, callable]:
    """Build a function per condition that generates label_map from a choices list.

    For the parse ablation, we need the label_map for each condition.
    The label_map depends on (label_style, n_choices). Since all TruthfulQA
    questions have variable numbers of choices, we build label_maps on the fly.
    """
    condition_vars = {}
    for cond in conditions:
        tv = cond["template_vars"]
        condition_vars[cond["id"]] = {
            "label_style": tv["label_style"],
            "option_separator": tv["option_separator"],
        }
    return condition_vars


def _make_label_map(n_choices: int, label_style: str) -> dict[str, str]:
    """Build a label_map for n_choices under the given label style."""
    label_map: dict[str, str] = {}
    for i in range(n_choices):
        if label_style == "numeric":
            displayed = str(i + 1)
        else:
            displayed = chr(ord("A") + i)
        canonical = chr(ord("A") + i)
        label_map[displayed] = canonical
    return label_map


def compute_parse_ablation(
    predictions_path: Path,
    conditions_path: Path | None = None,
) -> pd.DataFrame:
    """Re-parse all predictions and classify by parse stage.

    Returns a DataFrame with columns:
        model_id, condition_id, stage, count, pct, accuracy_parsed
    """
    from src.config import CONDITIONS_PATH

    if conditions_path is None:
        conditions_path = CONDITIONS_PATH

    conditions = load_conditions(conditions_path)
    cond_vars = _build_label_maps(conditions)

    # Load predictions
    records = []
    with open(predictions_path) as f:
        for line in f:
            records.append(json.loads(line))
    df = pd.DataFrame(records)
    logger.info(f"Loaded {len(df)} predictions")

    # Determine n_choices per question from the output (we need to infer
    # from correct_label: if correct_label is 'D', there are at least 4 choices)
    # Actually, we can determine from the prompt_text by counting option lines.
    # But the simplest approach: use 4 as default (TruthfulQA MC1 has variable
    # numbers of choices). Let's count from the prompt text.

    staged_results = []
    for _, row in df.iterrows():
        cond_id = row["condition_id"]
        if cond_id not in cond_vars:
            logger.warning(f"Unknown condition {cond_id}, skipping")
            continue

        cv = cond_vars[cond_id]
        label_style = cv["label_style"]

        # Infer n_choices: count how many option labels appear in prompt_text
        # Look for "A)" or "1)" patterns
        prompt = row["prompt_text"]
        if label_style == "numeric":
            n_choices = sum(1 for i in range(1, 20) if f"{i})" in prompt)
        else:
            n_choices = sum(
                1 for c in "ABCDEFGHIJKLMNOP" if f"{c})" in prompt
            )
        # Fallback: at least 2 choices
        n_choices = max(n_choices, 2)

        label_map = _make_label_map(n_choices, label_style)
        parsed_label, is_invalid, stage = parse_mc_answer_staged(
            row["output_text"], label_map
        )

        staged_results.append({
            "model_id": row["model_id"],
            "condition_id": cond_id,
            "question_id": row["question_id"],
            "stage": stage,
            "parsed_label": parsed_label,
            "correct_label": row["correct_label"],
            "is_correct": parsed_label == row["correct_label"] if parsed_label else False,
            "is_invalid": is_invalid,
        })

    staged_df = pd.DataFrame(staged_results)

    # Aggregate: per (model_id, condition_id, stage) counts and pct
    agg_rows = []
    for (model_id, cond_id), grp in staged_df.groupby(["model_id", "condition_id"]):
        total = len(grp)
        for stage_name, stage_grp in grp.groupby("stage"):
            count = len(stage_grp)
            pct = count / total if total > 0 else 0.0
            # Accuracy among successfully parsed answers at this stage
            parsed_only = stage_grp[~stage_grp["is_invalid"]]
            acc_parsed = parsed_only["is_correct"].mean() if len(parsed_only) > 0 else float("nan")

            agg_rows.append({
                "model_id": model_id,
                "condition_id": cond_id,
                "stage": stage_name,
                "count": count,
                "pct": round(pct, 4),
                "accuracy_parsed": round(acc_parsed, 4) if not pd.isna(acc_parsed) else None,
            })

    return pd.DataFrame(agg_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Retroactive parse stage ablation."
    )
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output_dir", type=Path, required=True)
    parser.add_argument("--conditions", type=Path, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Running parse stage ablation on {args.predictions} ...")
    result_df = compute_parse_ablation(args.predictions, args.conditions)

    out_path = args.output_dir / "parse_ablation.csv"
    result_df.to_csv(out_path, index=False)
    print(f"Saved parse ablation to {out_path} ({len(result_df)} rows)")

    # Print summary by condition
    print("\n=== Parse Stage Distribution by Condition ===")
    for cond_id in result_df["condition_id"].unique():
        cdf = result_df[result_df["condition_id"] == cond_id]
        # Aggregate across models
        stage_totals = cdf.groupby("stage")["count"].sum()
        total = stage_totals.sum()
        print(f"\n  {cond_id}:")
        for stage in ["answer_pattern", "exact_token", "label_punctuation", "word_boundary", "invalid"]:
            cnt = stage_totals.get(stage, 0)
            pct = cnt / total * 100 if total > 0 else 0
            print(f"    {stage:20s}: {cnt:5d} ({pct:5.1f}%)")


if __name__ == "__main__":
    main()
