"""Smoke test: write a small synthetic dataset, run analysis pipeline on it."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def _make_synthetic_predictions(n_questions: int = 5) -> list[dict]:
    """Create synthetic prediction records for testing the analysis pipeline."""
    import random

    rng = random.Random(99)
    models = ["model_a", "model_b"]
    conditions = ["baseline", "expert", "concise"]
    records = []

    for model in models:
        for cond in conditions:
            for qid in range(n_questions):
                is_correct = rng.random() > 0.4
                records.append({
                    "run_id": "smoke_test",
                    "timestamp": "2025-01-01T00:00:00Z",
                    "model_id": model,
                    "condition_id": cond,
                    "question_id": qid,
                    "split": "dev",
                    "prompt_text": f"Question {qid}",
                    "output_text": "A",
                    "parsed_label": "A" if is_correct else "B",
                    "correct_label": "A",
                    "is_correct": is_correct,
                    "is_invalid": False,
                    "token_count": 1,
                    "latency_ms": 10.0,
                    "seed": 42,
                    "decoding_params": {},
                })
    return records


class TestAnalysisPipeline:
    """Run the analysis modules on synthetic data."""

    def test_compute_metrics(self) -> None:
        from src.analysis.compute_metrics import (
            compute_accuracy_table,
            compute_model_summary,
            compute_ranking_metrics,
        )
        import pandas as pd

        records = _make_synthetic_predictions()
        df = pd.DataFrame(records)

        acc_table = compute_accuracy_table(df)
        assert len(acc_table) == 6  # 2 models × 3 conditions
        assert "accuracy" in acc_table.columns
        assert "invalid_rate" in acc_table.columns

        summary = compute_model_summary(acc_table)
        assert len(summary) == 2
        assert "PSI_drop" in summary.columns
        assert "baseline_acc" in summary.columns

        ranking = compute_ranking_metrics(acc_table)
        assert "kendall_tau" in ranking
        assert "rank_flip_rate" in ranking
        assert "top1_instability" in ranking

    def test_bootstrap_deltas(self) -> None:
        from src.analysis.stats import compute_all_deltas
        import pandas as pd

        records = _make_synthetic_predictions(n_questions=20)
        df = pd.DataFrame(records)

        deltas = compute_all_deltas(df)
        # 2 models × 2 non-baseline conditions = 4 rows
        assert len(deltas) == 4
        assert "observed_delta" in deltas.columns
        assert "ci_low" in deltas.columns
        assert "ci_high" in deltas.columns

    def test_predictions_jsonl_roundtrip(self) -> None:
        """Ensure we can write and re-read predictions JSONL."""
        records = _make_synthetic_predictions()

        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for rec in records:
                f.write(json.dumps(rec) + "\n")
            tmp_path = Path(f.name)

        from src.analysis.compute_metrics import load_predictions

        df = load_predictions(tmp_path)
        assert len(df) == len(records)
        tmp_path.unlink()

    def test_n_per_condition_matches(self) -> None:
        """Sanity check: n per condition should be consistent."""
        import pandas as pd
        from src.analysis.compute_metrics import compute_accuracy_table

        records = _make_synthetic_predictions(n_questions=10)
        df = pd.DataFrame(records)
        acc_table = compute_accuracy_table(df)

        for _, row in acc_table.iterrows():
            assert row["n"] == 10, f"Expected 10 questions for {row['model_id']}/{row['condition_id']}"
