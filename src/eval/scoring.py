"""Scoring utilities for MC evaluation."""

from __future__ import annotations


def score_prediction(parsed_label: str | None, correct_label: str) -> tuple[bool, bool]:
    """Score a single prediction.

    Returns:
        (is_correct, is_invalid)
    """
    if parsed_label is None:
        return False, True
    return parsed_label == correct_label, False
