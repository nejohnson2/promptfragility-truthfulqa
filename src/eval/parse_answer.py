"""Strict answer parser for MC evaluation.

Rules:
  1. If the output contains "Answer: X" (case-insensitive), parse X.
  2. If the output is a single token matching a valid label, accept it.
  3. Otherwise, look for standalone label tokens (word boundaries).
  4. Valid labels depend on label_style: A-D (alpha) or 1-4 (numeric).
  5. If no valid label is found, mark as invalid.
"""

from __future__ import annotations

import re


def parse_mc_answer(
    output_text: str,
    label_map: dict[str, str],
) -> tuple[str | None, bool]:
    """Parse a model's output text into a canonical answer label.

    Args:
        output_text: Raw model output string.
        label_map: Mapping from displayed labels (e.g. "1","2","3","4" or "A","B","C","D")
                   to canonical labels ("A","B","C","D").

    Returns:
        (parsed_canonical_label, is_invalid)
        If parsing fails, returns (None, True).
    """
    text = output_text.strip()
    valid_displayed = set(label_map.keys())

    # Determine if we're looking for alpha or numeric labels
    has_alpha = any(c.isalpha() for c in valid_displayed)
    has_numeric = any(c.isdigit() for c in valid_displayed)

    # Strategy 1: look for "Answer: X" pattern
    m = re.search(r"[Aa]nswer\s*:\s*([A-Da-d1-4])", text)
    if m:
        token = m.group(1).upper()
        if token in valid_displayed:
            return label_map[token], False

    # Strategy 2: if the entire output (stripped) is a single valid label
    if text.upper() in valid_displayed:
        return label_map[text.upper()], False

    # Also check patterns like "A)" or "A." at the start
    m = re.match(r"^([A-Da-d1-4])[).\s]", text)
    if m:
        token = m.group(1).upper()
        if token in valid_displayed:
            return label_map[token], False

    # Strategy 3: look for standalone label tokens (word-boundary matching)
    # For alpha labels, require word boundaries to avoid matching letters inside words
    if has_alpha:
        pattern = r"(?<![a-zA-Z])([A-Da-d])(?![a-zA-Z])"
    else:
        pattern = r"(?<!\d)([1-4])(?!\d)"

    for m in re.finditer(pattern, text):
        token = m.group(1).upper()
        if token in valid_displayed:
            return label_map[token], False

    return None, True
