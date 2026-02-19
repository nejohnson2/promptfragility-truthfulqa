"""Strict answer parser for MC evaluation.

Rules:
  1. If the output contains "Answer: X" (case-insensitive), parse X.
  2. If the output is a single token matching a valid label, accept it.
  3. Otherwise, look for standalone label tokens (word boundaries).
  4. Valid labels are derived from the label_map (supports any number of choices).
  5. If no valid label is found, mark as invalid.
"""

from __future__ import annotations

import re


def _build_label_pattern(valid_displayed: set[str]) -> str:
    """Build a regex character class matching all valid displayed labels."""
    # Separate alpha and numeric labels, build pattern from actual keys
    alphas = sorted(c for c in valid_displayed if c.isalpha())
    digits = sorted(c for c in valid_displayed if c.isdigit())

    parts: list[str] = []
    if alphas:
        lo, hi = alphas[0], alphas[-1]
        parts.append(f"{lo}-{hi}{lo.lower()}-{hi.lower()}")
    if digits:
        lo, hi = digits[0], digits[-1]
        parts.append(f"{lo}-{hi}")

    return f"[{''.join(parts)}]"


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

    # Build regex fragment matching any valid displayed label
    label_class = _build_label_pattern(valid_displayed)

    # Strategy 1: look for "Answer: X" pattern
    m = re.search(rf"[Aa]nswer\s*:\s*({label_class})", text)
    if m:
        token = m.group(1).upper()
        if token in valid_displayed:
            return label_map[token], False

    # Strategy 2: if the entire output (stripped) is a single valid label
    if text.upper() in valid_displayed:
        return label_map[text.upper()], False

    # Also check patterns like "A)" or "A." at the start
    m = re.match(rf"^({label_class})[).:\s]", text)
    if m:
        token = m.group(1).upper()
        if token in valid_displayed:
            return label_map[token], False

    # Strategy 3: look for standalone label tokens (word-boundary matching)
    # For alpha labels, require word boundaries to avoid matching letters inside words
    if has_alpha:
        pattern = rf"(?<![a-zA-Z])({label_class})(?![a-zA-Z])"
    else:
        pattern = rf"(?<!\d)({label_class})(?!\d)"

    for m in re.finditer(pattern, text):
        token = m.group(1).upper()
        if token in valid_displayed:
            return label_map[token], False

    return None, True


def parse_mc_answer_staged(
    output_text: str,
    label_map: dict[str, str],
) -> tuple[str | None, bool, str]:
    """Parse with stage tracking — identical logic to parse_mc_answer.

    Returns:
        (parsed_canonical_label, is_invalid, stage)
        stage is one of: "answer_pattern", "exact_token", "label_punctuation",
                         "word_boundary", "invalid"
    """
    text = output_text.strip()
    valid_displayed = set(label_map.keys())
    has_alpha = any(c.isalpha() for c in valid_displayed)
    label_class = _build_label_pattern(valid_displayed)

    # Stage 1: "Answer: X"
    m = re.search(rf"[Aa]nswer\s*:\s*({label_class})", text)
    if m:
        token = m.group(1).upper()
        if token in valid_displayed:
            return label_map[token], False, "answer_pattern"

    # Stage 2a: exact single token
    if text.upper() in valid_displayed:
        return label_map[text.upper()], False, "exact_token"

    # Stage 2b: label + punctuation at start
    m = re.match(rf"^({label_class})[).:\s]", text)
    if m:
        token = m.group(1).upper()
        if token in valid_displayed:
            return label_map[token], False, "label_punctuation"

    # Stage 3: word-boundary scan
    if has_alpha:
        pattern = rf"(?<![a-zA-Z])({label_class})(?![a-zA-Z])"
    else:
        pattern = rf"(?<!\d)({label_class})(?!\d)"

    for m in re.finditer(pattern, text):
        token = m.group(1).upper()
        if token in valid_displayed:
            return label_map[token], False, "word_boundary"

    return None, True, "invalid"
