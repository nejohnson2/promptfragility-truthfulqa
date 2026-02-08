"""Smoke tests for prompt rendering and answer parsing."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.prompts.render import load_conditions, render_prompt
from src.eval.parse_answer import parse_mc_answer
from src.eval.scoring import score_prediction
from src.config import CONDITIONS_PATH, TEMPLATE_PATH


# A canonical example record matching our JSONL schema
EXAMPLE_RECORD = {
    "question_id": 0,
    "question": "What is the capital of France?",
    "choices": [
        {"label": "A", "text": "Berlin"},
        {"label": "B", "text": "Paris"},
        {"label": "C", "text": "London"},
        {"label": "D", "text": "Madrid"},
    ],
    "correct_label": "B",
    "category": "Geography",
    "split": "dev",
}


class TestPromptRendering:
    """Verify all conditions render without errors."""

    def test_all_conditions_render(self) -> None:
        conditions = load_conditions(CONDITIONS_PATH)
        assert len(conditions) >= 13, f"Expected ≥13 conditions, got {len(conditions)}"

        for cond in conditions:
            prompt_text, label_map = render_prompt(
                EXAMPLE_RECORD, cond, TEMPLATE_PATH
            )
            assert isinstance(prompt_text, str)
            assert len(prompt_text) > 20, f"Prompt too short for condition {cond['id']}"
            assert isinstance(label_map, dict)
            assert len(label_map) == 4

    def test_baseline_contains_question(self) -> None:
        conditions = load_conditions(CONDITIONS_PATH)
        baseline = [c for c in conditions if c["id"] == "baseline"][0]
        prompt_text, _ = render_prompt(EXAMPLE_RECORD, baseline, TEMPLATE_PATH)
        assert "What is the capital of France?" in prompt_text

    def test_numeric_labels_condition(self) -> None:
        conditions = load_conditions(CONDITIONS_PATH)
        numeric = [c for c in conditions if c["id"] == "numeric_labels"][0]
        prompt_text, label_map = render_prompt(EXAMPLE_RECORD, numeric, TEMPLATE_PATH)
        assert "1)" in prompt_text
        assert "2)" in prompt_text
        assert label_map["1"] == "A"
        assert label_map["2"] == "B"

    def test_code_block_condition(self) -> None:
        conditions = load_conditions(CONDITIONS_PATH)
        code = [c for c in conditions if c["id"] == "code_block"][0]
        prompt_text, _ = render_prompt(EXAMPLE_RECORD, code, TEMPLATE_PATH)
        assert "```" in prompt_text

    def test_distractor_sentence_present(self) -> None:
        conditions = load_conditions(CONDITIONS_PATH)
        dist = [c for c in conditions if c["id"] == "distractor_sentence"][0]
        prompt_text, _ = render_prompt(EXAMPLE_RECORD, dist, TEMPLATE_PATH)
        assert "Helsinki" in prompt_text


class TestAnswerParser:
    """Verify answer parsing works correctly."""

    ALPHA_MAP = {"A": "A", "B": "B", "C": "C", "D": "D"}
    NUMERIC_MAP = {"1": "A", "2": "B", "3": "C", "4": "D"}

    def test_parse_single_letter(self) -> None:
        label, invalid = parse_mc_answer("B", self.ALPHA_MAP)
        assert label == "B"
        assert not invalid

    def test_parse_answer_colon(self) -> None:
        label, invalid = parse_mc_answer("The answer is: Answer: C", self.ALPHA_MAP)
        assert label == "C"
        assert not invalid

    def test_parse_numeric(self) -> None:
        label, invalid = parse_mc_answer("2", self.NUMERIC_MAP)
        assert label == "B"
        assert not invalid

    def test_parse_invalid(self) -> None:
        label, invalid = parse_mc_answer("I don't know", self.ALPHA_MAP)
        assert label is None
        assert invalid

    def test_parse_first_valid(self) -> None:
        label, invalid = parse_mc_answer("A and B are both good but A", self.ALPHA_MAP)
        assert label == "A"
        assert not invalid

    def test_parse_with_explanation(self) -> None:
        label, invalid = parse_mc_answer(
            "The answer is D because Madrid is the capital of Spain.",
            self.ALPHA_MAP,
        )
        assert label == "D"
        assert not invalid

    def test_parse_letter_in_word_ignored(self) -> None:
        # Letters embedded in words like "don't" should not match
        label, invalid = parse_mc_answer("I don't know what it could be", self.ALPHA_MAP)
        assert label is None
        assert invalid


class TestScoring:
    def test_correct(self) -> None:
        is_correct, is_invalid = score_prediction("B", "B")
        assert is_correct
        assert not is_invalid

    def test_incorrect(self) -> None:
        is_correct, is_invalid = score_prediction("A", "B")
        assert not is_correct
        assert not is_invalid

    def test_invalid(self) -> None:
        is_correct, is_invalid = score_prediction(None, "B")
        assert not is_correct
        assert is_invalid


class TestSanityChecks:
    """Ensure baseline condition exists and conditions have required fields."""

    def test_baseline_present(self) -> None:
        conditions = load_conditions(CONDITIONS_PATH)
        ids = [c["id"] for c in conditions]
        assert "baseline" in ids

    def test_all_conditions_have_required_fields(self) -> None:
        conditions = load_conditions(CONDITIONS_PATH)
        for cond in conditions:
            assert "id" in cond
            assert "category" in cond
            assert "description" in cond
            assert "template_vars" in cond
            tv = cond["template_vars"]
            assert "system_prefix" in tv
            assert "label_style" in tv
            assert "option_separator" in tv
