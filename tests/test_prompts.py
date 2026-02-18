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


EXAMPLE_RECORD_7_CHOICES = {
    "question_id": 99,
    "question": "Which of the following is true?",
    "choices": [
        {"label": "A", "text": "Option one"},
        {"label": "B", "text": "Option two"},
        {"label": "C", "text": "Option three"},
        {"label": "D", "text": "Option four"},
        {"label": "E", "text": "Option five"},
        {"label": "F", "text": "Option six"},
        {"label": "G", "text": "Option seven"},
    ],
    "correct_label": "E",
    "category": "Test",
    "split": "dev",
}


class TestPromptRenderingExtended:
    """Verify prompt rendering works with >4 choices."""

    def test_render_7_choices_alpha(self) -> None:
        conditions = load_conditions(CONDITIONS_PATH)
        baseline = [c for c in conditions if c["id"] == "baseline"][0]
        prompt_text, label_map = render_prompt(
            EXAMPLE_RECORD_7_CHOICES, baseline, TEMPLATE_PATH
        )
        assert "G)" in prompt_text
        assert len(label_map) == 7
        assert label_map["E"] == "E"
        assert label_map["G"] == "G"

    def test_render_7_choices_numeric(self) -> None:
        conditions = load_conditions(CONDITIONS_PATH)
        numeric = [c for c in conditions if c["id"] == "numeric_labels"][0]
        prompt_text, label_map = render_prompt(
            EXAMPLE_RECORD_7_CHOICES, numeric, TEMPLATE_PATH
        )
        assert "7)" in prompt_text
        assert len(label_map) == 7
        assert label_map["5"] == "E"
        assert label_map["7"] == "G"


class TestAnswerParser:
    """Verify answer parsing works correctly."""

    ALPHA_MAP = {"A": "A", "B": "B", "C": "C", "D": "D"}
    NUMERIC_MAP = {"1": "A", "2": "B", "3": "C", "4": "D"}
    # Extended maps for questions with >4 choices (TruthfulQA has up to 13)
    ALPHA_MAP_7 = {
        "A": "A", "B": "B", "C": "C", "D": "D",
        "E": "E", "F": "F", "G": "G",
    }
    NUMERIC_MAP_7 = {
        "1": "A", "2": "B", "3": "C", "4": "D",
        "5": "E", "6": "F", "7": "G",
    }

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

    # --- Extended label tests (>4 choices) ---

    def test_parse_extended_alpha_single(self) -> None:
        label, invalid = parse_mc_answer("E", self.ALPHA_MAP_7)
        assert label == "E"
        assert not invalid

    def test_parse_extended_alpha_answer_colon(self) -> None:
        label, invalid = parse_mc_answer("Answer: G", self.ALPHA_MAP_7)
        assert label == "G"
        assert not invalid

    def test_parse_extended_alpha_with_paren(self) -> None:
        label, invalid = parse_mc_answer("F) is the correct one", self.ALPHA_MAP_7)
        assert label == "F"
        assert not invalid

    def test_parse_extended_numeric_single(self) -> None:
        label, invalid = parse_mc_answer("5", self.NUMERIC_MAP_7)
        assert label == "E"
        assert not invalid

    def test_parse_extended_numeric_answer_colon(self) -> None:
        label, invalid = parse_mc_answer("Answer: 7", self.NUMERIC_MAP_7)
        assert label == "G"
        assert not invalid

    def test_parse_extended_numeric_with_paren(self) -> None:
        label, invalid = parse_mc_answer("6) seems right", self.NUMERIC_MAP_7)
        assert label == "F"
        assert not invalid

    def test_parse_extended_alpha_invalid_beyond_range(self) -> None:
        # H is not in the 7-option map
        label, invalid = parse_mc_answer("H", self.ALPHA_MAP_7)
        assert label is None
        assert invalid

    def test_parse_extended_numeric_invalid_beyond_range(self) -> None:
        # 8 is not in the 7-option map
        label, invalid = parse_mc_answer("8", self.NUMERIC_MAP_7)
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
