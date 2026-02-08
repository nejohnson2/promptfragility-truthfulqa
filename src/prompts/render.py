"""Render TruthfulQA MC prompts from conditions and Jinja template."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml
from jinja2 import Environment, FileSystemLoader

from src.config import CONDITIONS_PATH, TEMPLATE_PATH


def load_conditions(path: Path = CONDITIONS_PATH) -> list[dict[str, Any]]:
    """Load all prompt conditions from YAML."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return data["conditions"]


def get_condition_by_id(
    condition_id: str, conditions: list[dict[str, Any]] | None = None
) -> dict[str, Any]:
    """Retrieve a single condition by its id."""
    if conditions is None:
        conditions = load_conditions()
    for c in conditions:
        if c["id"] == condition_id:
            return c
    raise ValueError(f"Unknown condition_id: {condition_id}")


def _make_label(index: int, label_style: str) -> str:
    if label_style == "numeric":
        return str(index + 1)
    return chr(ord("A") + index)


def _format_options(
    choices: list[dict[str, str]],
    label_style: str,
    option_separator: str,
) -> tuple[str, dict[str, str]]:
    """Format options block and return (block_text, label_map).

    label_map maps the *displayed* label back to the canonical (A-based) label
    so the answer parser knows what to expect.
    """
    lines: list[str] = []
    label_map: dict[str, str] = {}  # displayed -> canonical

    for i, choice in enumerate(choices):
        displayed = _make_label(i, label_style)
        canonical = chr(ord("A") + i)
        label_map[displayed] = canonical
        lines.append(f"{displayed}) {choice['text']}")

    if option_separator == "inline":
        block = "  ;  ".join(lines)
    else:  # "newline"
        block = "\n".join(lines)

    return block, label_map


def render_prompt(
    record: dict[str, Any],
    condition: dict[str, Any],
    template_path: Path = TEMPLATE_PATH,
) -> tuple[str, dict[str, str]]:
    """Render a complete prompt string for one question under one condition.

    Returns:
        (prompt_text, label_map)  where label_map maps displayed labels to
        canonical A/B/C/D labels used in the dataset.
    """
    tv = condition["template_vars"]

    options_block, label_map = _format_options(
        record["choices"],
        tv["label_style"],
        tv["option_separator"],
    )

    env = Environment(
        loader=FileSystemLoader(str(template_path.parent)),
        keep_trailing_newline=False,
        trim_blocks=True,
        lstrip_blocks=True,
    )
    template = env.get_template(template_path.name)

    prompt_text = template.render(
        system_prefix=tv["system_prefix"],
        question=record["question"],
        options_block=options_block,
        distractor=tv.get("distractor", ""),
        suffix=tv.get("suffix", "Answer:"),
        question_wrap=tv.get("question_wrap", "plain"),
    )

    return prompt_text.strip(), label_map
