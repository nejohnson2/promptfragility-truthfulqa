"""Central configuration for the prompt fragility evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
RESULTS_DIR = PROJECT_ROOT / "results" / "runs"
CONDITIONS_PATH = PROMPTS_DIR / "conditions.yaml"
TEMPLATE_PATH = PROMPTS_DIR / "templates" / "truthfulqa_mc.jinja"
CACHE_PATH = DATA_DIR / "truthfulqa_cache.jsonl"

# ---------------------------------------------------------------------------
# Models to evaluate
# ---------------------------------------------------------------------------
MODEL_IDS: list[str] = [
    # ── Scale axis: Qwen family (3B → 7B → 72B) ──
    "Qwen/Qwen2.5-3B-Instruct",
    "Qwen/Qwen2.5-7B-Instruct",
    "Qwen/Qwen2.5-72B-Instruct",
    # ── Scale axis: Llama family (8B → 70B) ──
    "meta-llama/Llama-3.1-8B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct",
    # ── Cross-family at ~7-12B ──
    "mistralai/Mistral-7B-Instruct-v0.3",
    "google/gemma-3-12b-it",
    # ── Small-but-strong outlier ──
    "microsoft/Phi-4-mini-instruct",
]

# ---------------------------------------------------------------------------
# Decoding
# ---------------------------------------------------------------------------
@dataclass
class DecodingParams:
    temperature: float = 0.0
    do_sample: bool = False
    top_p: float = 1.0
    max_new_tokens: int = 10
    seed: int = 42

# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------
@dataclass
class EvalConfig:
    model_id: str = MODEL_IDS[0]
    conditions_path: Path = CONDITIONS_PATH
    template_path: Path = TEMPLATE_PATH
    dataset_path: Path = CACHE_PATH
    split: str = "dev"
    output_dir: Optional[Path] = None
    batch_size: int = 8
    quantize_4bit: bool = False
    decoding: DecodingParams = field(default_factory=DecodingParams)
    seed: int = 42

# ---------------------------------------------------------------------------
# Dataset split
# ---------------------------------------------------------------------------
DEV_FRACTION = 0.10
SPLIT_SEED = 42

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 123
