# Prompt Fragility in TruthfulQA MC

Measuring prompt-induced instability in TruthfulQA multiple-choice scoring for open-source LLMs.

## Quick Start

```bash
# 1. Install dependencies (Python 3.10+ with CUDA recommended)
pip install -r requirements.txt

# 2. Fetch and cache the dataset
python data/fetch_truthfulqa.py --out data/truthfulqa_cache.jsonl --seed 42

# 3. Run evaluation for a single model on the dev split
python -m src.eval.run_eval \
    --model_id TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
    --split dev \
    --output_dir results/runs

# 4. Aggregate and analyze results
python -m src.analysis.compute_metrics \
    --predictions results/runs/<RUN_ID>/predictions.jsonl \
    --output_dir results/runs/<RUN_ID>

python -m src.analysis.stats \
    --predictions results/runs/<RUN_ID>/predictions.jsonl \
    --output_dir results/runs/<RUN_ID>

python -m src.analysis.plots \
    --aggregated results/runs/<RUN_ID>/aggregated.csv \
    --ranking_metrics results/runs/<RUN_ID>/ranking_metrics.json \
    --model_summary results/runs/<RUN_ID>/model_summary.csv \
    --output_dir results/runs/<RUN_ID>/figures
```

## Full Pipeline

Run all models on a split with a single command:

```bash
bash scripts/run_matrix.sh dev      # dev split
bash scripts/run_matrix.sh final    # final split (for reported results)
```

## Project Structure

```
prompts/
  conditions.yaml           # 13 prompt conditions
  templates/truthfulqa_mc.jinja  # Jinja template

data/
  fetch_truthfulqa.py        # Dataset download and caching
  truthfulqa_cache.jsonl     # Cached dataset (generated)

src/
  config.py                  # Central configuration
  models/
    loader.py                # Model/tokenizer loading
    hf_model.py              # HF model runner
  prompts/
    render.py                # Prompt rendering from conditions
  eval/
    run_eval.py              # Evaluation runner CLI
    parse_answer.py          # Answer parser
    scoring.py               # Correctness scoring
    logging.py               # Run metadata and JSONL logging
  analysis/
    compute_metrics.py       # Accuracy tables, PSI, ranking metrics
    stats.py                 # Paired bootstrap CIs
    plots.py                 # Matplotlib figure generation

scripts/
  run_matrix.sh              # Full pipeline script

tests/
  test_prompts.py            # Prompt rendering and parser tests
  test_smoke.py              # Analysis pipeline smoke tests

results/runs/                # Output directory for runs
paper/outline.md             # Workshop paper outline
protocol.md                  # Locked evaluation protocol
```

## Adding a New Model

1. Add the HuggingFace model ID to `MODEL_IDS` in `src/config.py`.
2. Add it to the `MODELS` array in `scripts/run_matrix.sh`.
3. If the model needs special tokenizer handling, update `src/models/loader.py`.

## Adding a New Prompt Condition

1. Add a new entry to `prompts/conditions.yaml` with a unique `id`, `category`,
   `description`, and `template_vars`.
2. The template variables are: `system_prefix`, `label_style` (alpha/numeric),
   `option_separator` (newline/inline), `question_wrap` (plain/code_block),
   `distractor`, `suffix`, `style`.
3. Run `python -m pytest tests/test_prompts.py` to verify it renders correctly.

## Tests

```bash
python -m pytest tests/ -v
```

## Key Metrics

| Metric | Definition |
|---|---|
| PSI (Prompt Sensitivity Index) | baseline_acc − worst_acc |
| Kendall τ | Rank correlation vs baseline ranking |
| Rank flip rate | Fraction of model pairs that swap order across conditions |
| Top-1 instability | Conditions where top model differs from baseline top-1 |
