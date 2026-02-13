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

## Full Pipeline (Local)

Run all models on a split with a single command:

```bash
bash scripts/run_matrix.sh dev      # dev split
bash scripts/run_matrix.sh final    # final split (for reported results)
```

## Running on a SLURM Cluster

### Available Cluster Resources

| Partition | GPUs | GPUs/Node | Time Limit | Notes |
|---|---|---|---|---|
| `debug-b40x4` | NVIDIA RTX PRO 6000 Blackwell | 4 | 1 hour | Quick tests |
| `b40x4` | NVIDIA RTX PRO 6000 Blackwell | 4 | 8 hours | Standard runs |
| `b40x4-long` | NVIDIA RTX PRO 6000 Blackwell | 4 | 48 hours | Long runs |
| `debug-h200x4` | H200 | 4 | 1 hour | Quick tests |
| `h200x4` | H200 | 4 | 8 hours | Standard runs |
| `h200x4-long` | H200 | 4 | 48 hours | Long runs (default) |
| `h200x8` | H200 | 8 | 8 hours | Multi-GPU |
| `h200x8-long` | H200 | 8 | 48 hours | Multi-GPU long |

The pipeline defaults to `h200x4-long` with 1x H200 (80GB HBM3e), which can handle all models up to 72B in fp16.

### Prerequisites

1. **Python environment**: Set up a conda env on the cluster:
   ```bash
   module load cuda12.8/toolkit/12.8.0
   conda create -n promptfragility python=3.11 -y
   conda activate promptfragility
   pip install -r requirements.txt
   ```

2. **HuggingFace authentication** (required for Llama 3.1 models):
   ```bash
   pip install huggingface_hub
   python -c "from huggingface_hub import login; login()"
   ```
   Then accept the Llama 3.1 license at:
   - https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct
   - https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct

### Testing the Environment

Before running the full pipeline, verify your cluster environment:

```bash
sbatch scripts/slurm_test.sbatch
```

This runs on the `debug-h200x4` partition (10 min) and checks GPU access, Python packages, HuggingFace auth, and project files without running any model inference.

### Submitting the Job

```bash
# Basic submission (dev split, 1x H200, 24h on h200x4-long)
sbatch scripts/slurm_pipeline.sbatch

# Dev split explicitly
sbatch scripts/slurm_pipeline.sbatch dev

# Final split
sbatch scripts/slurm_pipeline.sbatch final

# With 4-bit quantization
sbatch scripts/slurm_pipeline.sbatch dev --quantize

# For 70B+ models, request more GPUs if needed
sbatch --gres=gpu:h200:2 --mem=256G scripts/slurm_pipeline.sbatch final

# Quick debug run (1h limit, debug partitions have max 2 CPUs)
sbatch --partition=debug-h200x4 --time=01:00:00 --cpus-per-task=2 --mem=22G scripts/slurm_pipeline.sbatch dev
```

### Customizing the SLURM Script

Edit `scripts/slurm_pipeline.sbatch` to change defaults:

| Setting | Default | What to change |
|---|---|---|
| `--partition` | `h200x4-long` | Use `debug-h200x4` for quick tests, `h200x8` for multi-GPU |
| `--gres` | `gpu:h200:1` | `gpu:h200:2` for 70B+ models if OOM |
| `--mem` | `128G` | `256G` if using multiple GPUs |
| `--time` | `24:00:00` | Max is 48h on `-long` partitions, 8h on standard |
| `--cpus-per-task` | `8` | `debug-*` partitions allow max 2 CPUs |

**Environment activation** (already configured in the script):
```bash
module load cuda12.8/toolkit/12.8.0
conda activate promptfragility
```

### Monitoring Jobs

```bash
# Check job status
squeue -u $USER

# Watch SLURM output in real-time (logs are moved to logs/ after job completes)
tail -f pf-truthfulqa_<JOBID>.out

# Check for errors
tail -f pf-truthfulqa_<JOBID>.err

# Cancel a job
scancel <JOBID>
```

### Output Structure

After completion, results are in `results/runs/merged_<RUN_TAG>/`:

```
results/runs/merged_dev_20250211_143022_job12345/
├── predictions.jsonl      # All model × condition predictions
├── aggregated.csv         # Accuracy by model × condition
├── model_summary.csv      # Per-model PSI, best/worst conditions
├── ranking_metrics.json   # Kendall tau, rank flips, top-1 instability
├── bootstrap_cis.json     # 95% confidence intervals
├── figures/
│   ├── accuracy_heatmap.png
│   ├── psi_bar.png
│   ├── kendall_tau.png
│   └── baseline_vs_worst.png
└── <run_id>/              # Per-model subdirectories
    ├── predictions.jsonl
    ├── metadata.json
    └── run.log            # Detailed per-model log with timing and GPU info
```

### Logging and Debugging

The pipeline provides multiple levels of logging:

- **SLURM logs** (`pf-truthfulqa_<JOBID>.out/.err`): Job-level output including GPU diagnostics, model progress, and any SLURM errors. Moved to `logs/` after job completes.
- **Per-model run logs** (`results/runs/<RUN_ID>/run.log`): Detailed logs with per-condition accuracy, timing, GPU memory usage, and any inference errors.
- **GPU diagnostics**: GPU name, memory, and CUDA version are logged at job start and after each model/condition.

If a model fails (e.g., OOM), the pipeline continues with remaining models and reports failures at the end.

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
  run_matrix.sh              # Full pipeline script (local)
  slurm_pipeline.sbatch      # SLURM batch script (cluster)
  slurm_test.sbatch          # SLURM environment test (cluster)

tests/
  test_prompts.py            # Prompt rendering and parser tests
  test_smoke.py              # Analysis pipeline smoke tests

results/runs/                # Output directory for runs
paper/outline.md             # Workshop paper outline
protocol.md                  # Locked evaluation protocol
```

## Adding a New Model

1. Add the HuggingFace model ID to `MODEL_IDS` in `src/config.py`.
2. Add it to the `MODELS` array in both `scripts/run_matrix.sh` and `scripts/slurm_pipeline.sbatch`.
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
| PSI (Prompt Sensitivity Index) | baseline_acc - worst_acc |
| Kendall tau | Rank correlation vs baseline ranking |
| Rank flip rate | Fraction of model pairs that swap order across conditions |
| Top-1 instability | Conditions where top model differs from baseline top-1 |
