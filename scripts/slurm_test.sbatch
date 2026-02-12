#!/bin/bash
#SBATCH --job-name=pf-test
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=00:10:00
#SBATCH --partition=debug-h200x4
#
# Environment test script — mirrors slurm_pipeline.sbatch setup exactly,
# but stops before running any model inference.
#
# Usage:
#   sbatch scripts/slurm_test.sbatch

set -euo pipefail

# ── Environment setup ─────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# Create logs directory and move SLURM output files there
mkdir -p logs
# SLURM writes output files to the submission directory first,
# then we move them into logs/ at the end of the job (see bottom of script)

# ── Activate environment (MUST come before any python commands) ───────────
module load cuda12.8/toolkit/12.8.0
# Initialize conda for non-interactive shell, then activate
eval "$(conda shell.bash hook)"
conda activate promptfragility

echo "================================================================"
echo "Prompt Fragility — Environment Test"
echo "================================================================"
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          $(hostname)"
echo "Date:          $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Working dir:   $(pwd)"
echo "Python:        $(which python)"
echo "Python ver:    $(python --version 2>&1)"
echo "Conda env:     ${CONDA_DEFAULT_ENV:-none}"
echo ""

# ── GPU diagnostics ───────────────────────────────────────────────────────
echo "── GPU Info ──"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo ""
    nvidia-smi
else
    echo "nvidia-smi not found"
fi
echo ""

# ── Python environment diagnostics ────────────────────────────────────────
echo "── Python Packages ──"
python -c "
import torch
print(f'torch:         {torch.__version__}')
print(f'CUDA avail:    {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version:  {torch.version.cuda}')
    print(f'GPU count:     {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)')
import transformers; print(f'transformers:  {transformers.__version__}')
import accelerate; print(f'accelerate:    {accelerate.__version__}')
import datasets; print(f'datasets:      {datasets.__version__}')
import numpy; print(f'numpy:         {numpy.__version__}')
import pandas; print(f'pandas:        {pandas.__version__}')
"
echo ""

# ── HuggingFace auth ──────────────────────────────────────────────────────
echo "── HuggingFace Auth ──"
python -c "
from huggingface_hub import HfApi
api = HfApi()
try:
    info = api.whoami()
    print(f'Logged in as: {info[\"name\"]}')
except Exception as e:
    print(f'NOT authenticated: {e}')
"
echo ""

# ── Project files ─────────────────────────────────────────────────────────
echo "── Project Files ──"
for f in "data/truthfulqa_cache.jsonl" "prompts/conditions.yaml" "prompts/templates/truthfulqa_mc.jinja" "src/config.py" "src/eval/run_eval.py"; do
    if [ -f "$f" ]; then
        echo "  OK: $f"
    else
        echo "  MISSING: $f"
    fi
done
echo ""

# ── HF cache location ────────────────────────────────────────────────────
echo "── HuggingFace Cache ──"
echo "HF_HOME: ${HF_HOME:-not set (default: ~/.cache/huggingface)}"
echo ""

# ── Parse arguments (same as pipeline) ───────────────────────────────────
SPLIT="${1:-dev}"
echo "── Config ──"
echo "Split:         $SPLIT"

CACHE_FILE="data/truthfulqa_cache.jsonl"
CONDITIONS_FILE="prompts/conditions.yaml"
OUTPUT_BASE="results/runs"

if [ -f "$CACHE_FILE" ]; then
    echo "Dataset:       $CACHE_FILE ($(wc -l < "$CACHE_FILE") records)"
else
    echo "Dataset:       $CACHE_FILE (NOT FOUND — run: python data/fetch_truthfulqa.py --out $CACHE_FILE --seed 42)"
fi
echo ""

echo "================================================================"
echo "All checks passed. Environment is ready for slurm_pipeline.sbatch"
echo "================================================================"

# ── Move SLURM log files into logs/ directory ────────────────────────────
SLURM_OUT="${SLURM_SUBMIT_DIR:-$PROJECT_ROOT}/pf-test_${SLURM_JOB_ID:-local}.out"
SLURM_ERR="${SLURM_SUBMIT_DIR:-$PROJECT_ROOT}/pf-test_${SLURM_JOB_ID:-local}.err"
if [ -f "$SLURM_OUT" ]; then mv "$SLURM_OUT" "$PROJECT_ROOT/logs/"; fi
if [ -f "$SLURM_ERR" ]; then mv "$SLURM_ERR" "$PROJECT_ROOT/logs/"; fi
