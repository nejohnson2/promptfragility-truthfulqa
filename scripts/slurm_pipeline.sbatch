#!/bin/bash
#SBATCH --job-name=pf-truthfulqa
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=24:00:00
#SBATCH --partition=h200x4-long
#
# Prompt Fragility — Full TruthfulQA MC Evaluation Pipeline (SLURM)
#
# Usage:
#   sbatch scripts/slurm_pipeline.sbatch              # dev split, no quantization
#   sbatch scripts/slurm_pipeline.sbatch dev           # dev split explicitly
#   sbatch scripts/slurm_pipeline.sbatch final         # final split
#   sbatch scripts/slurm_pipeline.sbatch dev --quantize # dev split + 4-bit quant
#
# For 70B+ models, request more GPUs:
#   sbatch --gres=gpu:h200:2 --mem=256G scripts/slurm_pipeline.sbatch final
#
# Logs are initially written to <submit_dir>/pf-truthfulqa_<JOBID>.out/.err
# and moved to logs/ at the end of the job.

set -euo pipefail

# ── Environment setup ─────────────────────────────────────────────────────
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
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
echo "Prompt Fragility — TruthfulQA MC Evaluation (SLURM)"
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

# ── Parse arguments ───────────────────────────────────────────────────────
SPLIT="${1:-dev}"
QUANTIZE_FLAG=""

for arg in "$@"; do
    case "$arg" in
        dev|final) SPLIT="$arg" ;;
        --quantize) QUANTIZE_FLAG="--quantize_4bit" ;;
    esac
done

TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_TAG="${SPLIT}_${TIMESTAMP}_job${SLURM_JOB_ID:-local}"

echo "── Run Config ──"
echo "Split:         $SPLIT"
echo "Run tag:       $RUN_TAG"
echo "Quantize:      ${QUANTIZE_FLAG:-none}"
echo ""

# ── HuggingFace token ────────────────────────────────────────────────────
# Load the token from the default home directory location BEFORE HF_HOME
# is potentially redirected to scratch. This ensures gated models (e.g.
# Llama 3.1) can authenticate even when HF_HOME points elsewhere.
HF_TOKEN_FILE="${HOME}/.cache/huggingface/token"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HF_TOKEN_FILE" ]; then
    export HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
    echo "HF_TOKEN loaded from $HF_TOKEN_FILE"
else
    echo "WARNING: No HF token found at $HF_TOKEN_FILE — gated models (Llama 3.1) will fail"
fi

# ── HuggingFace cache — explicit scratch path ────────────────────────────
# Must match the path used by slurm_download.sbatch so pre-downloaded
# models are found. Using scratch avoids NFS bottlenecks during inference.
export HF_HOME="/lustre/nvwulf/scratch/${USER}/hf_cache"
mkdir -p "$HF_HOME"
echo "HF_HOME:       $HF_HOME"

# Verify scratch is writable
if touch "$HF_HOME/.write_test" 2>/dev/null; then
    rm -f "$HF_HOME/.write_test"
    echo "Scratch:       writable"
else
    echo "ERROR: Cannot write to $HF_HOME — check scratch filesystem"
    exit 1
fi
echo ""

# ── Models ────────────────────────────────────────────────────────────────
MODELS=(
    # Scale axis: Qwen family
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-72B-Instruct"
    # Scale axis: Llama family
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.1-70B-Instruct"
    # Cross-family
    "mistralai/Mistral-7B-Instruct-v0.3"
    "google/gemma-3-12b-it"
    # Small-but-strong
    "microsoft/Phi-4-mini-instruct"
)

CACHE_FILE="data/truthfulqa_cache.jsonl"
CONDITIONS_FILE="prompts/conditions.yaml"
OUTPUT_BASE="results/runs"

echo "Models:        ${#MODELS[@]}"
echo ""

# ── Step 1: Fetch data ────────────────────────────────────────────────────
echo "================================================================"
echo "[Step 1/6] Fetching TruthfulQA dataset"
echo "================================================================"
if [ ! -f "$CACHE_FILE" ]; then
    python data/fetch_truthfulqa.py --out "$CACHE_FILE" --seed 42
else
    echo "Dataset cache exists: $CACHE_FILE ($(wc -l < "$CACHE_FILE") records)"
fi
echo ""

# ── Step 2: Run evaluation for each model ─────────────────────────────────
echo "================================================================"
echo "[Step 2/6] Running evaluation matrix"
echo "================================================================"

FAILED_MODELS=()
MODEL_IDX=0
TOTAL_MODELS=${#MODELS[@]}

for MODEL_ID in "${MODELS[@]}"; do
    MODEL_IDX=$((MODEL_IDX + 1))
    MODEL_SHORT=$(echo "$MODEL_ID" | sed 's|.*/||')

    echo ""
    echo "────────────────────────────────────────────────────────────"
    echo "[$MODEL_IDX/$TOTAL_MODELS] Model: $MODEL_SHORT"
    echo "  Full ID: $MODEL_ID"
    echo "  Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "────────────────────────────────────────────────────────────"

    # Log GPU memory before loading model
    if command -v nvidia-smi &>/dev/null; then
        echo "  GPU memory before load:"
        nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | sed 's/^/    /'
    fi

    if python -m src.eval.run_eval \
        --model_id "$MODEL_ID" \
        --dataset "$CACHE_FILE" \
        --conditions "$CONDITIONS_FILE" \
        --split "$SPLIT" \
        --output_dir "$OUTPUT_BASE" \
        $QUANTIZE_FLAG 2>&1; then
        echo "  Completed: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    else
        echo "  FAILED: $MODEL_ID (exit code $?)"
        FAILED_MODELS+=("$MODEL_ID")
        echo "  Continuing with next model..."
    fi

    # Log GPU memory after run
    if command -v nvidia-smi &>/dev/null; then
        echo "  GPU memory after run:"
        nvidia-smi --query-gpu=memory.used,memory.free --format=csv,noheader | sed 's/^/    /'
    fi
    echo ""
done

# ── Report failures ───────────────────────────────────────────────────────
if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo ""
    echo "WARNING: ${#FAILED_MODELS[@]} model(s) failed:"
    for m in "${FAILED_MODELS[@]}"; do
        echo "  - $m"
    done
    echo ""
    echo "Check the SLURM error log for details: pf-truthfulqa_${SLURM_JOB_ID:-local}.err"
    echo ""
fi

# ── Step 3: Merge predictions ─────────────────────────────────────────────
echo "================================================================"
echo "[Step 3/6] Merging predictions"
echo "================================================================"

MERGED_DIR="$OUTPUT_BASE/merged_${RUN_TAG}"
mkdir -p "$MERGED_DIR/figures"

> "$MERGED_DIR/predictions.jsonl"
MERGE_COUNT=0
for RUN_DIR in "$OUTPUT_BASE"/20*; do
    [ -d "$RUN_DIR" ] || continue
    [ -f "$RUN_DIR/predictions.jsonl" ] || continue
    cat "$RUN_DIR/predictions.jsonl" >> "$MERGED_DIR/predictions.jsonl"
    MERGE_COUNT=$((MERGE_COUNT + 1))
done

PRED_COUNT=$(wc -l < "$MERGED_DIR/predictions.jsonl")
echo "Merged $MERGE_COUNT run directories → $PRED_COUNT prediction records"
echo ""

if [ "$PRED_COUNT" -eq 0 ]; then
    echo "ERROR: No predictions found to analyze. Exiting."
    exit 1
fi

# ── Step 4: Compute metrics ──────────────────────────────────────────────
echo "================================================================"
echo "[Step 4/6] Computing metrics"
echo "================================================================"
python -m src.analysis.compute_metrics \
    --predictions "$MERGED_DIR/predictions.jsonl" \
    --output_dir "$MERGED_DIR"
echo ""

# ── Step 5: Bootstrap statistics ─────────────────────────────────────────
echo "================================================================"
echo "[Step 5/6] Computing bootstrap confidence intervals"
echo "================================================================"
python -m src.analysis.stats \
    --predictions "$MERGED_DIR/predictions.jsonl" \
    --output_dir "$MERGED_DIR"
echo ""

# ── Step 6: Generate plots ───────────────────────────────────────────────
echo "================================================================"
echo "[Step 6/6] Generating plots"
echo "================================================================"
python -m src.analysis.plots \
    --aggregated "$MERGED_DIR/aggregated.csv" \
    --ranking_metrics "$MERGED_DIR/ranking_metrics.json" \
    --model_summary "$MERGED_DIR/model_summary.csv" \
    --output_dir "$MERGED_DIR/figures"
echo ""

# ── Summary ──────────────────────────────────────────────────────────────
echo "================================================================"
echo "Pipeline complete!"
echo "================================================================"
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Finished:      $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Results dir:   $MERGED_DIR"
echo ""
echo "Output files:"
ls -lh "$MERGED_DIR/" 2>/dev/null
echo ""
echo "Figures:"
ls -lh "$MERGED_DIR/figures/" 2>/dev/null || echo "  (none)"
echo ""

if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo "WARNING: ${#FAILED_MODELS[@]} model(s) failed. Review errors above."
    EXIT_CODE=1
else
    echo "All ${TOTAL_MODELS} models completed successfully."
    EXIT_CODE=0
fi

# ── Move SLURM log files into logs/ directory ────────────────────────────
# SLURM writes output files to the submission directory; move them for cleanliness
SLURM_OUT="${SLURM_SUBMIT_DIR:-$PROJECT_ROOT}/pf-truthfulqa_${SLURM_JOB_ID:-local}.out"
SLURM_ERR="${SLURM_SUBMIT_DIR:-$PROJECT_ROOT}/pf-truthfulqa_${SLURM_JOB_ID:-local}.err"
if [ -f "$SLURM_OUT" ]; then mv "$SLURM_OUT" "$PROJECT_ROOT/logs/"; fi
if [ -f "$SLURM_ERR" ]; then mv "$SLURM_ERR" "$PROJECT_ROOT/logs/"; fi

exit $EXIT_CODE
