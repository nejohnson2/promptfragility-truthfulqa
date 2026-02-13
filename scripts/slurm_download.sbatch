#!/bin/bash
#SBATCH --job-name=pf-download
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --partition=h200x4-long
#
# Pre-download all model weights to scratch before running evaluation.
# No GPU needed — downloads only.
#
# Usage:
#   sbatch scripts/slurm_download.sbatch
#
# Models are cached to /lustre/nvwulf/scratch/$USER/hf_cache/
# The pipeline script reads from this same location.

set -euo pipefail

# ── Environment setup ─────────────────────────────────────────────────────
PROJECT_ROOT="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$PROJECT_ROOT"
mkdir -p logs

# ── Activate environment ─────────────────────────────────────────────────
module load cuda12.8/toolkit/12.8.0
eval "$(conda shell.bash hook)"
conda activate promptfragility

echo "================================================================"
echo "Prompt Fragility — Model Pre-Download"
echo "================================================================"
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          $(hostname)"
echo "Date:          $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""

# ── HuggingFace token ────────────────────────────────────────────────────
HF_TOKEN_FILE="${HOME}/.cache/huggingface/token"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HF_TOKEN_FILE" ]; then
    export HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
    echo "HF_TOKEN loaded from $HF_TOKEN_FILE"
else
    echo "WARNING: No HF token found at $HF_TOKEN_FILE — gated models will fail"
fi

# ── HuggingFace cache — explicit scratch path ────────────────────────────
export HF_HOME="/lustre/nvwulf/scratch/${USER}/hf_cache"
mkdir -p "$HF_HOME"
echo "HF_HOME:       $HF_HOME"

# Verify scratch is writable
if touch "$HF_HOME/.write_test" 2>/dev/null; then
    rm -f "$HF_HOME/.write_test"
    echo "Scratch:       writable"
else
    echo "ERROR: Cannot write to $HF_HOME"
    exit 1
fi

# Show available disk space
df -h "$HF_HOME" 2>/dev/null | tail -1
echo ""

# ── Models to download ───────────────────────────────────────────────────
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

TOTAL=${#MODELS[@]}
FAILED=()
IDX=0

echo "Downloading $TOTAL models to $HF_HOME"
echo ""

for MODEL_ID in "${MODELS[@]}"; do
    IDX=$((IDX + 1))
    MODEL_SHORT=$(echo "$MODEL_ID" | sed 's|.*/||')

    echo "────────────────────────────────────────────────────────────"
    echo "[$IDX/$TOTAL] $MODEL_SHORT"
    echo "  Full ID: $MODEL_ID"
    echo "  Started: $(date -u '+%H:%M:%S UTC')"

    if python -c "
import os
os.environ['HF_HOME'] = '$HF_HOME'
from huggingface_hub import snapshot_download
path = snapshot_download(
    '$MODEL_ID',
    cache_dir='$HF_HOME',
    resume_download=True,
)
print(f'  Downloaded to: {path}')
" 2>&1; then
        echo "  Completed: $(date -u '+%H:%M:%S UTC')"
    else
        echo "  FAILED: $MODEL_ID (exit code $?)"
        FAILED+=("$MODEL_ID")
    fi
    echo ""
done

# ── Also download tokenizers (ensures fast tokenizer files are cached) ───
echo "────────────────────────────────────────────────────────────"
echo "Verifying tokenizers load correctly..."
for MODEL_ID in "${MODELS[@]}"; do
    MODEL_SHORT=$(echo "$MODEL_ID" | sed 's|.*/||')
    if python -c "
import os
os.environ['HF_HOME'] = '$HF_HOME'
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('$MODEL_ID', cache_dir='$HF_HOME')
print(f'  {\"$MODEL_SHORT\"}: vocab_size={tok.vocab_size}, ok')
" 2>&1; then
        :
    else
        echo "  WARNING: Tokenizer failed for $MODEL_SHORT"
    fi
done
echo ""

# ── Summary ──────────────────────────────────────────────────────────────
echo "================================================================"
echo "Download complete!"
echo "================================================================"
echo "Cache dir:     $HF_HOME"
echo "Cache size:    $(du -sh "$HF_HOME" 2>/dev/null | cut -f1)"
echo ""

if [ ${#FAILED[@]} -gt 0 ]; then
    echo "WARNING: ${#FAILED[@]} model(s) failed to download:"
    for m in "${FAILED[@]}"; do
        echo "  - $m"
    done
    EXIT_CODE=1
else
    echo "All $TOTAL models downloaded successfully."
    echo ""
    echo "Next step: sbatch scripts/slurm_pipeline.sbatch dev"
    EXIT_CODE=0
fi

# ── Move SLURM log files into logs/ directory ────────────────────────────
SLURM_OUT="${SLURM_SUBMIT_DIR:-$PROJECT_ROOT}/pf-download_${SLURM_JOB_ID:-local}.out"
SLURM_ERR="${SLURM_SUBMIT_DIR:-$PROJECT_ROOT}/pf-download_${SLURM_JOB_ID:-local}.err"
if [ -f "$SLURM_OUT" ]; then mv "$SLURM_OUT" "$PROJECT_ROOT/logs/"; fi
if [ -f "$SLURM_ERR" ]; then mv "$SLURM_ERR" "$PROJECT_ROOT/logs/"; fi

exit $EXIT_CODE
