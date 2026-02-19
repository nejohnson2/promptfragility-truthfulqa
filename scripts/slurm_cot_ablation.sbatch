#!/bin/bash
#SBATCH --job-name=pf-cot-ablation
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --gres=gpu:h200:1
#SBATCH --time=8:00:00
#SBATCH --partition=h200x4
#
# Token Budget Sensitivity Ablation for cot_constrained condition.
#
# Runs ONLY the cot_constrained condition with max_new_tokens ∈ {10, 20, 50}
# across all 8 models, to test whether the Llama-3.1-8B catastrophic drop
# is primarily a budget artifact or a genuine sensitivity to CoT elicitation.
#
# Usage:
#   sbatch scripts/slurm_cot_ablation.sbatch              # final split
#   sbatch scripts/slurm_cot_ablation.sbatch dev           # dev split
#
# For 70B+ models, request more GPUs:
#   sbatch --gres=gpu:h200:2 --mem=256G scripts/slurm_cot_ablation.sbatch final

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
echo "Prompt Fragility — CoT Token Budget Ablation (SLURM)"
echo "================================================================"
echo "Job ID:        ${SLURM_JOB_ID:-local}"
echo "Node:          $(hostname)"
echo "Date:          $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Working dir:   $(pwd)"
echo "Python:        $(which python)"
echo ""

# ── GPU diagnostics ───────────────────────────────────────────────────────
echo "── GPU Info ──"
if command -v nvidia-smi &>/dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
fi
echo ""

# ── Parse arguments ───────────────────────────────────────────────────────
SPLIT="${1:-final}"
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"

echo "── Run Config ──"
echo "Split:         $SPLIT"
echo "Ablation:      cot_constrained with max_new_tokens={10, 20, 50}"
echo ""

# ── HuggingFace token ────────────────────────────────────────────────────
HF_TOKEN_FILE="${HOME}/.cache/huggingface/token"
if [ -z "${HF_TOKEN:-}" ] && [ -f "$HF_TOKEN_FILE" ]; then
    export HF_TOKEN="$(cat "$HF_TOKEN_FILE")"
    echo "HF_TOKEN loaded from $HF_TOKEN_FILE"
fi

export HF_HOME="/lustre/nvwulf/scratch/${USER}/hf_cache"
mkdir -p "$HF_HOME"
echo "HF_HOME:       $HF_HOME"
echo ""

# ── Models ────────────────────────────────────────────────────────────────
MODELS=(
    "Qwen/Qwen2.5-3B-Instruct"
    "Qwen/Qwen2.5-7B-Instruct"
    "Qwen/Qwen2.5-72B-Instruct"
    "meta-llama/Llama-3.1-8B-Instruct"
    "meta-llama/Llama-3.1-70B-Instruct"
    "mistralai/Mistral-7B-Instruct-v0.3"
    "google/gemma-3-12b-it"
    "microsoft/Phi-4-mini-instruct"
)

CACHE_FILE="data/truthfulqa_cache.jsonl"
CONDITIONS_FILE="prompts/conditions.yaml"
OUTPUT_BASE="results/runs"

MAX_TOKENS_VALUES=(10 20 50)

echo "Models:        ${#MODELS[@]}"
echo "Token budgets: ${MAX_TOKENS_VALUES[*]}"
echo ""

# ── Fetch data if needed ─────────────────────────────────────────────────
if [ ! -f "$CACHE_FILE" ]; then
    python data/fetch_truthfulqa.py --out "$CACHE_FILE" --seed 42
fi

# ── Run ablation ─────────────────────────────────────────────────────────
FAILED=()

for MAX_TOK in "${MAX_TOKENS_VALUES[@]}"; do
    echo "================================================================"
    echo "Token budget: max_new_tokens=$MAX_TOK"
    echo "================================================================"

    # Track dirs created in this token-budget round
    EXISTING_DIRS_FILE="$(mktemp)"
    find "$OUTPUT_BASE" -maxdepth 1 -type d -name '20*' 2>/dev/null | sort > "$EXISTING_DIRS_FILE" || true

    MODEL_IDX=0
    for MODEL_ID in "${MODELS[@]}"; do
        MODEL_IDX=$((MODEL_IDX + 1))
        MODEL_SHORT=$(echo "$MODEL_ID" | sed 's|.*/||')

        echo ""
        echo "── [$MODEL_IDX/${#MODELS[@]}] $MODEL_SHORT (max_tokens=$MAX_TOK) ──"
        echo "  Started: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"

        if python -m src.eval.run_eval \
            --model_id "$MODEL_ID" \
            --dataset "$CACHE_FILE" \
            --conditions "$CONDITIONS_FILE" \
            --split "$SPLIT" \
            --output_dir "$OUTPUT_BASE" \
            --max_new_tokens "$MAX_TOK" \
            --conditions_filter cot_constrained baseline 2>&1; then
            echo "  Completed: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
        else
            echo "  FAILED: $MODEL_ID (max_tokens=$MAX_TOK)"
            FAILED+=("$MODEL_ID@$MAX_TOK")
        fi
    done

    # Merge predictions for this token budget
    MERGED_DIR="$OUTPUT_BASE/cot_ablation_${SPLIT}_tok${MAX_TOK}_${TIMESTAMP}_job${SLURM_JOB_ID:-local}"
    mkdir -p "$MERGED_DIR"

    > "$MERGED_DIR/predictions.jsonl"
    NEW_DIRS_FILE="$(mktemp)"
    find "$OUTPUT_BASE" -maxdepth 1 -type d -name '20*' 2>/dev/null | sort > "$NEW_DIRS_FILE" || true
    while IFS= read -r RUN_DIR; do
        [ -f "$RUN_DIR/predictions.jsonl" ] || continue
        cat "$RUN_DIR/predictions.jsonl" >> "$MERGED_DIR/predictions.jsonl"
    done < <(comm -13 "$EXISTING_DIRS_FILE" "$NEW_DIRS_FILE")
    rm -f "$EXISTING_DIRS_FILE" "$NEW_DIRS_FILE"

    PRED_COUNT=$(wc -l < "$MERGED_DIR/predictions.jsonl")
    echo ""
    echo "Merged $PRED_COUNT predictions for max_tokens=$MAX_TOK -> $MERGED_DIR"

    # Compute metrics for this subset
    if [ "$PRED_COUNT" -gt 0 ]; then
        python -m src.analysis.compute_metrics \
            --predictions "$MERGED_DIR/predictions.jsonl" \
            --output_dir "$MERGED_DIR"
    fi
done

# ── Summary ──────────────────────────────────────────────────────────────
echo ""
echo "================================================================"
echo "CoT ablation complete!"
echo "================================================================"
echo "Finished: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo ""
echo "Results directories:"
for MAX_TOK in "${MAX_TOKENS_VALUES[@]}"; do
    ls -d "$OUTPUT_BASE/cot_ablation_${SPLIT}_tok${MAX_TOK}_"* 2>/dev/null || echo "  (none for tok=$MAX_TOK)"
done

if [ ${#FAILED[@]} -gt 0 ]; then
    echo ""
    echo "WARNING: ${#FAILED[@]} run(s) failed:"
    for f in "${FAILED[@]}"; do echo "  - $f"; done
fi

# ── Move SLURM logs ──────────────────────────────────────────────────────
SLURM_OUT="${SLURM_SUBMIT_DIR:-$PROJECT_ROOT}/pf-cot-ablation_${SLURM_JOB_ID:-local}.out"
SLURM_ERR="${SLURM_SUBMIT_DIR:-$PROJECT_ROOT}/pf-cot-ablation_${SLURM_JOB_ID:-local}.err"
if [ -f "$SLURM_OUT" ]; then mv "$SLURM_OUT" "$PROJECT_ROOT/logs/"; fi
if [ -f "$SLURM_ERR" ]; then mv "$SLURM_ERR" "$PROJECT_ROOT/logs/"; fi
