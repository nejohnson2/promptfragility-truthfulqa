#!/usr/bin/env bash
# run_matrix.sh — Full evaluation pipeline.
#
# Usage:
#   bash scripts/run_matrix.sh [dev|final] [--quantize]
#
# This script:
#   1. Fetches/caches the TruthfulQA dataset
#   2. Runs evaluation for each model (continues on failure)
#   3. Merges predictions across runs
#   4. Computes metrics, bootstrap CIs, and generates plots
#
# For SLURM clusters, use scripts/slurm_pipeline.sbatch instead.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$PROJECT_ROOT"

# ── Defaults ──────────────────────────────────────────────────────────────
SPLIT="${1:-dev}"
QUANTIZE_FLAG=""
TIMESTAMP="$(date -u +%Y%m%d_%H%M%S)"
RUN_TAG="${SPLIT}_${TIMESTAMP}"

# Parse args
for arg in "$@"; do
    case "$arg" in
        --split)  shift; SPLIT="$1"; shift ;;
        dev|final) SPLIT="$arg" ;;
        --quantize) QUANTIZE_FLAG="--quantize_4bit" ;;
    esac
done

# Create logs directory
mkdir -p logs

# Models to evaluate (must match protocol.md)
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

echo "============================================"
echo "Prompt Fragility — TruthfulQA MC Evaluation"
echo "============================================"
echo "Split:     $SPLIT"
echo "Run tag:   $RUN_TAG"
echo "Models:    ${#MODELS[@]}"
echo "Host:      $(hostname)"
echo "Date:      $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "Python:    $(python --version 2>&1)"
echo ""

# ── GPU diagnostics ──────────────────────────────────────────────────────
if command -v nvidia-smi &>/dev/null; then
    echo "── GPU Info ──"
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    echo ""
fi

# ── Step 1: Fetch data ───────────────────────────────────────────────────
echo "[Step 1/6] Fetching TruthfulQA dataset"
if [ ! -f "$CACHE_FILE" ]; then
    python data/fetch_truthfulqa.py --out "$CACHE_FILE" --seed 42
else
    echo "  Dataset cache exists: $CACHE_FILE ($(wc -l < "$CACHE_FILE") records)"
fi
echo ""

# ── Step 2: Run evaluation for each model ────────────────────────────────
echo "[Step 2/6] Running evaluation matrix"

FAILED_MODELS=()
MODEL_IDX=0
TOTAL_MODELS=${#MODELS[@]}

for MODEL_ID in "${MODELS[@]}"; do
    MODEL_IDX=$((MODEL_IDX + 1))
    MODEL_SHORT=$(echo "$MODEL_ID" | sed 's|.*/||')
    echo ""
    echo "  [$MODEL_IDX/$TOTAL_MODELS] $MODEL_SHORT"
    echo "  Started: $(date -u '+%H:%M:%S UTC')"

    if python -m src.eval.run_eval \
        --model_id "$MODEL_ID" \
        --dataset "$CACHE_FILE" \
        --conditions "$CONDITIONS_FILE" \
        --split "$SPLIT" \
        --output_dir "$OUTPUT_BASE" \
        $QUANTIZE_FLAG 2>&1; then
        echo "  Completed: $(date -u '+%H:%M:%S UTC')"
    else
        echo "  FAILED: $MODEL_ID (exit code $?)"
        FAILED_MODELS+=("$MODEL_ID")
        echo "  Continuing with next model..."
    fi
done

# Report failures
if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo ""
    echo "WARNING: ${#FAILED_MODELS[@]} model(s) failed:"
    for m in "${FAILED_MODELS[@]}"; do
        echo "  - $m"
    done
    echo ""
fi

# ── Step 3: Merge predictions ────────────────────────────────────────────
echo ""
echo "[Step 3/6] Merging predictions"

MERGED_DIR="$OUTPUT_BASE/merged_${RUN_TAG}"
mkdir -p "$MERGED_DIR/figures"

> "$MERGED_DIR/predictions.jsonl"
for RUN_DIR in "$OUTPUT_BASE"/20*; do
    [ -d "$RUN_DIR" ] || continue
    [ -f "$RUN_DIR/predictions.jsonl" ] || continue
    cat "$RUN_DIR/predictions.jsonl" >> "$MERGED_DIR/predictions.jsonl"
done

PRED_COUNT=$(wc -l < "$MERGED_DIR/predictions.jsonl")
echo "  Merged → $PRED_COUNT prediction records"

if [ "$PRED_COUNT" -eq 0 ]; then
    echo "ERROR: No predictions found. Exiting."
    exit 1
fi

# ── Step 4: Aggregate and compute metrics ────────────────────────────────
echo ""
echo "[Step 4/6] Computing metrics"
python -m src.analysis.compute_metrics \
    --predictions "$MERGED_DIR/predictions.jsonl" \
    --output_dir "$MERGED_DIR"

# ── Step 5: Bootstrap statistics ─────────────────────────────────────────
echo ""
echo "[Step 5/6] Computing bootstrap confidence intervals"
python -m src.analysis.stats \
    --predictions "$MERGED_DIR/predictions.jsonl" \
    --output_dir "$MERGED_DIR"

# ── Step 6: Generate plots ───────────────────────────────────────────────
echo ""
echo "[Step 6/6] Generating plots"
python -m src.analysis.plots \
    --aggregated "$MERGED_DIR/aggregated.csv" \
    --ranking_metrics "$MERGED_DIR/ranking_metrics.json" \
    --model_summary "$MERGED_DIR/model_summary.csv" \
    --output_dir "$MERGED_DIR/figures"

echo ""
echo "============================================"
echo "Done! Results in: $MERGED_DIR"
echo "============================================"
echo ""
echo "Contents:"
ls -lh "$MERGED_DIR/"
echo ""
echo "Figures:"
ls -lh "$MERGED_DIR/figures/" 2>/dev/null || echo "  (none)"

if [ ${#FAILED_MODELS[@]} -gt 0 ]; then
    echo ""
    echo "WARNING: ${#FAILED_MODELS[@]} model(s) failed. See output above."
    exit 1
fi
