#!/usr/bin/env bash
# run_matrix.sh — Full evaluation pipeline.
#
# Usage:
#   bash scripts/run_matrix.sh [--split dev|final] [--quantize]
#
# This script:
#   1. Fetches/caches the TruthfulQA dataset
#   2. Runs evaluation for each model
#   3. Aggregates results and generates plots

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
echo ""

# ── Step 1: Fetch data ───────────────────────────────────────────────────
if [ ! -f "$CACHE_FILE" ]; then
    echo ">> Fetching TruthfulQA dataset …"
    python data/fetch_truthfulqa.py --out "$CACHE_FILE" --seed 42
    echo ""
else
    echo ">> Dataset cache exists: $CACHE_FILE"
fi

# ── Step 2: Run evaluation for each model ────────────────────────────────
echo ""
echo ">> Running evaluation matrix …"
for MODEL_ID in "${MODELS[@]}"; do
    MODEL_SHORT=$(echo "$MODEL_ID" | sed 's|.*/||')
    echo ""
    echo "  ──── Model: $MODEL_SHORT ────"
    python -m src.eval.run_eval \
        --model_id "$MODEL_ID" \
        --dataset "$CACHE_FILE" \
        --conditions "$CONDITIONS_FILE" \
        --split "$SPLIT" \
        --output_dir "$OUTPUT_BASE" \
        $QUANTIZE_FLAG
done

# ── Step 3: Find the most recent run directory ──────────────────────────
# All model runs are saved in separate timestamped dirs. We need to
# merge them or analyze the latest one. For simplicity, we merge all
# predictions.jsonl from the latest batch into a combined file.
echo ""
echo ">> Merging predictions from latest runs …"

MERGED_DIR="$OUTPUT_BASE/merged_${RUN_TAG}"
mkdir -p "$MERGED_DIR/figures"

# Concatenate all predictions from runs created after our timestamp
> "$MERGED_DIR/predictions.jsonl"
for RUN_DIR in "$OUTPUT_BASE"/20*; do
    [ -d "$RUN_DIR" ] || continue
    [ -f "$RUN_DIR/predictions.jsonl" ] || continue
    cat "$RUN_DIR/predictions.jsonl" >> "$MERGED_DIR/predictions.jsonl"
done

echo "  Merged predictions: $(wc -l < "$MERGED_DIR/predictions.jsonl") records"

# ── Step 4: Aggregate and compute metrics ────────────────────────────────
echo ""
echo ">> Computing metrics …"
python -m src.analysis.compute_metrics \
    --predictions "$MERGED_DIR/predictions.jsonl" \
    --output_dir "$MERGED_DIR"

# ── Step 5: Bootstrap statistics ─────────────────────────────────────────
echo ""
echo ">> Computing bootstrap confidence intervals …"
python -m src.analysis.stats \
    --predictions "$MERGED_DIR/predictions.jsonl" \
    --output_dir "$MERGED_DIR"

# ── Step 6: Generate plots ───────────────────────────────────────────────
echo ""
echo ">> Generating plots …"
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
ls -la "$MERGED_DIR/"
echo ""
echo "Figures:"
ls -la "$MERGED_DIR/figures/" 2>/dev/null || echo "  (none)"
