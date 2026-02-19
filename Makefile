# Makefile — Prompt Fragility in TruthfulQA
# ==========================================================================
#
# Targets:
#   make paper        — Regenerate tables/figures from latest results, copy
#                       into paper/.  Does NOT compile the LaTeX document.
#   make figures      — Re-run plots.py against the active run directory.
#   make tables       — Re-run generate_tables.py to produce .tex fragments.
#   make analysis     — Run compute_metrics + stats on the active run.
#   make eval-dev     — Full evaluation pipeline on the dev split.
#   make eval-final   — Full evaluation pipeline on the final split.
#   make clean-paper  — Remove generated paper artifacts.
#   make all          — Run the full pipeline (eval-final + analysis + paper).
#
# Configuration:
#   Set RUN_DIR below to point to the merged results directory that should
#   feed into the paper.  This is the single source of truth.
# ==========================================================================

.PHONY: all paper figures tables analysis eval-dev eval-final clean-paper help

# ── Configuration ──────────────────────────────────────────────────────────
# Active merged run directory — point this to the latest completed run.
RUN_DIR  ?= results/runs/merged_final_20260218_120244_job14605
PAPER_DIR = paper
GEN_DIR   = $(PAPER_DIR)/generated
FIG_SRC   = $(RUN_DIR)/figures
FIG_DST   = $(PAPER_DIR)/figures
PYTHON   ?= .venv/bin/python

# ── High-level targets ─────────────────────────────────────────────────────

all: analysis paper

help:
	@echo "Targets:"
	@echo "  make paper      — Update tables + figures in paper/ from RUN_DIR"
	@echo "  make figures    — Regenerate plot PNGs from analysis outputs"
	@echo "  make tables     — Regenerate LaTeX table fragments + macros"
	@echo "  make analysis   — Run compute_metrics + stats + plots"
	@echo "  make eval-dev   — Full pipeline on dev split"
	@echo "  make eval-final — Full pipeline on final split"
	@echo "  make clean-paper— Remove generated paper artifacts"
	@echo ""
	@echo "Configuration:"
	@echo "  RUN_DIR=$(RUN_DIR)"

# ── Paper integration (no LaTeX compilation) ───────────────────────────────

paper: tables figures-copy
	@echo "✓ Paper tables and figures updated from $(RUN_DIR)"
	@echo "  Generated .tex fragments: $(GEN_DIR)/"
	@echo "  Figures copied to:        $(FIG_DST)/"
	@echo ""
	@echo "  The LaTeX document is NOT compiled automatically."
	@echo "  Open paper/main.tex in your editor and build manually."

tables: $(RUN_DIR)/aggregated.csv $(RUN_DIR)/model_summary.csv \
        $(RUN_DIR)/ranking_metrics.json $(RUN_DIR)/bootstrap_deltas.csv
	@mkdir -p $(GEN_DIR)
	$(PYTHON) scripts/generate_tables.py \
		--run_dir $(RUN_DIR) \
		--output_dir $(GEN_DIR)

figures-copy: $(FIG_SRC)/accuracy_heatmap.png
	@mkdir -p $(FIG_DST)
	cp -v $(FIG_SRC)/accuracy_heatmap.png    $(FIG_DST)/
	cp -v $(FIG_SRC)/accuracy_bars.png       $(FIG_DST)/
	cp -v $(FIG_SRC)/kendall_tau.png         $(FIG_DST)/
	cp -v $(FIG_SRC)/baseline_vs_worst.png   $(FIG_DST)/
	@echo "✓ Figures copied"

# ── Analysis pipeline ──────────────────────────────────────────────────────

analysis: $(RUN_DIR)/predictions.jsonl
	$(PYTHON) -m src.analysis.compute_metrics \
		--predictions $(RUN_DIR)/predictions.jsonl \
		--output_dir $(RUN_DIR)
	$(PYTHON) -m src.analysis.stats \
		--predictions $(RUN_DIR)/predictions.jsonl \
		--output_dir $(RUN_DIR)
	$(PYTHON) -m src.analysis.plots \
		--aggregated $(RUN_DIR)/aggregated.csv \
		--ranking_metrics $(RUN_DIR)/ranking_metrics.json \
		--model_summary $(RUN_DIR)/model_summary.csv \
		--output_dir $(RUN_DIR)/figures
	@echo "✓ Analysis complete: $(RUN_DIR)"

figures: $(RUN_DIR)/aggregated.csv $(RUN_DIR)/ranking_metrics.json \
         $(RUN_DIR)/model_summary.csv
	@mkdir -p $(RUN_DIR)/figures
	$(PYTHON) -m src.analysis.plots \
		--aggregated $(RUN_DIR)/aggregated.csv \
		--ranking_metrics $(RUN_DIR)/ranking_metrics.json \
		--model_summary $(RUN_DIR)/model_summary.csv \
		--output_dir $(RUN_DIR)/figures

# ── Evaluation targets ─────────────────────────────────────────────────────

eval-dev:
	bash scripts/run_matrix.sh dev

eval-final:
	bash scripts/run_matrix.sh final

# ── Cleanup ────────────────────────────────────────────────────────────────

clean-paper:
	rm -rf $(GEN_DIR)
	@echo "Removed $(GEN_DIR)"
