# Prompt Fragility in TruthfulQA: How Much Do Benign Prompt Changes Affect Multiple-Choice Accuracy and Model Rankings?

## Abstract

We measure the sensitivity of TruthfulQA multiple-choice (MC1) accuracy to benign prompt
perturbations across five open-source instruction-tuned LLMs.  Using 13 prompt conditions
spanning instruction wording, formatting, benign distractors, social style, and a chain-of-thought
stress variant, we quantify per-model accuracy instability and cross-model ranking instability.
We introduce the Prompt Sensitivity Index (PSI) — the gap between baseline and worst-case
accuracy — and report paired bootstrap confidence intervals for all condition deltas.  Our results
show that [RESULTS PLACEHOLDER: e.g., "prompt wording alone can shift accuracy by up to X
percentage points, and the top-ranked model changes in Y out of 13 conditions"].

## 1. Introduction

- LLM benchmarks are treated as stable measurements, but the prompt is a free variable.
- TruthfulQA is widely used to assess factual accuracy; MC format should be less sensitive
  than open-ended generation, yet even MC results may shift with prompt wording.
- Research questions:
  1. How much does MC accuracy vary under semantically equivalent prompt changes?
  2. Which categories of perturbation cause the largest drops?
  3. Do model rankings remain stable across prompt variants?

## 2. Related Work

- Benchmark sensitivity and prompt sensitivity literature (Sclar et al., 2023; Mizrahi et al., 2024).
- TruthfulQA original paper (Lin et al., 2022) and its MC evaluation protocol.
- Robustness evaluation frameworks and adversarial prompt studies.

## 3. Method

### 3.1 Dataset and Split
- TruthfulQA MC1 (817 questions), 10% dev / 90% final split, seed 42.

### 3.2 Models
- Table of 5 models (Mistral-7B-Instruct, Llama-2-7B-Chat, Phi-2, Gemma-2B-IT, TinyLlama-1.1B-Chat).
- All run in float16 on a single GPU; deterministic decoding (temperature=0).

### 3.3 Prompt Conditions
- 1 baseline + 12 perturbations across 5 categories (Table 1).
- Jinja-based template system for reproducibility.
- All conditions are semantically equivalent: the question, choices, and correct answer are unchanged.

### 3.4 Metrics
- Per-model: baseline accuracy, mean/worst/std/range accuracy, PSI, invalid rate.
- Cross-model: Kendall tau vs baseline ranking, pairwise rank flip rate, top-1 instability.
- Statistics: paired bootstrap (10k resamples) for delta CIs.

## 4. Results

### 4.1 Accuracy Sensitivity
- **Figure 1:** Heatmap of accuracy across models × conditions.
- **Figure 2:** Per-model bar charts showing accuracy spread.
- Table: model summary with baseline_acc, worst_acc, PSI, std_acc.

### 4.2 Perturbation Category Analysis
- Which category (instruction wording, formatting, distractor, style, stress) causes the
  largest drops on average?
- Are some models more robust to specific perturbation categories?

### 4.3 Ranking Instability
- **Figure 3:** Kendall tau per condition.
- **Figure 4:** Baseline vs worst-case scatter.
- Rank flip rate and top-1 instability count.

### 4.4 Statistical Significance
- Bootstrap delta table: which condition deltas have CIs excluding zero?

## 5. Discussion

- Practical implications: benchmark scores are not single numbers but distributions over
  prompt choices.
- Recommendations: report sensitivity ranges alongside point estimates; standardize prompt
  templates across benchmark comparisons.
- Limitations: 5 models, single GPU scale; MC format only; English only; perturbations are
  hand-crafted, not exhaustive.

## 6. Conclusion

- Benign prompt changes cause non-trivial accuracy variation in TruthfulQA MC.
- Model rankings are [stable/unstable] under these perturbations.
- PSI provides a simple, actionable metric for reporting prompt sensitivity.

## References

- Lin, S., Hilton, J., & Evans, O. (2022). TruthfulQA: Measuring how models mimic human falsehoods.
- Sclar, M., et al. (2023). Quantifying language models' sensitivity to spurious features in prompt design.
- Mizrahi, M., et al. (2024). State of what art? A call for multi-prompt LLM evaluation.

## Appendix

- A: Full prompt text for all 13 conditions.
- B: Per-question breakdown of condition sensitivity.
- C: Detailed bootstrap delta tables.
