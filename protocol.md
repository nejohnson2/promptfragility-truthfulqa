# Evaluation Protocol — Prompt Fragility in TruthfulQA MC

**Status:** LOCKED — do not modify after first dev run.

## 1. Dataset

- **Source:** TruthfulQA, loaded via `datasets` library (`truthful_qa`, `multiple_choice` config).
- **Task:** Multiple choice (MC1 — single correct answer).
- **Split procedure:** Deterministic 10% dev / 90% final split using `seed=42`.  All condition
  exploration, debugging, and parser tuning must use the dev split.  The final split is reserved
  for reported results.

## 2. Models

| Short name | HuggingFace identifier | Params | Role |
|---|---|---|---|
| Qwen2.5-3B-Instruct | `Qwen/Qwen2.5-3B-Instruct` | 3B | Qwen small |
| Qwen2.5-7B-Instruct | `Qwen/Qwen2.5-7B-Instruct` | 7B | Qwen medium |
| Qwen2.5-72B-Instruct | `Qwen/Qwen2.5-72B-Instruct` | 72B | Qwen large |
| Llama-3.1-8B-Instruct | `meta-llama/Llama-3.1-8B-Instruct` | 8B | Llama medium |
| Llama-3.1-70B-Instruct | `meta-llama/Llama-3.1-70B-Instruct` | 70B | Llama large |
| Mistral-7B-Instruct-v0.3 | `mistralai/Mistral-7B-Instruct-v0.3` | 7B | Cross-family |
| Gemma-3-12B-IT | `google/gemma-3-12b-it` | 12B | Cross-family |
| Phi-4-mini-instruct | `microsoft/Phi-4-mini-instruct` | 3.8B | Small-but-strong |

**Analysis axes:**
- *Scale within family:* Qwen 3B → 7B → 72B and Llama 8B → 70B test whether
  scaling reduces prompt fragility.
- *Cross-family at ~7-12B:* Qwen-7B, Llama-8B, Mistral-7B, Gemma-12B isolate
  training recipe effects at similar scale.

Models ≤12B run on a single GPU (fp16).  70B+ models use multi-GPU or 4-bit
quantization.  4-bit quantization (bitsandbytes NF4) is available as a flag.

## 3. Decoding Parameters

| Parameter | Value |
|---|---|
| `temperature` | 0.0 |
| `do_sample` | False |
| `top_p` | 1.0 |
| `max_new_tokens` | 10 |
| `seed` (torch manual seed) | 42 |

Identical for all models and conditions.

## 4. Prompt Conditions

All conditions are defined in `prompts/conditions.yaml`.

| ID | Category | Description |
|---|---|---|
| `baseline` | — | Direct instruction: "Choose the best answer." |
| `expert` | Instruction wording | "You are an expert…" framing |
| `cautious` | Instruction wording | "Be careful and think critically…" framing |
| `concise` | Instruction wording | "Reply with the letter only." |
| `avoid_misconceptions` | Instruction wording | "Avoid common misconceptions…" |
| `numeric_labels` | Formatting | Labels 1/2/3/4 instead of A/B/C/D |
| `single_line` | Formatting | All options on a single line |
| `code_block` | Formatting | Question wrapped in a code/monospace block |
| `distractor_sentence` | Benign distractor | Irrelevant sentence prepended |
| `distractor_paragraph` | Benign distractor | Irrelevant paragraph prepended |
| `polite` | Social style | "Please kindly…" wording |
| `direct` | Social style | Blunt, no filler wording |
| `cot_constrained` | Stress | "Think step by step" but require single-label output |

Total: 1 baseline + 12 perturbations = 13 conditions.

## 5. Metrics

### Per-model accuracy metrics (across conditions)

| Metric | Definition |
|---|---|
| Baseline accuracy | Accuracy under the `baseline` condition |
| Mean accuracy | Mean accuracy over all 13 conditions |
| Worst-case accuracy | Min accuracy over all 13 conditions |
| Std dev | Standard deviation of accuracy over conditions |
| Range | Max accuracy − min accuracy |
| Invalid rate | Fraction of outputs where no valid label was parsed |
| **PSI (Prompt Sensitivity Index)** | `baseline_acc − worst_acc` |

### Ranking stability metrics (across models)

| Metric | Definition |
|---|---|
| Kendall τ | Kendall rank correlation between baseline ranking and condition-k ranking |
| Pairwise rank flip rate | Fraction of model pairs whose relative order changes across conditions |
| Top-1 instability | Number of conditions where the top-ranked model differs from baseline top-1 |

## 6. Statistical Tests

- **Paired bootstrap** (10,000 resamples, seed 123) over questions per model:
  - δ = acc(condition_k) − acc(baseline) for each question
  - Report median δ and 95% CI [low, high]
- Save results to `bootstrap_deltas.csv`.

## 7. Figures

1. Heatmap: accuracy by model × condition
2. Bar chart: per-model accuracy across conditions
3. Kendall τ by condition
4. Baseline vs worst-case scatter

## 8. Reproducibility

- Fixed seeds everywhere (data split, torch, numpy).
- All runs log `metadata.json` with git hash, package versions, GPU info, decoding params.
- Predictions stored as JSONL with full provenance per record.
