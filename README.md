# Patent Lottery — Code and Data Release

Companion repository for the paper *"The Patent Lottery: Generating a Structured Judgment Space from Single-Sample Examination"* (EACL 2027 Findings).

This repository contains the pipeline code, configuration files, the 100-case evaluation cohort, the 50-case ablation subset, all per-case experiment results, and the LLM-as-judge coherence scores reported in the paper.

The patent full-text corpus (applicant and prior-art specifications for 15,531 PTAB decisions) is released separately due to size at https://huggingface.co/datasets/Yehoon/ptab_patents; see *Data* below.

---

## Repository Layout

```
.
├── data/
│   ├── eval_100_enriched_input_v3.json     100-case cohort (main experiment)
│   └── ablation_50_cases.json              50-case stratified subset (ablations)
├── configs/
│   ├── archetypes_v3/                      Seven examiner archetype YAMLs (O1–O5, C1, C2)
│   └── baselines/                          Vanilla + persona baseline YAMLs
├── src/                                    Pipeline core (roles, examiner population, LLM adapter, ...)
├── scripts/                                Experiment runners + analysis + sampling
├── results/
│   ├── main_exp_100/per_case/              100 × 14 decisions (main experiment)
│   ├── baseline_100/per_case/              100 × 14 vanilla baseline decisions
│   ├── ablation_cross_model/
│   │   ├── sonnet45/per_case/              50 cases × 14 examiners × Claude Sonnet 4.5
│   │   ├── deepseek_v3/per_case/
│   │   ├── qwen3_32b/per_case/
│   │   ├── qwen3_8b_local/per_case/
│   │   └── mistral_large/per_case/
│   ├── ablation_oversight/per_case/        Treatment (oversight) decisions; control = main_exp_100 on the 50 shared cases
│   ├── baselines_pilot10/                  10-case pilot baselines (Appendix H): vanilla/, persona/, gpt_persona/ (single case)
│   ├── coherence_scores/                   LLM-as-judge per-trace scoring, Claude Sonnet 4.6 (1,400 traces)
│   └── coherence_scores_gpt56/             LLM-as-judge per-trace scoring, GPT-5.6 Sol (same 1,400 traces)
└── requirements.txt
```

---

## Setup

```bash
pip install -r requirements.txt

# API keys (set the ones you need)
export OPENROUTER_API_KEY=...               # Qwen3, DeepSeek, Mistral (via OpenRouter)
export ANTHROPIC_API_KEY=...                # Claude Sonnet 4.5 (and judge model)
```

For the locally-served Qwen3-8B run, point `LLM_BASE_URL` to a HuggingFace `text-generation-inference` endpoint serving `Qwen/Qwen3-8B`.

---

## Data

* **`data/eval_100_enriched_input_v3.json`** — 100 PTAB cases stratified from the 7,698-case source pool (45 Affirmed / 35 Reversed / 15 Affirmed-in-Part / 5 Other with new grounds of rejection). Each case includes claims, full description, and prior-art texts.
* **`data/ablation_50_cases.json`** — Stratified random sample (`seed=42`) of 50 cases from the 100-case cohort, used by both the cross-model and oversight ablations.

The companion patent corpus (applicant + prior-art full-text specifications; 15,531 decisions with the applicant specification, 7,698 with both) is released separately at https://huggingface.co/datasets/Yehoon/ptab_patents (CC BY 4.0).

---

## Reproducing the Experiments

All runners use `Path(__file__).resolve().parents[1]` as the project root, so they can be executed from anywhere as long as the directory structure is preserved.

### Main experiment (100 cases × 14 examiners, Qwen3-235B-A22B)

```bash
python scripts/run_exp1_90.py        # initial 90-case batch
python scripts/merge_to_full_100.py  # merge with pilot 10 to form full 100
```

Outputs are written to `outputs/exp1_main/per_case/` (mirror of `results/main_exp_100/per_case/`).

### Vanilla baseline (100 cases × 14 examiners, all dimensions = 0.50)

```bash
python scripts/run_baseline_100.py
python scripts/retry_failed_baseline.py  # re-runs §112 stage failures with lower parallelism
```

### Cross-model ablation (50 cases × 14 examiners × 5 backbones)

```bash
# First, regenerate the 50-case subset (deterministic, seed=42)
python scripts/sample_ablation_50.py

# Run one backbone at a time
python scripts/run_ablation.py --model anthropic/claude-sonnet-4.5 --output-name sonnet45
python scripts/run_ablation.py --model deepseek/deepseek-chat-v3-0324 --output-name deepseek_v3
python scripts/run_ablation.py --model qwen/qwen3-32b --output-name qwen3_32b
python scripts/run_ablation.py --model mistralai/mistral-large-2512 --output-name mistral_large
python scripts/run_ablation.py --model Qwen/Qwen3-8B --output-name qwen3_8b_local
```

Claude runs route through the Anthropic API directly (the LLM adapter forbids OpenRouter for `claude*` models when `ANTHROPIC_API_KEY` is set). All other backbones go through OpenRouter.

### Institutional oversight ablation

```bash
MODEL=qwen/qwen3-235b-a22b-2507 python scripts/run_ablation_oversight.py
```

Runs both `oversight=0` (control) and `oversight=1` (treatment) over the same 50 cases.

### LLM-as-judge coherence scoring

```bash
python scripts/score_coherence_llm.py
```

Scores every trace under `outputs/exp1_full_100/traces/` with Claude Sonnet 4.6 on the 5-item legal coherence checklist.

```bash
python scripts/score_coherence_gpt.py --n 1400
```

Scores the same traces with GPT-5.6 Sol using the same checklist and prompt (output under `outputs/coherence_scores_gpt56/`).

```bash
python scripts/aggregate_coherence_scores.py
```

Recomputes `_aggregate.json` from the per-trace files (`--dir results/coherence_scores_gpt56` for the second judge). Each trace's total is the sum of its five item scores (the judge's own `total` field is one point low in 8 of 1,400 traces).

### Analysis scripts

| Script | Purpose |
|---|---|
| `oversight_signflip_test.py` | Paired sign-flip permutation test on per-case archetype favorable fractions (Appendix I, Table 11) |
| `analyze_ablation_oversight.py` | Per-archetype Δ favorable rate with McNemar's test on grant decisions (diagnostic; not the test reported in the paper) |
| `compute_s101_recall.py` | §101 recall figures reported in Limitations |
| `compute_additive_baseline.py` | Additive-baseline predictions for the composite archetypes (Appendix B) |
| `aggregate_coherence_scores.py` | Summary statistics of the coherence scores (Section 5.3, Appendix D) |
| `compare_coherence_judges.py` | Agreement between the two coherence judges (Section 5.3, Appendix D) |
| `compute_robustness_r.py` | Per-archetype Pearson `r` across the five cross-model runs |
| `q4_permutation_test.py` | Profile-attributable disagreement permutation test |

---

## Result Format

Each `per_case/<proceeding_number>.json` has the shape:

```json
{
  "proceeding_number": "2017005064",
  "baseline_type": "main",
  "gt_issues": ["103"],
  "gt_outcome": "Reversed",
  "grant_rate": 0.286,
  "amend_rate": 0.429,
  "examiner_results": [
    {
      "examiner": "lenient_advocate_0044",
      "archetype": "lenient_advocate",
      "strictness": 0.248,
      "combination_threshold": 0.676,
      "seniority": 0.402,
      "tech_specialization": 0.612,
      "pred": ["103"],
      "decision": "grant",
      "decision_score": 0.9
    },
    ...
  ]
}
```

Coherence-score files have one JSON per `(case, examiner)` pair under `coherence_scores/<case>/<examiner>.json` (Claude Sonnet 4.6) and `coherence_scores_gpt56/<case>/<examiner>.json` (GPT-5.6 Sol) with five-item binary scores, a justification string per item, and the judge-reported total ∈ {0,…,5}. Aggregate statistics use the sum of the five item scores (`_aggregate.json`).

---

## License

Code is released under the MIT License (see `LICENSE`).

The companion patent corpus consists of public USPTO records and is released under CC BY 4.0 at https://huggingface.co/datasets/Yehoon/ptab_patents.
