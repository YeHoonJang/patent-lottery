# Patent Lottery — Code and Data Release

Companion repository for the paper *"Simulating the Patent Lottery: From Single-Sample Examination to Structured Judgment Space."*

This repository contains the pipeline code, configuration files, the 100-case evaluation cohort, the 50-case ablation subset, all per-case experiment results, and the LLM-as-judge coherence scores reported in the paper.

The patent full-text corpus (applicant and prior-art specifications for 14,714 PTAB cases) is released separately due to size; see *Data* below.

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
│   ├── ablation_oversight/per_case/        Control + treatment (oversight) decisions
│   └── coherence_scores/                   LLM-as-judge per-trace scoring (1,400 traces)
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

The companion patent corpus covering 14,714 PTAB cases (applicant + prior-art full-text specifications) is released separately. URL is given in the paper.

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

### Analysis scripts

| Script | Purpose |
|---|---|
| `analyze_ablation_oversight.py` | Per-archetype Δ favorable rate + paired permutation test |
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

Coherence-score files have one JSON per `(case, examiner)` pair under `coherence_scores/<case>/<examiner>.json` with five-item binary scores, a justification string per item, and a total ∈ {0,…,5}.

---

## License

Code is released under the MIT License (see `LICENSE`).

The companion patent corpus consists of public USPTO records and is released under CC-BY 4.0 in the linked repository.
