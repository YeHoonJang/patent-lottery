"""
Benchmark runner — evaluates our system against PILOT-Bench ground truth.

Task: Given examiner_findings text, predict issue_type (101/102/103/112).
This tests whether our system correctly identifies rejection bases.

Metrics: Exact Match, Macro-F1, Micro-F1 (same as PILOT-Bench paper)
"""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

from src.utils.llm_adapter import LLMAdapter

logger = logging.getLogger("vpatent.benchmark")

ISSUE_LABELS = ["101", "102", "103", "112", "Others"]

ISSUE_PREDICTION_PROMPT = """You are a US patent law expert analyzing PTAB (Patent Trial and Appeal Board) ex parte appeal cases.

Given the examiner's findings and the appellant's arguments from a PTAB decision, identify which statutory grounds (issue types) are being contested.

Possible issue types:
- 101: Patent eligibility (35 U.S.C. §101) — abstract idea, law of nature
- 102: Novelty (35 U.S.C. §102) — anticipation by prior art
- 103: Obviousness (35 U.S.C. §103) — obvious over prior art combination
- 112: Written description / Enablement (35 U.S.C. §112)
- Others: Other statutory grounds

A case may involve MULTIPLE issue types simultaneously."""


def predict_issue_type(
    llm: LLMAdapter,
    examiner_findings: str,
    appellant_arguments: str = "",
) -> list[str]:
    """Predict issue_type from examiner findings using LLM."""
    user_prompt = f"""### Examiner's Findings
{examiner_findings[:3000]}

{f"### Appellant's Arguments{chr(10)}{appellant_arguments[:2000]}" if appellant_arguments else ""}

Based on the above, identify ALL statutory grounds being contested.

Respond with JSON:
{{"issue_type": ["101", "103"]}}

Only include issue types that are explicitly or clearly implicitly discussed. Use the labels: 101, 102, 103, 112, Others."""

    result = llm.call_json(ISSUE_PREDICTION_PROMPT, user_prompt, role_name="issue_prediction")
    if result and "issue_type" in result:
        return [str(x) for x in result["issue_type"]]
    return []


def compute_metrics(predictions: list[set], ground_truths: list[set]) -> dict:
    """Compute Exact Match, Macro-F1, Micro-F1 for multi-label classification."""
    n = len(predictions)
    if n == 0:
        return {}

    # Exact Match
    exact_matches = sum(1 for p, g in zip(predictions, ground_truths) if p == g)
    exact_match = exact_matches / n

    # Per-label TP/FP/FN for F1
    label_tp = defaultdict(int)
    label_fp = defaultdict(int)
    label_fn = defaultdict(int)

    for pred, gold in zip(predictions, ground_truths):
        for label in ISSUE_LABELS:
            in_pred = label in pred
            in_gold = label in gold
            if in_pred and in_gold:
                label_tp[label] += 1
            elif in_pred and not in_gold:
                label_fp[label] += 1
            elif not in_pred and in_gold:
                label_fn[label] += 1

    # Macro-F1
    f1_scores = {}
    for label in ISSUE_LABELS:
        tp = label_tp[label]
        fp = label_fp[label]
        fn = label_fn[label]
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores[label] = {"precision": round(precision, 4), "recall": round(recall, 4), "f1": round(f1, 4),
                            "tp": tp, "fp": fp, "fn": fn}

    macro_f1 = sum(f["f1"] for f in f1_scores.values()) / len(f1_scores) if f1_scores else 0

    # Micro-F1
    total_tp = sum(label_tp.values())
    total_fp = sum(label_fp.values())
    total_fn = sum(label_fn.values())
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0

    return {
        "n": n,
        "exact_match": round(exact_match, 4),
        "macro_f1": round(macro_f1, 4),
        "micro_f1": round(micro_f1, 4),
        "micro_precision": round(micro_precision, 4),
        "micro_recall": round(micro_recall, 4),
        "per_label": f1_scores,
    }


def run_benchmark(
    dataset_path: str,
    output_dir: str,
    llm: LLMAdapter,
    max_cases: int = 50,
    include_appellant: bool = False,
) -> dict:
    """
    Run issue_type prediction benchmark.

    Args:
        dataset_path: Path to benchmark_dataset.json
        output_dir: Where to save results
        llm: LLM adapter to use
        max_cases: Max cases to evaluate (for cost control)
        include_appellant: Include appellant_arguments in input
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(dataset_path) as f:
        data = json.load(f)

    cases = data["cases"][:max_cases]
    logger.info("Running benchmark on %d cases (mode=%s)", len(cases), llm.mode)

    predictions = []
    ground_truths = []
    details = []
    start = time.time()

    for i, case in enumerate(cases):
        gt = set(case["ground_truth"]["issue_type"])
        ground_truths.append(gt)

        appellant = case["appellant_arguments"] if include_appellant else ""
        pred_list = predict_issue_type(llm, case["examiner_findings"], appellant)
        pred = set(pred_list)
        predictions.append(pred)

        match = "match" if pred == gt else "mismatch"
        details.append({
            "case_id": case["case_id"],
            "ground_truth": sorted(gt),
            "prediction": sorted(pred),
            "exact_match": pred == gt,
        })

        if (i + 1) % 10 == 0 or i == 0:
            logger.info("[%d/%d] %s GT=%s PRED=%s", i + 1, len(cases), match, sorted(gt), sorted(pred))

    elapsed = time.time() - start
    metrics = compute_metrics(predictions, ground_truths)
    metrics["elapsed_seconds"] = round(elapsed, 1)
    metrics["cases_per_second"] = round(len(cases) / elapsed, 2) if elapsed > 0 else 0

    # Save results
    results = {
        "config": {
            "dataset": str(dataset_path),
            "max_cases": max_cases,
            "llm_mode": llm.mode,
            "include_appellant": include_appellant,
        },
        "metrics": metrics,
        "details": details,
    }

    with open(output_dir / "benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    # Print summary
    print(f"\n{'='*60}")
    print(f"PILOT-Bench Issue Type Prediction")
    print(f"{'='*60}")
    print(f"Cases: {metrics['n']}")
    print(f"Time:  {metrics['elapsed_seconds']}s ({metrics['cases_per_second']} cases/s)")
    print(f"")
    print(f"Exact Match:  {metrics['exact_match']:.1%}")
    print(f"Macro-F1:     {metrics['macro_f1']:.4f}")
    print(f"Micro-F1:     {metrics['micro_f1']:.4f}")
    print(f"")
    print(f"Per-label F1:")
    for label, scores in metrics.get("per_label", {}).items():
        print(f"  {label:>6}: P={scores['precision']:.3f} R={scores['recall']:.3f} F1={scores['f1']:.3f} (TP={scores['tp']} FP={scores['fp']} FN={scores['fn']})")
    print(f"\nSaved to: {output_dir}/benchmark_results.json")

    return results


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser(description="Run PILOT-Bench issue type prediction benchmark")
    parser.add_argument("--dataset", default="data/pilot_bench/benchmark/benchmark_dataset.json")
    parser.add_argument("--output", default="outputs/benchmark")
    parser.add_argument("--mode", choices=["mock", "codex", "openai", "auto"], default="mock")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--max_cases", type=int, default=50)
    parser.add_argument("--include_appellant", action="store_true")
    args = parser.parse_args()

    llm = LLMAdapter(mode=args.mode, model=args.model)

    # For mock mode, register a simple mock
    if args.mode == "mock":
        from src.utils.llm_adapter import register_mock
        # Mock that looks for keywords in the prompt
        # (This won't be used in real benchmarks, just for architecture testing)

    run_benchmark(args.dataset, args.output, llm, args.max_cases, args.include_appellant)
