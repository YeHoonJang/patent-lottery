"""
Pipeline Benchmark — runs full 7-step examination on enriched PILOT-Bench cases
and compares system's rejection bases with ground truth.

This is Scenario B: the real test of whether our system can act as a patent examiner.
"""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path

from src.models.schema import PatentCase, Claim, ClaimType
from src.examiner.population import instantiate_examiner, load_archetype, load_era_context
from src.pipeline.run_case import run_single_examiner
from src.utils.llm_adapter import LLMAdapter

logger = logging.getLogger("vpatent.pipeline_bench")

ISSUE_LABELS = ["101", "102", "103", "112"]


def enriched_to_patent_case(enriched_case: dict) -> PatentCase:
    """Convert an enriched PILOT-Bench case to PatentCase."""
    pd = enriched_case.get("patent_data", {})

    claims = []
    for cl in pd.get("claims", []):
        ct = ClaimType.DEPENDENT if cl.get("claim_type") == "dependent" else ClaimType.INDEPENDENT
        claims.append(Claim(
            claim_number=cl.get("claim_number", 0),
            claim_type=ct,
            text=cl.get("text", ""),
            depends_on=cl.get("depends_on"),
        ))

    # Use examiner_findings as a proxy for "description_summary" context
    # This gives the system the same information a real examiner would have considered
    examiner_context = enriched_case.get("examiner_findings", "")

    return PatentCase(
        case_id=enriched_case.get("case_id", "unknown"),
        title=pd.get("title", "Unknown Patent"),
        abstract=pd.get("abstract", ""),
        description_summary=f"[Examiner's prior findings for context]\n{examiner_context[:2000]}" if examiner_context else "",
        claims=claims,
        prior_art=[],  # PILOT-Bench doesn't provide prior art refs
        ipc_codes=pd.get("ipc_codes", []),
        applicant="",
        filing_date="",
        jurisdiction="USPTO",
    )


def extract_system_issues(trace) -> set:
    """Extract issue types from system's examination trace."""
    issues = set()

    # From office action rejection grounds
    oa = trace.office_action
    if isinstance(oa, dict):
        for ground in oa.get("rejection_grounds", []):
            basis = ground.get("basis", "")
            if "101" in basis:
                issues.add("101")
            elif "102" in basis:
                issues.add("102")
            elif "103" in basis:
                issues.add("103")
            elif "112" in basis:
                issues.add("112")

    # Also check novelty/obviousness/written_desc assessments directly
    for na in (trace.novelty_assessments or []):
        na_dict = na if isinstance(na, dict) else na.__dict__
        if not na_dict.get("is_novel", True):
            issues.add("102")

    for oa_item in (trace.obviousness_assessments or []):
        oa_dict = oa_item if isinstance(oa_item, dict) else oa_item.__dict__
        if oa_dict.get("is_obvious", False):
            issues.add("103")

    for wd in (trace.written_desc_assessments or []):
        wd_dict = wd if isinstance(wd, dict) else wd.__dict__
        if wd_dict.get("has_issues", False):
            issues.add("112")

    return issues


def compute_metrics(predictions: list, ground_truths: list) -> dict:
    """Compute multi-label metrics."""
    n = len(predictions)
    if n == 0:
        return {}

    exact_matches = sum(1 for p, g in zip(predictions, ground_truths) if p == g)

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

    f1_scores = {}
    for label in ISSUE_LABELS:
        tp, fp, fn = label_tp[label], label_fp[label], label_fn[label]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        f1_scores[label] = {"precision": round(p, 4), "recall": round(r, 4),
                            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}

    total_tp = sum(label_tp.values())
    total_fp = sum(label_fp.values())
    total_fn = sum(label_fn.values())
    micro_p = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    micro_r = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    micro_f1 = 2 * micro_p * micro_r / (micro_p + micro_r) if (micro_p + micro_r) > 0 else 0

    return {
        "n": n,
        "exact_match": round(exact_matches / n, 4),
        "macro_f1": round(sum(s["f1"] for s in f1_scores.values()) / len(f1_scores), 4),
        "micro_f1": round(micro_f1, 4),
        "per_label": f1_scores,
    }


def run_pipeline_benchmark(
    enriched_path: str,
    output_dir: str,
    archetype_path: str = "configs/archetypes/thorough_analyst.yaml",
    era_path: str = None,
    llm_mode: str = "codex",
    llm_model: str = "gpt-5.4-mini",
    max_cases: int = 10,
):
    """Run full pipeline benchmark on enriched PILOT-Bench cases."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(enriched_path) as f:
        data = json.load(f)

    cases = data["cases"][:max_cases]
    logger.info("Pipeline benchmark: %d cases, mode=%s, model=%s", len(cases), llm_mode, llm_model)

    # Create one examiner (thorough_analyst, neutral)
    archetype = load_archetype(archetype_path)
    era_context = load_era_context(era_path) if era_path else None
    examiner = instantiate_examiner(archetype, era_context, noise_seed=42)
    logger.info("Examiner: %s (strict=%.2f)", examiner.display_name, examiner.strictness)

    predictions = []
    ground_truths = []
    details = []
    start = time.time()

    for i, enriched_case in enumerate(cases):
        case = enriched_to_patent_case(enriched_case)
        gt = set(enriched_case["ground_truth"]["issue_type"])
        gt_decision = enriched_case["ground_truth"]["subdecision_coarse"]

        logger.info("[%d/%d] %s — US%s (%d claims) GT=%s",
                    i+1, len(cases), case.case_id, enriched_case["patent_number"],
                    len(case.claims), sorted(gt))

        llm = LLMAdapter(mode=llm_mode, model=llm_model, timeout=180)
        trace = run_single_examiner(
            case=case,
            examiner=examiner,
            llm=llm,
            output_dir=str(output_dir / "traces"),
        )

        pred = extract_system_issues(trace)
        predictions.append(pred)
        ground_truths.append(gt)

        sys_decision = "?"
        if isinstance(trace.final_decision, dict):
            sys_decision = trace.final_decision.get("decision", "?")

        match = "MATCH" if pred == gt else "MISS"
        logger.info("[%d/%d] %s — PRED=%s GT=%s | sys_dec=%s gt_dec=%s",
                    i+1, len(cases), match, sorted(pred), sorted(gt), sys_decision, gt_decision)

        details.append({
            "case_id": case.case_id,
            "patent_number": enriched_case["patent_number"],
            "n_claims": len(case.claims),
            "ground_truth_issues": sorted(gt),
            "predicted_issues": sorted(pred),
            "exact_match": pred == gt,
            "ground_truth_decision": gt_decision,
            "system_decision": sys_decision,
        })

    elapsed = time.time() - start
    metrics = compute_metrics(predictions, ground_truths)
    metrics["elapsed_seconds"] = round(elapsed, 1)

    results = {
        "config": {
            "enriched_path": enriched_path,
            "archetype": archetype.get("archetype_name", "?"),
            "llm_mode": llm_mode,
            "llm_model": llm_model,
            "max_cases": max_cases,
        },
        "metrics": metrics,
        "details": details,
    }

    with open(output_dir / "pipeline_benchmark_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"PIPELINE BENCHMARK (Scenario B)")
    print(f"{'='*60}")
    print(f"Cases:        {metrics['n']}")
    print(f"Time:         {metrics['elapsed_seconds']}s")
    print(f"Examiner:     {examiner.display_name}")
    print(f"")
    print(f"Exact Match:  {metrics['exact_match']:.1%}")
    print(f"Macro-F1:     {metrics['macro_f1']:.4f}")
    print(f"Micro-F1:     {metrics['micro_f1']:.4f}")
    print(f"")
    print(f"Per-label:")
    for label, s in metrics.get("per_label", {}).items():
        print(f"  {label}: P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f}")
    print(f"\nSaved to: {output_dir}/pipeline_benchmark_results.json")

    return results


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--enriched", default="data/pilot_bench/enriched/enriched_benchmark.json")
    parser.add_argument("--output", default="outputs/pipeline_benchmark")
    parser.add_argument("--archetype", default="configs/archetypes/thorough_analyst.yaml")
    parser.add_argument("--era", default=None)
    parser.add_argument("--mode", default="codex")
    parser.add_argument("--model", default="gpt-5.4-mini")
    parser.add_argument("--max_cases", type=int, default=10)
    args = parser.parse_args()

    run_pipeline_benchmark(
        args.enriched, args.output, args.archetype, args.era,
        args.mode, args.model, args.max_cases,
    )
