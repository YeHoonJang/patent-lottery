"""
Pipeline Benchmark v2 — uses BQ-enriched data (claims + description + citations).

Key improvement: prior art is now taken from BigQuery citations,
not hallucinated by LLM. This should dramatically reduce 102 FP.
"""

import json
import logging
import time
from collections import defaultdict
from pathlib import Path

from src.models.schema import PatentCase, Claim, ClaimType, PriorArtReference
from src.examiner.population import instantiate_examiner, load_archetype, load_era_context
from src.pipeline.run_case import run_single_examiner
from src.utils.llm_adapter import LLMAdapter

logger = logging.getLogger("vpatent.pipeline_bench_v2")

ISSUE_LABELS = ["101", "102", "103", "112"]


def bq_enriched_to_patent_case(case: dict) -> PatentCase:
    """Convert BQ-enriched PILOT-Bench case to PatentCase with full data."""
    bq = case.get("bq_data", {})

    # Parse claims from full claims text
    claims = []
    claims_text = bq.get("claims_text", "")
    if claims_text:
        import re
        # Split by claim numbers: "1. ...", "2. ..."
        parts = re.split(r'\n\s*(\d+)\.\s+', claims_text)
        # parts = ['preamble', '1', 'claim1 text', '2', 'claim2 text', ...]
        i = 1
        while i < len(parts) - 1:
            try:
                claim_num = int(parts[i])
                claim_text = parts[i + 1].strip()
                if len(claim_text) > 20:
                    is_dep = bool(re.search(r'claim \d+|according to claim', claim_text, re.I))
                    dep_on = None
                    if is_dep:
                        dep_match = re.search(r'claim (\d+)', claim_text, re.I)
                        if dep_match:
                            dep_on = int(dep_match.group(1))
                    claims.append(Claim(
                        claim_number=claim_num,
                        claim_type=ClaimType.DEPENDENT if is_dep else ClaimType.INDEPENDENT,
                        text=claim_text[:2000],
                        depends_on=dep_on,
                    ))
            except (ValueError, IndexError):
                pass
            i += 2

    # If parsing failed, use Google Patents data as fallback
    if not claims and case.get("patent_data", {}).get("claims"):
        for cl in case["patent_data"]["claims"]:
            ct = ClaimType.DEPENDENT if cl.get("claim_type") == "dependent" else ClaimType.INDEPENDENT
            claims.append(Claim(
                claim_number=cl.get("claim_number", 0),
                claim_type=ct,
                text=cl.get("text", "")[:2000],
                depends_on=cl.get("depends_on"),
            ))

    # Build prior art from enriched citations (with abstract + claims)
    prior_art = []
    enriched_refs = bq.get("cited_refs_enriched", [])
    if enriched_refs:
        for ref in enriched_refs[:5]:
            if ref.get("abstract"):
                prior_art.append(PriorArtReference(
                    ref_id=ref["ref_id"],
                    title=ref.get("title", ""),
                    abstract=ref.get("abstract", ""),
                    relevant_text=ref.get("claims_preview", "")[:500],
                    source="BigQuery-citation",
                ))
    else:
        # Fallback to plain citation IDs
        for ref_id in bq.get("cited_refs", [])[:5]:
            prior_art.append(PriorArtReference(
                ref_id=ref_id,
                title=f"Cited reference {ref_id}",
                abstract="",
                relevant_text="",
                source="BigQuery-citation",
            ))

    # Use description summary (first 3000 chars)
    desc = bq.get("desc_summary", "")[:10000]

    return PatentCase(
        case_id=case.get("case_id", "unknown"),
        title=bq.get("title", "") or case.get("patent_data", {}).get("title", "Unknown"),
        abstract=bq.get("abstract", "") or "",
        description_summary=desc,
        claims=claims,
        prior_art=prior_art,
        ipc_codes=[c for c in (bq.get("ipc_codes", "") or "").split("|") if c][:5],
        applicant=bq.get("assignees", ""),
        filing_date=bq.get("filing_date", ""),
        jurisdiction="USPTO",
    )


def extract_system_issues(trace) -> set:
    """Extract issue types from system's examination trace."""
    issues = set()
    oa = trace.office_action
    if isinstance(oa, dict):
        for ground in oa.get("rejection_grounds", []):
            basis = ground.get("basis", "")
            if "101" in basis: issues.add("101")
            elif "102" in basis: issues.add("102")
            elif "103" in basis: issues.add("103")
            elif "112" in basis: issues.add("112")

    for na in (trace.novelty_assessments or []):
        d = na if isinstance(na, dict) else na.__dict__
        if not d.get("is_novel", True): issues.add("102")

    for oa_item in (trace.obviousness_assessments or []):
        d = oa_item if isinstance(oa_item, dict) else oa_item.__dict__
        if d.get("is_obvious", False): issues.add("103")

    for wd in (trace.written_desc_assessments or []):
        d = wd if isinstance(wd, dict) else wd.__dict__
        if d.get("has_issues", False): issues.add("112")

    return issues


def compute_metrics(predictions, ground_truths):
    n = len(predictions)
    if n == 0: return {}

    exact = sum(1 for p, g in zip(predictions, ground_truths) if p == g) / n

    label_tp = defaultdict(int)
    label_fp = defaultdict(int)
    label_fn = defaultdict(int)
    for pred, gold in zip(predictions, ground_truths):
        for label in ISSUE_LABELS:
            if label in pred and label in gold: label_tp[label] += 1
            elif label in pred: label_fp[label] += 1
            elif label in gold: label_fn[label] += 1

    per_label = {}
    for label in ISSUE_LABELS:
        tp, fp, fn = label_tp[label], label_fp[label], label_fn[label]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0
        per_label[label] = {"precision": round(p, 4), "recall": round(r, 4),
                            "f1": round(f1, 4), "tp": tp, "fp": fp, "fn": fn}

    total_tp = sum(label_tp.values())
    total_fp = sum(label_fp.values())
    total_fn = sum(label_fn.values())
    mp = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0
    mr = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0
    mf1 = 2 * mp * mr / (mp + mr) if (mp + mr) > 0 else 0

    return {"n": n, "exact_match": round(exact, 4), "micro_f1": round(mf1, 4),
            "macro_f1": round(sum(s["f1"] for s in per_label.values()) / len(per_label), 4),
            "per_label": per_label}


def run_pipeline_benchmark_v2(
    enriched_path: str,
    output_dir: str,
    archetype_path: str = "configs/archetypes/thorough_analyst.yaml",
    era_path: str = None,
    llm_mode: str = "deepseek",
    llm_model: str = "qwen/qwen3-235b-a22b-2507",
    max_cases: int = 3,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(enriched_path) as f:
        data = json.load(f)

    cases = data["cases"][:max_cases]
    logger.info("Pipeline benchmark v2: %d cases (BQ enriched), mode=%s", len(cases), llm_mode)

    archetype = load_archetype(archetype_path)
    era_context = load_era_context(era_path) if era_path else None
    examiner = instantiate_examiner(archetype, era_context, noise_seed=42)

    predictions = []
    ground_truths = []
    details = []
    start = time.time()

    for i, enriched_case in enumerate(cases):
        case = bq_enriched_to_patent_case(enriched_case)
        gt = set(enriched_case["ground_truth"]["issue_type"])
        ground_truths.append(gt)

        logger.info("[%d/%d] %s — %s (%d claims, %d prior_art) GT=%s",
                    i+1, len(cases), case.case_id,
                    enriched_case.get("patent_number", "?"),
                    len(case.claims), len(case.prior_art), sorted(gt))

        llm = LLMAdapter(mode=llm_mode, model=llm_model, timeout=180)
        trace = run_single_examiner(case=case, examiner=examiner, llm=llm,
                                     output_dir=str(output_dir / "traces"))

        pred = extract_system_issues(trace)
        predictions.append(pred)

        sys_dec = "?"
        if isinstance(trace.final_decision, dict):
            sys_dec = trace.final_decision.get("decision", "?")

        match = "MATCH" if pred == gt else "MISS"
        logger.info("[%d/%d] %s — PRED=%s GT=%s | dec=%s",
                    i+1, len(cases), match, sorted(pred), sorted(gt), sys_dec)

        details.append({
            "case_id": case.case_id,
            "patent_number": enriched_case.get("patent_number", ""),
            "n_claims": len(case.claims),
            "n_prior_art": len(case.prior_art),
            "has_description": bool(case.description_summary),
            "ground_truth_issues": sorted(gt),
            "predicted_issues": sorted(pred),
            "exact_match": pred == gt,
            "system_decision": sys_dec,
            "ground_truth_decision": enriched_case["ground_truth"].get("subdecision_coarse", ""),
        })

    elapsed = time.time() - start
    metrics = compute_metrics(predictions, ground_truths)
    metrics["elapsed_seconds"] = round(elapsed, 1)

    results = {"config": {"enriched_path": enriched_path, "llm_mode": llm_mode,
                          "llm_model": llm_model, "max_cases": max_cases,
                          "data_source": "BigQuery enriched"},
               "metrics": metrics, "details": details}

    with open(output_dir / "pipeline_benchmark_v2_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"PIPELINE BENCHMARK v2 (BQ Enriched)")
    print(f"{'='*60}")
    print(f"Cases: {metrics.get('n', 0)}  |  Time: {metrics.get('elapsed_seconds', 0)}s")
    print(f"Data: claims + description + citations from BigQuery")
    print(f"")
    print(f"Exact Match:  {metrics.get('exact_match', 0):.1%}")
    print(f"Macro-F1:     {metrics.get('macro_f1', 0):.4f}")
    print(f"Micro-F1:     {metrics.get('micro_f1', 0):.4f}")
    print(f"")
    print(f"Per-label:")
    for label, s in metrics.get("per_label", {}).items():
        print(f"  {label}: P={s['precision']:.3f} R={s['recall']:.3f} F1={s['f1']:.3f} (TP={s['tp']} FP={s['fp']} FN={s['fn']})")
    print(f"\nPer-case:")
    for d in details:
        tag = "MATCH" if d["exact_match"] else "MISS "
        print(f"  {tag} GT={str(d['ground_truth_issues']):25s} PRED={str(d['predicted_issues']):30s} claims={d['n_claims']} refs={d['n_prior_art']}")

    return results


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--enriched", default="data/pilot_bench/enriched_bq/pilot_bench_bq_v2.json")
    parser.add_argument("--output", default="outputs/pipeline_benchmark_v2_bq")
    parser.add_argument("--mode", default="deepseek")
    parser.add_argument("--model", default="qwen/qwen3-235b-a22b-2507")
    parser.add_argument("--max_cases", type=int, default=3)
    args = parser.parse_args()

    run_pipeline_benchmark_v2(args.enriched, args.output, llm_mode=args.mode,
                               llm_model=args.model, max_cases=args.max_cases)
