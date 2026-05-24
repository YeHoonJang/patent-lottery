#!/usr/bin/env python3
"""
Run an ablation experiment on the fixed 50-case sample with a given model.

All four model-robustness experiments (Sonnet 4.6, DeepSeek-V3, Qwen3-32B,
Qwen3-8B) use this same script — only the --model and --output-name
arguments change. Resume support is built in: re-running skips cases
already saved under per_case/.

Examples:
  python scripts/run_ablation.py \\
    --model anthropic/claude-sonnet-4.6 --output-name sonnet46

  python scripts/run_ablation.py \\
    --model deepseek/deepseek-v3 --output-name deepseek_v3 --workers 4

  python scripts/run_ablation.py \\
    --model qwen/qwen3-32b --output-name qwen3_32b

  python scripts/run_ablation.py \\
    --model qwen/qwen3-8b --output-name qwen3_8b
"""
import argparse
import json
import logging
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.utils.enriched_loader import dict_to_case_enriched
from src.examiner.population import generate_population
from src.pipeline.run_case import run_single_examiner
from src.utils.llm_adapter import LLMAdapter

ABLATION_INPUT = REPO_ROOT / "data/ablation_50_cases.json"
ARCHETYPE_DIR = REPO_ROOT / "configs/archetypes_v3"


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True,
                   help="OpenRouter model id (e.g., anthropic/claude-sonnet-4.6)")
    p.add_argument("--output-name", required=True,
                   help="Short name for output directory (outputs/ablation_<name>/)")
    p.add_argument("--workers", type=int, default=4,
                   help="Cases processed in parallel (default: 4)")
    p.add_argument("--timeout", type=int, default=300,
                   help="Per-LLM-call timeout seconds (default: 300)")
    p.add_argument("--limit", type=int, default=None,
                   help="Cap number of cases (for smoke testing). Default: all 50.")
    p.add_argument("--input", default=str(ABLATION_INPUT),
                   help="Path to input JSON with {'cases': [...]} (default: data/ablation_50_cases.json)")
    return p.parse_args()


def setup_logging(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(output_dir / "run.log"),
        ],
    )
    return logging.getLogger("ablation")


def load_cases(input_path: str = None):
    p = Path(input_path) if input_path else ABLATION_INPUT
    src = json.loads(p.read_text())
    return src["cases"]


def discover_completed(per_case_dir: Path, expected_decisions: int) -> set:
    """Scan per_case dir for fully-completed cases (all examiner decisions valid).

    A case is "complete" only if all expected examiners finished with a valid
    decision (grant/amend_suggested/reject). Cases with any 'error' are re-run
    on resume — important when credits run out mid-pipeline.
    """
    valid_decs = {"grant", "amend_suggested", "reject"}
    done = set()
    for f in per_case_dir.glob("*.json"):
        try:
            r = json.loads(f.read_text())
            er = r.get("examiner_results", []) if r else []
            valid_count = sum(1 for x in er if x.get("decision") in valid_decs)
            if valid_count == expected_decisions:
                done.add(r["proceeding_number"])
        except Exception:
            pass
    return done


def extract_issues(trace):
    issues = set()
    oa = trace.office_action
    if isinstance(oa, dict):
        for g in oa.get("rejection_grounds", []):
            b = g.get("basis", "")
            for code in ("101", "102", "103", "112"):
                if code in b:
                    issues.add(code)
    for na in (trace.novelty_assessments or []):
        d = na if isinstance(na, dict) else na.__dict__
        if not d.get("is_novel", True):
            issues.add("102")
    for o in (trace.obviousness_assessments or []):
        d = o if isinstance(o, dict) else o.__dict__
        if d.get("is_obvious", False):
            issues.add("103")
    return issues


def process_case(args):
    case_idx, total, cd, population, model_id, timeout, output_dir, logger = args
    case = dict_to_case_enriched(cd)
    pn = cd["proceeding_number"]
    gt = cd.get("ground_truth", {}) or {}

    case_start = time.time()
    results = []

    for examiner in population:
        llm = LLMAdapter(mode="deepseek", model=model_id, timeout=timeout)
        trace_dir = str(output_dir / "traces" / pn / examiner.display_name)
        try:
            trace = run_single_examiner(
                case=case, examiner=examiner, llm=llm, output_dir=trace_dir
            )
            pred = extract_issues(trace)
            fd = trace.final_decision
            dec = fd.get("decision", "?") if isinstance(fd, dict) else "?"
            score = fd.get("decision_score", 0) if isinstance(fd, dict) else 0
            weighing = fd.get("weighing", []) if isinstance(fd, dict) else []
        except Exception as e:
            logger.error("[%s/%s] FAILED: %s", pn, examiner.display_name, e)
            dec, score, pred, weighing = "error", 0, set(), []

        results.append({
            "examiner": examiner.display_name,
            "archetype": examiner.archetype_name,
            "strictness": round(examiner.strictness, 3),
            "combination_threshold": round(examiner.combination_threshold, 3),
            "seniority": round(examiner.seniority, 3),
            "tech_specialization": round(examiner.tech_specialization, 3),
            "pred": sorted(pred),
            "decision": dec,
            "decision_score": score,
            "weighing": weighing,
        })

    elapsed = time.time() - case_start
    grant = sum(1 for r in results if r["decision"] == "grant")
    amend = sum(1 for r in results if r["decision"] == "amend_suggested")
    reject = sum(1 for r in results if r["decision"] == "reject")
    err = sum(1 for r in results if r["decision"] == "error")

    case_result = {
        "case_id": pn,
        "proceeding_number": pn,
        "gt_issues": gt.get("issue_type", []),
        "gt_outcome": gt.get("subdecision_coarse", ""),
        "grant_rate": grant / len(population),
        "amend_rate": amend / len(population),
        "examiner_results": results,
    }

    per_case_path = output_dir / "per_case" / f"{pn}.json"
    per_case_path.write_text(json.dumps(case_result, indent=2, ensure_ascii=False))

    logger.info("[%d/%d] %s GT=%s g=%d a=%d r=%d err=%d (%.0fs)",
                case_idx + 1, total, pn, gt.get("subdecision_coarse", "?"),
                grant, amend, reject, err, elapsed)
    return case_result


def main():
    args = parse_args()

    if not os.environ.get("OPENROUTER_API_KEY"):
        sys.exit("ERROR: OPENROUTER_API_KEY env var not set. Export it before running.")

    output_dir = REPO_ROOT / "outputs" / f"ablation_{args.output_name}"
    per_case_dir = output_dir / "per_case"
    per_case_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logging(output_dir)
    logger.info("=" * 70)
    logger.info("Ablation run")
    logger.info("  model       : %s", args.model)
    logger.info("  output dir  : %s", output_dir)
    logger.info("  workers     : %d", args.workers)
    logger.info("  timeout     : %ds", args.timeout)
    logger.info("  input       : %s", args.input)
    logger.info("=" * 70)

    cases = load_cases(args.input)
    if args.limit:
        cases = cases[: args.limit]
        logger.info("Limited to first %d cases for smoke test", args.limit)

    population = generate_population(
        archetype_dir=str(ARCHETYPE_DIR),
        instances_per_archetype=2,
        base_seed=42,
    )
    expected = len(population)
    logger.info("Population: %d examiners from 7 archetypes", expected)

    done = discover_completed(per_case_dir, expected)
    if done:
        logger.info("Resume: skipping %d already-completed cases", len(done))
        cases = [c for c in cases if c["proceeding_number"] not in done]
    logger.info("Cases to run this session: %d", len(cases))

    if not cases:
        logger.info("Nothing to do. Exiting.")
        return

    start = time.time()
    all_results = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(process_case,
                      (i, len(cases), cd, population, args.model,
                       args.timeout, output_dir, logger)): i
            for i, cd in enumerate(cases)
        }
        for fut in as_completed(futures):
            try:
                all_results.append(fut.result())
            except Exception as e:
                logger.error("Case CRASHED: %s", e)

    elapsed = time.time() - start
    logger.info("=" * 70)
    logger.info("DONE — %d cases this session, %.1f min total",
                len(all_results), elapsed / 60)

    # Quick archetype summary
    arch_total = Counter()
    arch_grant = Counter()
    for r in all_results:
        for er in r["examiner_results"]:
            arch_total[er["archetype"]] += 1
            if er["decision"] == "grant":
                arch_grant[er["archetype"]] += 1
    logger.info("\nGrant rate by archetype (this session only):")
    for a in sorted(arch_total):
        t = arch_total[a]
        g = arch_grant.get(a, 0)
        logger.info("  %-22s %d/%d = %.1f%%", a, g, t, 100 * g / t if t else 0)


if __name__ == "__main__":
    main()
