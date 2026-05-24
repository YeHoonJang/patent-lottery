#!/usr/bin/env python3
"""
Exp 1: 90 remaining cases x 14 examiners = 1,260 decisions.
Pilot 10 cases already completed separately.

Features:
- Resume support: skips cases with existing results
- Per-case result saving: each case saved immediately after completion
- Graceful error handling: failed examiners recorded, not crashed
- Progress logging: detailed timing and status
"""
import sys, json, logging, os, time, signal
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed

from pathlib import Path
REPO_ROOT = str(Path(__file__).resolve().parents[1])

sys.path.insert(0, REPO_ROOT)
os.environ["OPENROUTER_API_KEY"] = os.environ.get("OPENROUTER_API_KEY", "YOUR_KEY_HERE")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{REPO_ROOT}/outputs/exp1_90.log"),
    ]
)
logger = logging.getLogger("exp1")

from src.utils.enriched_loader import dict_to_case_enriched
from src.examiner.population import generate_population
from src.pipeline.run_case import run_single_examiner
from src.utils.llm_adapter import LLMAdapter

# ─── Config ───────────────────────────────────────
PARALLEL_CASES = 4          # cases processed in parallel
MODEL = "qwen/qwen3-235b-a22b-2507"
TIMEOUT = 300               # per LLM call
OUTPUT_DIR = Path(f"{REPO_ROOT}/outputs/exp1_main")
RESULTS_FILE = OUTPUT_DIR / "results_90.json"

# ─── Load data ────────────────────────────────────
logger.info("Loading data...")

with open(f"{REPO_ROOT}/data/exp1_remaining_90.json") as f:
    remaining = json.load(f)
target_pns = set(remaining["proceeding_numbers"])

with open(f"{REPO_ROOT}/data/eval_100_enriched_input_v3.json") as f:
    enriched = json.load(f)

cases_data = [c for c in enriched["cases"] if c["proceeding_number"] in target_pns]
logger.info("Loaded %d cases to process", len(cases_data))

# Load GT
with open(f"{REPO_ROOT}/data/experiment_ready_cases.json") as f:
    ready = json.load(f)
gt_by_pn = {}
for c in ready["cases"]:
    pn = str(c.get("proceeding_number", ""))
    gt_by_pn[pn] = c.get("ground_truth", {})

# ─── Population ───────────────────────────────────
population = generate_population(
    archetype_dir=f"{REPO_ROOT}/configs/archetypes_v3",
    instances_per_archetype=2,
    base_seed=42,
)
logger.info("Population: %d examiners from 7 archetypes", len(population))
for ex in population:
    logger.info("  %s: strict=%.2f comb=%.2f senior=%.2f tech=%.2f",
                ex.display_name, ex.strictness, ex.combination_threshold,
                ex.seniority, ex.tech_specialization)

# ─── Resume support ───────────────────────────────
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
per_case_dir = OUTPUT_DIR / "per_case"
per_case_dir.mkdir(exist_ok=True)

completed_pns = set()
for f in per_case_dir.glob("*.json"):
    try:
        with open(f) as fh:
            r = json.load(fh)
        if r and len(r.get("examiner_results", [])) == 14:
            completed_pns.add(r["proceeding_number"])
    except Exception:
        pass

if completed_pns:
    logger.info("Resuming: %d cases already completed, %d remaining",
                len(completed_pns), len(target_pns) - len(completed_pns))
    cases_data = [c for c in cases_data if c["proceeding_number"] not in completed_pns]

logger.info("Cases to run: %d", len(cases_data))

# ─── Helper ───────────────────────────────────────
def extract_issues(trace):
    issues = set()
    oa = trace.office_action
    if isinstance(oa, dict):
        for g in oa.get("rejection_grounds", []):
            b = g.get("basis", "")
            for code in ["101", "102", "103", "112"]:
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

# ─── Process one case ─────────────────────────────
def process_case(args):
    case_idx, cd, total = args
    case = dict_to_case_enriched(cd)
    pn = cd["proceeding_number"]
    gt = gt_by_pn.get(pn, {})
    gt_issues = gt.get("issue_type", [])
    gt_outcome = gt.get("subdecision_coarse", "")

    case_start = time.time()
    results = []

    for examiner in population:
        llm = LLMAdapter(mode="deepseek", model=MODEL, timeout=TIMEOUT)
        trace_dir = str(OUTPUT_DIR / "traces" / pn / examiner.display_name)
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
            dec = "error"
            score = 0
            pred = set()
            weighing = []

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

    case_time = time.time() - case_start
    grant_count = sum(1 for r in results if r["decision"] == "grant")
    amend_count = sum(1 for r in results if r["decision"] == "amend_suggested")
    reject_count = sum(1 for r in results if r["decision"] == "reject")
    error_count = sum(1 for r in results if r["decision"] == "error")

    case_result = {
        "case_id": "PILOT-%s" % pn,
        "proceeding_number": pn,
        "gt_issues": gt_issues,
        "gt_outcome": gt_outcome,
        "grant_rate": grant_count / len(population),
        "amend_rate": amend_count / len(population),
        "examiner_results": results,
    }

    # Save per-case immediately (resume support)
    with open(per_case_dir / ("%s.json" % pn), "w") as f:
        json.dump(case_result, f, indent=2, ensure_ascii=False)

    logger.info("[Case %d/%d] %s GT=%s(%s) g=%d a=%d r=%d err=%d (%.0fs)",
                case_idx + 1, total, pn, gt_issues, gt_outcome,
                grant_count, amend_count, reject_count, error_count, case_time)

    return case_result

# ─── Run ──────────────────────────────────────────
logger.info("=" * 70)
logger.info("EXP 1: %d cases x %d examiners = %d decisions",
            len(cases_data), len(population), len(cases_data) * len(population))
logger.info("Parallel: %d cases at a time", PARALLEL_CASES)
logger.info("Output: %s", OUTPUT_DIR)
logger.info("=" * 70)

all_results = []
start_time = time.time()

with ThreadPoolExecutor(max_workers=PARALLEL_CASES) as executor:
    futures = {}
    for i, cd in enumerate(cases_data):
        future = executor.submit(process_case, (i, cd, len(cases_data)))
        futures[future] = i

    for future in as_completed(futures):
        idx = futures[future]
        try:
            result = future.result()
            all_results.append(result)
        except Exception as e:
            logger.error("Case %d CRASHED: %s", idx, e)

# ─── Save all results ─────────────────────────────
with open(RESULTS_FILE, "w") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

# ─── Summary ──────────────────────────────────────
elapsed = time.time() - start_time
logger.info("")
logger.info("=" * 70)
logger.info("EXP 1 COMPLETE: %d cases, %d decisions, %.1f hours",
            len(all_results), len(all_results) * 14, elapsed / 3600)
logger.info("=" * 70)

# Archetype stats
arch_stats = Counter()
arch_total = Counter()
for r in all_results:
    for er in r["examiner_results"]:
        arch_total[er["archetype"]] += 1
        arch_stats[(er["archetype"], er["decision"])] += 1

logger.info("\nArchetype grant/amend/reject rates:")
for arch in sorted(set(a for a, _ in arch_stats)):
    t = arch_total[arch]
    g = arch_stats.get((arch, "grant"), 0)
    a = arch_stats.get((arch, "amend_suggested"), 0)
    r = arch_stats.get((arch, "reject"), 0)
    e = arch_stats.get((arch, "error"), 0)
    logger.info("  %-22s g=%d(%.0f%%) a=%d(%.0f%%) r=%d(%.0f%%) err=%d",
                arch, g, g/t*100, a, a/t*100, r, r/t*100, e)

# Disagreement
disagree = [r for r in all_results
            if 0 < sum(1 for er in r["examiner_results"] if er["decision"] in ["grant","amend_suggested"]) < 14]
logger.info("\nDisagreement cases: %d/%d (%.0f%%)",
            len(disagree), len(all_results),
            len(disagree)/len(all_results)*100 if all_results else 0)

# Errors
errors = [r for r in all_results
          for er in r["examiner_results"] if er["decision"] == "error"]
logger.info("Total error decisions: %d/%d", len(errors), len(all_results)*14)

logger.info("\nResults saved: %s", RESULTS_FILE)
logger.info("Per-case results: %s/", per_case_dir)
logger.info("Traces: %s/traces/", OUTPUT_DIR)
logger.info("Done!")
