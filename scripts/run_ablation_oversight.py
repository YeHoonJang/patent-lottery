#!/usr/bin/env python3
"""
Mini-ablation: institutional oversight effect on lenient archetypes (Q3 답).

Hypothesis: C2's 99.5% grant rate reflects unconstrained leniency in absence of
SPE / OPQA / asymmetric-risk oversight. Adding oversight_awareness should:
  - Reduce C2 grant rate toward realistic ~85-90% range
  - Reduce O3, O2 grant rates (strictness floor formation)
  - Preserve PTAB-Reversed agreement (no over-correction)

Design:
  - 30 stratified cases: 15 Reversed / 8 Affirmed-in-Part / 7 Affirmed
  - 3 archetypes: C2_lenient_advocate, O2_broad_interpreter, O3_easy_combiner
  - 2 instances per archetype = 6 examiners per condition
  - 2 conditions: oversight=0.0 (control), oversight=1.0 (treatment)
  - Total: 30 cases x 6 examiners x 2 conditions x 8 roles = 2,880 LLM calls

Run:
  MODEL=qwen/qwen3-235b-a22b-2507 python scripts/run_ablation_oversight.py

Output: outputs/ablation_oversight/per_case/{pn}_oversight{0|1}.json
"""
import sys, json, logging, os, time, random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

from pathlib import Path
REPO_ROOT = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO_ROOT)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{REPO_ROOT}/outputs/ablation_oversight.log"),
    ]
)
logger = logging.getLogger("ablation_oversight")

from src.utils.enriched_loader import dict_to_case_enriched
from src.examiner.population import generate_population
from src.pipeline.run_case import run_single_examiner
from src.utils.llm_adapter import LLMAdapter

# ─── Config ───────────────────────────────────────
MODEL = os.environ.get("MODEL", "qwen/qwen3-235b-a22b-2507")
PARALLEL_CASES = int(os.environ.get("PARALLEL_CASES", "4"))
TIMEOUT = 300
TARGET_ARCHETYPES = None  # None = all 7 archetypes (oversight effect across all)
INSTANCES_PER_ARCHETYPE = 2

OUTPUT_DIR = Path(f"{REPO_ROOT}/outputs/ablation_oversight")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
per_case_dir = OUTPUT_DIR / "per_case"
per_case_dir.mkdir(exist_ok=True)

logger.info("Model: %s", MODEL)
logger.info("Archetypes: %s", TARGET_ARCHETYPES if TARGET_ARCHETYPES else "ALL 7")

# ─── Load 50-case ablation sample (same as run_ablation.py) ──
with open(f"{REPO_ROOT}/data/ablation_50_cases.json") as f:
    ablation_data = json.load(f)
selected_cases = ablation_data["cases"]

# Build GT lookup from selected cases (data/ablation_50_cases.json includes ground_truth)
gt_by_pn = {}
for c in selected_cases:
    pn = str(c.get("proceeding_number", ""))
    gt_by_pn[pn] = c.get("ground_truth", {}) or {}

logger.info("Loaded %d cases from ablation_50_cases.json", len(selected_cases))
outcomes = [gt_by_pn[str(c["proceeding_number"])].get("subdecision_coarse", "?")
            for c in selected_cases]
from collections import Counter as _C
logger.info("Outcome distribution: %s", dict(_C(outcomes)))

# ─── Helpers ──────────────────────────────────────
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

def process_case(args):
    case_idx, cd, total, population, oversight_level = args
    case = dict_to_case_enriched(cd)
    pn = cd["proceeding_number"]
    gt = gt_by_pn.get(pn, {})
    gt_issues = gt.get("issue_type", [])
    gt_outcome = gt.get("subdecision_coarse", "")
    tag = f"oversight{int(oversight_level)}"

    case_start = time.time()
    results = []

    for examiner in population:
        llm = LLMAdapter(mode="deepseek", model=MODEL, timeout=TIMEOUT)
        trace_dir = str(OUTPUT_DIR / "traces" / tag / pn / examiner.display_name)
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
            logger.error("[%s/%s/%s] FAILED: %s", tag, pn, examiner.display_name, e)
            dec, score, pred, weighing = "error", 0, set(), []

        results.append({
            "examiner": examiner.display_name,
            "archetype": examiner.archetype_name,
            "oversight_awareness": examiner.oversight_awareness,
            "strictness": round(examiner.strictness, 3),
            "pred": sorted(pred),
            "decision": dec,
            "decision_score": score,
            "weighing": weighing,
        })

    case_time = time.time() - case_start
    grant_count = sum(1 for r in results if r["decision"] == "grant")
    amend_count = sum(1 for r in results if r["decision"] == "amend_suggested")
    reject_count = sum(1 for r in results if r["decision"] == "reject")

    case_result = {
        "case_id": f"ABL-{pn}",
        "proceeding_number": pn,
        "oversight_level": oversight_level,
        "gt_issues": gt_issues,
        "gt_outcome": gt_outcome,
        "grant_rate": grant_count / len(population) if population else 0,
        "amend_rate": amend_count / len(population) if population else 0,
        "reject_rate": reject_count / len(population) if population else 0,
        "examiner_results": results,
    }

    out_path = per_case_dir / f"{pn}_{tag}.json"
    with open(out_path, "w") as f:
        json.dump(case_result, f, indent=2, ensure_ascii=False)

    logger.info("[%s %d/%d] %s GT=%s g=%d a=%d r=%d (%.0fs)",
                tag, case_idx + 1, total, pn, gt_outcome,
                grant_count, amend_count, reject_count, case_time)
    return case_result

def run_condition(oversight_level: float):
    population = generate_population(
        archetype_dir=f"{REPO_ROOT}/configs/archetypes_v3",
        instances_per_archetype=INSTANCES_PER_ARCHETYPE,
        base_seed=42,
        archetype_filter=TARGET_ARCHETYPES,
        oversight_override=oversight_level,
    )
    logger.info("=" * 70)
    logger.info("CONDITION: oversight=%.2f | %d examiners | %d cases",
                oversight_level, len(population), len(selected_cases))
    logger.info("=" * 70)

    # Resume support: skip cases already complete (all valid decisions, no errors)
    tag = f"oversight{int(oversight_level)}"
    valid_decs = {"grant", "amend_suggested", "reject"}
    completed_pns = set()
    for f in per_case_dir.glob(f"*_{tag}.json"):
        try:
            with open(f) as fh:
                r = json.load(fh)
            er = r.get("examiner_results", []) if r else []
            valid_count = sum(1 for x in er if x.get("decision") in valid_decs)
            if valid_count == len(population):
                completed_pns.add(r["proceeding_number"])
        except Exception:
            pass
    cases_to_run = [c for c in selected_cases if str(c["proceeding_number"]) not in completed_pns]
    if completed_pns:
        logger.info("Resume: %d cases done, %d remaining", len(completed_pns), len(cases_to_run))

    results = []
    with ThreadPoolExecutor(max_workers=PARALLEL_CASES) as executor:
        futures = {}
        for i, cd in enumerate(cases_to_run):
            args = (i, cd, len(cases_to_run), population, oversight_level)
            futures[executor.submit(process_case, args)] = i
        for future in as_completed(futures):
            try:
                results.append(future.result())
            except Exception as e:
                logger.error("Case CRASHED: %s", e)
    return results

# ─── Run Treatment only (Control = main exp 50-case subset as baseline) ──
# Control (oversight=0.0) skipped: Pearson r=0.855 with main exp on shared cases,
# we use main exp data as baseline directly. Saves ~$17 and 1.5 days.
start_time = time.time()
treatment_results = run_condition(1.0)
elapsed = time.time() - start_time

logger.info("=" * 70)
logger.info("ABLATION COMPLETE: Treatment %d cases, %.1f hours",
            len(treatment_results), elapsed / 3600)
logger.info("Output: %s", OUTPUT_DIR)
logger.info("Baseline: outputs/exp1_full_100/per_case (50-case subset)")
logger.info("Next: python scripts/analyze_ablation_oversight.py")
logger.info("=" * 70)
