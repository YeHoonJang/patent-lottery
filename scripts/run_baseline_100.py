#!/usr/bin/env python3
"""
100-case vanilla baseline for §5.2 / App B re-calibration.

Setup:
- 14 examiners, all dimensions 0.50, noise σ=0.05 (vanilla_baseline.yaml)
- 100 cases from eval_100_enriched_input_v3.json (same cohort as main exp)
- Qwen3-235B-A22B-2507 (same backbone as main exp)
- ThreadPoolExecutor parallel = 8

Output: outputs/baselines/vanilla_100/
Cost estimate: ~$22 (10x the 10-case pilot at $2.22)
Resume-safe: skips cases with complete per_case JSON.
"""
import sys, json, logging, os, time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

REPO = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO)

if not os.environ.get("OPENROUTER_API_KEY"):
    raise SystemExit("ERROR: OPENROUTER_API_KEY not set. Run: set -a; source .env; set +a")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{REPO}/outputs/baseline_100.log"),
    ]
)
logger = logging.getLogger("baseline_100")

from src.utils.enriched_loader import dict_to_case_enriched
from src.examiner.population import load_archetype, instantiate_examiner
from src.pipeline.run_case import run_single_examiner
from src.utils.llm_adapter import LLMAdapter

MODEL = "qwen/qwen3-235b-a22b-2507"
TIMEOUT = 300
PARALLEL = 24
BASE_DIR = Path(f"{REPO}/outputs/baselines/vanilla_100")

# ─── Load 100 cases (ground_truth is inline in eval_100_enriched) ─
with open(f"{REPO}/data/eval_100_enriched_input_v3.json") as f:
    enriched = json.load(f)
cases_data = enriched["cases"]
logger.info("Loaded %d cases for baseline", len(cases_data))

# GT inline
gt_by_pn = {str(c["proceeding_number"]): c.get("ground_truth", {}) for c in cases_data}

# ─── Helper ──────────────────────────────────────────────────
def extract_issues(trace):
    issues = set()
    oa = trace.office_action
    if isinstance(oa, dict):
        for g in oa.get("rejection_grounds", []):
            b = g.get("basis", "")
            for code in ["101", "102", "103", "112"]:
                if code in b:
                    issues.add(code)
    return issues

def run_one_examiner(args):
    examiner, case_dict, pn, trace_base = args
    case = dict_to_case_enriched(case_dict)
    llm = LLMAdapter(mode="deepseek", model=MODEL, timeout=TIMEOUT)
    trace_dir = str(trace_base / pn / examiner.display_name)
    try:
        trace = run_single_examiner(case=case, examiner=examiner, llm=llm, output_dir=trace_dir)
        fd = trace.final_decision
        dec = fd.get("decision", "?") if isinstance(fd, dict) else "?"
        score = fd.get("decision_score", 0) if isinstance(fd, dict) else 0
        pred = extract_issues(trace)
    except Exception as e:
        logger.error("[%s/%s] FAILED: %s", pn, examiner.display_name, e)
        dec = "error"
        score = 0
        pred = set()

    return {
        "examiner": examiner.display_name,
        "archetype": examiner.archetype_name,
        "strictness": round(examiner.strictness, 3),
        "combination_threshold": round(examiner.combination_threshold, 3),
        "seniority": round(examiner.seniority, 3),
        "tech_specialization": round(examiner.tech_specialization, 3),
        "pred": sorted(pred),
        "decision": dec,
        "decision_score": score,
        "pn": pn,
    }

def run_experiment(name, population, output_dir):
    output_dir.mkdir(parents=True, exist_ok=True)
    per_case_dir = output_dir / "per_case"
    per_case_dir.mkdir(exist_ok=True)
    traces_dir = output_dir / "traces"
    traces_dir.mkdir(exist_ok=True)

    # Resume support
    completed_pns = set()
    for fp in per_case_dir.glob("*.json"):
        try:
            with open(fp) as fh:
                r = json.load(fh)
            if r and len(r.get("examiner_results", [])) == len(population):
                completed_pns.add(r["proceeding_number"])
        except Exception:
            pass

    remaining = [c for c in cases_data if c["proceeding_number"] not in completed_pns]
    logger.info("[%s] %d cases to run (%d already done), %d examiners, parallel=%d",
                name, len(remaining), len(completed_pns), len(population), PARALLEL)

    all_tasks = []
    for cd in remaining:
        pn = cd["proceeding_number"]
        for examiner in population:
            all_tasks.append((examiner, cd, pn, traces_dir))

    logger.info("[%s] Total tasks: %d", name, len(all_tasks))

    results_by_pn = {}
    with ThreadPoolExecutor(max_workers=PARALLEL) as executor:
        futures = {executor.submit(run_one_examiner, task): task for task in all_tasks}
        done_count = 0
        for future in as_completed(futures):
            done_count += 1
            try:
                result = future.result()
                pn = result.pop("pn")
                results_by_pn.setdefault(pn, []).append(result)

                if done_count % 14 == 0:
                    logger.info("[%s] %d/%d examiner-case pairs done", name, done_count, len(all_tasks))

                if len(results_by_pn[pn]) == len(population):
                    gt = gt_by_pn.get(pn, {})
                    grant_count = sum(1 for r in results_by_pn[pn] if r["decision"] == "grant")
                    amend_count = sum(1 for r in results_by_pn[pn] if r["decision"] == "amend_suggested")
                    case_result = {
                        "case_id": "BASELINE-%s-%s" % (name, pn),
                        "proceeding_number": pn,
                        "baseline_type": name,
                        "gt_issues": gt.get("issue_type", []),
                        "gt_outcome": gt.get("subdecision_coarse", ""),
                        "grant_rate": grant_count / len(population),
                        "amend_rate": amend_count / len(population),
                        "examiner_results": results_by_pn[pn],
                    }
                    with open(per_case_dir / ("%s.json" % pn), "w") as f:
                        json.dump(case_result, f, indent=2, ensure_ascii=False)
                    logger.info("[%s] Case %s SAVED — grant=%d amend=%d reject=%d",
                               name, pn, grant_count, amend_count,
                               sum(1 for r in results_by_pn[pn] if r["decision"] == "reject"))
            except Exception as e:
                logger.error("[%s] Task failed: %s", name, e)

    logger.info("[%s] COMPLETE!", name)

# ─── Build vanilla population ─────────────────────────────────
logger.info("=" * 60)
logger.info("VANILLA BASELINE — 100 cases × 14 examiners, parallel=%d", PARALLEL)
logger.info("=" * 60)

vanilla_arch = load_archetype(f"{REPO}/configs/baselines/vanilla_baseline.yaml")
vanilla_pop = []
for i in range(14):
    ex = instantiate_examiner(vanilla_arch, noise_seed=100 + i)
    vanilla_pop.append(ex)
logger.info("Vanilla population: %d examiners", len(vanilla_pop))
for ex in vanilla_pop:
    logger.info("  %s: strict=%.3f comb=%.3f sen=%.3f tech=%.3f",
                ex.display_name, ex.strictness, ex.combination_threshold,
                ex.seniority, ex.tech_specialization)

run_experiment("vanilla_100", vanilla_pop, BASE_DIR)

logger.info("=" * 60)
logger.info("BASELINE COMPLETE — outputs/baselines/vanilla_100/per_case/")
logger.info("=" * 60)
