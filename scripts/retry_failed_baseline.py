#!/usr/bin/env python3
"""
Retry §112 fails from baseline_100 with PARALLEL=8 (lower GPU pressure).

Strategy:
1. Scan baselines/vanilla_100/traces/ for (case, examiner) pairs missing
   05_written_desc_check.json (or empty).
2. Re-run pipeline for those specific pairs only (full pipeline, since
   §112 affects office_action and final_decision downstream).
3. Update per_case JSON with new decisions for those examiners.

Run AFTER run_baseline_100.py completes all 100 cases.
"""
import sys, json, os, time, glob
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import logging

REPO = str(Path(__file__).resolve().parents[1])
sys.path.insert(0, REPO)

if not os.environ.get("OPENROUTER_API_KEY"):
    raise SystemExit("ERROR: OPENROUTER_API_KEY not set")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"{REPO}/outputs/baseline_retry.log"),
    ]
)
logger = logging.getLogger("retry_baseline")

from src.utils.enriched_loader import dict_to_case_enriched
from src.examiner.population import load_archetype, instantiate_examiner
from src.pipeline.run_case import run_single_examiner
from src.utils.llm_adapter import LLMAdapter

MODEL = "qwen/qwen3-235b-a22b-2507"
TIMEOUT = 300
PARALLEL = 8   # Lower than 24 to avoid CUDA OOM on Step 5
BASE_DIR = Path(f"{REPO}/outputs/baselines/vanilla_100")
TRACES_DIR = BASE_DIR / "traces"
PER_CASE_DIR = BASE_DIR / "per_case"

# ─── Load case data ──────────────────────────────────────────
with open(f"{REPO}/data/eval_100_enriched_input_v3.json") as f:
    enriched = json.load(f)
cases_by_pn = {c["proceeding_number"]: c for c in enriched["cases"]}
gt_by_pn = {pn: c.get("ground_truth", {}) for pn, c in cases_by_pn.items()}

# ─── Build vanilla population (same as main run) ─────────────
vanilla_arch = load_archetype(f"{REPO}/configs/baselines/vanilla_baseline.yaml")
population = [instantiate_examiner(vanilla_arch, noise_seed=100 + i) for i in range(14)]
pop_by_name = {ex.display_name: ex for ex in population}

# ─── Scan for §112 fails ─────────────────────────────────────
fails = []
for case_dir in sorted(TRACES_DIR.glob("*")):
    if not case_dir.is_dir():
        continue
    pn = case_dir.name
    for ex_dir in sorted(case_dir.glob("*")):
        ex_name = ex_dir.name
        runs = sorted(ex_dir.glob("run_*"))
        if not runs:
            continue
        run_dir = runs[-1]   # latest run
        wd_file = run_dir / "05_written_desc_check.json"
        needs_retry = False
        if not wd_file.exists():
            needs_retry = True
        else:
            try:
                with open(wd_file) as f:
                    data = json.load(f)
                if not data or (isinstance(data, list) and len(data) == 0):
                    needs_retry = True
            except Exception:
                needs_retry = True
        if needs_retry:
            fails.append((pn, ex_name, run_dir))

logger.info("Found %d (case, examiner) pairs needing retry", len(fails))
if not fails:
    logger.info("Nothing to retry. Exiting.")
    sys.exit(0)

# ─── Remove failed run dirs (so run_single_examiner starts fresh) ──
for pn, ex_name, run_dir in fails:
    if run_dir.exists():
        import shutil
        shutil.rmtree(run_dir)
        logger.info("  Removed: %s", run_dir)

# ─── Retry function ──────────────────────────────────────────
def run_one(args):
    pn, ex_name = args
    examiner = pop_by_name[ex_name]
    case_dict = cases_by_pn[pn]
    case = dict_to_case_enriched(case_dict)
    llm = LLMAdapter(mode="deepseek", model=MODEL, timeout=TIMEOUT)
    trace_dir = str(TRACES_DIR / pn / ex_name)
    try:
        trace = run_single_examiner(case=case, examiner=examiner, llm=llm, output_dir=trace_dir)
        fd = trace.final_decision
        dec = fd.get("decision", "?") if isinstance(fd, dict) else "?"
        score = fd.get("decision_score", 0) if isinstance(fd, dict) else 0
        # Verify §112 succeeded this time
        wd = trace.written_desc_assessments
        wd_ok = wd and (isinstance(wd, list) and len(wd) > 0)
        return {
            "pn": pn,
            "examiner": ex_name,
            "archetype": examiner.archetype_name,
            "strictness": round(examiner.strictness, 3),
            "combination_threshold": round(examiner.combination_threshold, 3),
            "seniority": round(examiner.seniority, 3),
            "tech_specialization": round(examiner.tech_specialization, 3),
            "decision": dec,
            "decision_score": score,
            "wd_ok": wd_ok,
        }
    except Exception as e:
        logger.error("[%s/%s] RETRY FAILED: %s", pn, ex_name, e)
        return {"pn": pn, "examiner": ex_name, "decision": "error", "wd_ok": False}

# ─── Run retries in parallel ─────────────────────────────────
tasks = [(pn, ex_name) for pn, ex_name, _ in fails]
logger.info("Retrying %d pairs with PARALLEL=%d", len(tasks), PARALLEL)

results_by_pn = {}
with ThreadPoolExecutor(max_workers=PARALLEL) as executor:
    futures = {executor.submit(run_one, t): t for t in tasks}
    for fut in as_completed(futures):
        r = fut.result()
        pn = r["pn"]
        results_by_pn.setdefault(pn, []).append(r)
        logger.info("[%s/%s] decision=%s wd_ok=%s",
                    pn, r["examiner"], r["decision"], r["wd_ok"])

# ─── Update per_case JSONs ───────────────────────────────────
logger.info("Updating per_case JSONs...")
for pn, new_results in results_by_pn.items():
    pc_file = PER_CASE_DIR / f"{pn}.json"
    if not pc_file.exists():
        logger.warning("per_case missing for %s — skipping merge", pn)
        continue
    with open(pc_file) as f:
        pc = json.load(f)
    # Replace entries for retried examiners
    new_by_name = {r["examiner"]: r for r in new_results}
    updated = []
    for r in pc["examiner_results"]:
        if r["examiner"] in new_by_name:
            nr = new_by_name[r["examiner"]]
            # keep examiner profile fields, update decision-related fields
            r.update({
                "decision": nr["decision"],
                "decision_score": nr["decision_score"],
            })
        updated.append(r)
    # Recompute aggregate rates
    grant_count = sum(1 for r in updated if r["decision"] == "grant")
    amend_count = sum(1 for r in updated if r["decision"] == "amend_suggested")
    pc["examiner_results"] = updated
    pc["grant_rate"] = grant_count / len(updated)
    pc["amend_rate"] = amend_count / len(updated)
    with open(pc_file, "w") as f:
        json.dump(pc, f, indent=2, ensure_ascii=False)
    logger.info("  Updated %s — grant=%d amend=%d reject=%d",
                pn, grant_count, amend_count,
                sum(1 for r in updated if r["decision"] == "reject"))

# ─── Final fail check ────────────────────────────────────────
still_failed = sum(1 for results in results_by_pn.values() for r in results if not r.get("wd_ok"))
logger.info("=" * 60)
logger.info("RETRY COMPLETE — %d/%d still failing", still_failed, len(tasks))
logger.info("=" * 60)
