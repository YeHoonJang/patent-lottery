#!/usr/bin/env python3
"""
Merge pilot_10_v3 + exp1_main into a unified 100-case dataset.

Inputs:
  outputs/pilot_10_v3/results.json     (10 cases as list)
  outputs/exp1_main/per_case/*.json    (90 cases, one per file)

Output:
  outputs/exp1_full_100/per_case/*.json  (100 cases, one per file)

Normalisation:
  - pilot's case_id has a "PILOT-" prefix → strip to bare proceeding_number
  - all files saved as <proceeding_number>.json

Run Q4 permutation test afterwards on the merged set.
"""
import json
import shutil
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
PILOT_RESULTS = REPO_ROOT / "outputs/pilot_10_v3/results.json"
EXP1_PER_CASE = REPO_ROOT / "outputs/exp1_main/per_case"
OUT_DIR = REPO_ROOT / "outputs/exp1_full_100/per_case"

def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # ── Copy 90 from exp1_main ─────────────────────────
    copied = 0
    for src in sorted(EXP1_PER_CASE.glob("*.json")):
        dst = OUT_DIR / src.name
        shutil.copy2(src, dst)
        copied += 1
    print(f"Copied {copied} cases from exp1_main")

    # ── Split 10 from pilot_10_v3/results.json ─────────
    pilot = json.loads(PILOT_RESULTS.read_text())
    added = 0
    for case in pilot:
        # Strip "PILOT-" prefix from case_id
        if case["case_id"].startswith("PILOT-"):
            case["case_id"] = case["case_id"][len("PILOT-"):]
        pn = case["proceeding_number"]
        out_path = OUT_DIR / f"{pn}.json"
        if out_path.exists():
            print(f"  WARN: overlap on {pn}, skipping pilot version")
            continue
        out_path.write_text(json.dumps(case, indent=2, ensure_ascii=False))
        added += 1
    print(f"Added {added} cases from pilot_10_v3")

    # ── Verify ─────────────────────────────────────────
    total_files = len(list(OUT_DIR.glob("*.json")))
    print(f"\nTotal cases in {OUT_DIR}: {total_files}")
    assert total_files == 100, f"Expected 100, got {total_files}"

    # Decision count
    total_decisions = 0
    for f in OUT_DIR.glob("*.json"):
        d = json.loads(f.read_text())
        total_decisions += len(d.get("examiner_results", []))
    print(f"Total examiner decisions: {total_decisions}")
    assert total_decisions == 1400, f"Expected 1400, got {total_decisions}"
    print("\n[OK] Merge complete: 100 cases, 1,400 decisions.")

if __name__ == "__main__":
    main()
