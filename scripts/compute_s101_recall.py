#!/usr/bin/env python3
"""Recompute the §101 recall figures reported in the Limitations section.

Inputs (repo-relative by default):
  results/main_exp_100/per_case/*.json   per-case decisions (field `pred` = verified rejection grounds)
  data/eval_100_enriched_input_v3.json   technology center per case

Outputs: case-level recall over cases whose PTAB decision involved §101,
TC 3600 recall, examiner-level recall, and total §101 flags over all decisions.
"""
import argparse, glob, json, os
from collections import Counter

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-case", default=os.path.join(root, "results", "main_exp_100", "per_case"))
    ap.add_argument("--cohort", default=os.path.join(root, "data", "eval_100_enriched_input_v3.json"))
    a = ap.parse_args()

    cohort = json.load(open(a.cohort))
    cohort = cohort if isinstance(cohort, list) else cohort["cases"]
    tc = {str(c["proceeding_number"]): str(c.get("tech_center", ""))[:4] for c in cohort}

    gt101_cases = 0; hit_cases = 0; tc3600_cases = 0; tc3600_hits = 0
    gt101_decisions = 0; gt101_decision_hits = 0; total_decisions = 0; total_flags = 0
    hit_list = []
    for f in sorted(glob.glob(os.path.join(a.per_case, "*.json"))):
        d = json.load(open(f))
        pn = str(d.get("proceeding_number") or d.get("case_id"))
        flags = [any(str(g).startswith("101") for g in (ex.get("pred") or [])) for ex in d["examiner_results"]]
        total_decisions += len(flags); total_flags += sum(flags)
        if any(str(i).startswith("101") for i in d.get("gt_issues", [])):
            gt101_cases += 1; gt101_decisions += len(flags); gt101_decision_hits += sum(flags)
            if any(flags): hit_cases += 1; hit_list.append((pn, tc.get(pn)))
            if tc.get(pn) == "3600":
                tc3600_cases += 1; tc3600_hits += int(any(flags))
    print(f"GT-§101 cases: {gt101_cases}; cases where >=1 examiner raised §101: {hit_cases} "
          f"(recall {hit_cases/gt101_cases:.1%}) {hit_list}")
    print(f"TC 3600 GT-§101 cases: {tc3600_cases}; hits: {tc3600_hits}")
    print(f"Examiner-level on GT-§101 cases: {gt101_decision_hits}/{gt101_decisions}")
    print(f"All decisions: {total_decisions}; §101 flags: {total_flags}")

if __name__ == "__main__":
    main()
