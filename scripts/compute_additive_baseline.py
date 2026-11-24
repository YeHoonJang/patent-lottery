#!/usr/bin/env python3
"""Additive-baseline prediction for the composite archetypes (Appendix B).

Slope per dimension = (observed favorable rate of the orthogonal archetype - neutral baseline)
                      / (nominal extreme value - 0.50).
Prediction for a composite = baseline + sum_d slope_d * (x_d - 0.50), with the nominal
archetype values from configs/archetypes_v3.
Inputs: results/main_exp_100/per_case (main experiment), results/baseline_100/per_case
(14 neutral examiners, all dimensions 0.50).
"""
import argparse, glob, json, os
from collections import Counter

FAV = {"grant", "amend_suggested"}
PREFIX = {"C2": "lenient_advocate", "O4": "senior_examiner", "O3": "easy_combiner",
          "O2": "broad_interpreter", "O5": "domain_specialist", "O1": "strict_gatekeeper", "C1": "veteran_strict"}

def favorable_by_archetype(per_case_dir):
    counts = {a: Counter() for a in PREFIX}
    for f in glob.glob(os.path.join(per_case_dir, "*.json")):
        for r in json.load(open(f))["examiner_results"]:
            for code, pre in PREFIX.items():
                if r["examiner"].startswith(pre):
                    counts[code][r["decision"] in FAV] += 1
    return {a: 100 * c[True] / (c[True] + c[False]) for a, c in counts.items() if c}

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--main", default=os.path.join(root, "results", "main_exp_100", "per_case"))
    ap.add_argument("--baseline", default=os.path.join(root, "results", "baseline_100", "per_case"))
    a = ap.parse_args()

    main_fav = favorable_by_archetype(a.main)
    base_counts = Counter()
    for f in glob.glob(os.path.join(a.baseline, "*.json")):
        for r in json.load(open(f))["examiner_results"]:
            base_counts[r["decision"] in FAV] += 1
    base = 100 * base_counts[True] / (base_counts[True] + base_counts[False])
    print(f"Neutral baseline favorable rate: {base:.2f}%  ({base_counts[True] + base_counts[False]} decisions)")

    # (nominal extreme value, orthogonal archetype) per dimension, order: strictness, breadth, combination, seniority, specialization
    dims = [("strictness", 0.85, "O1"), ("claim_breadth", 0.80, "O2"), ("combination_threshold", 0.25, "O3"),
            ("seniority", 0.85, "O4"), ("tech_specialization", 0.90, "O5")]
    slopes = {}
    for name, x, code in dims:
        slopes[name] = (main_fav[code] - base) / (x - 0.50)
        print(f"  {name:22s} {code} observed {main_fav[code]:5.1f}%  effect {main_fav[code] - base:+6.2f} pp")
    composites = {"C2": [0.30, 0.35, 0.70, 0.40, 0.60], "C1": [0.80, 0.70, 0.30, 0.85, 0.70]}
    for code, vals in composites.items():
        pred = base + sum(slopes[n] * (v - 0.50) for (n, _, _), v in zip(dims, vals))
        print(f"{code}: additive prediction {pred:.2f}%  observed {main_fav[code]:.2f}%  gap {main_fav[code] - pred:+.2f} pp")

if __name__ == "__main__":
    main()
