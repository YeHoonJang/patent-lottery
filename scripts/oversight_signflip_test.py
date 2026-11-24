#!/usr/bin/env python3
"""Paired sign-flip permutation test for the institutional-oversight experiment (Appendix I).

Unit of analysis: for each of the 50 shared cases and each archetype, the favorable fraction
(grant or amend) of the archetype's two examiner instances, in the control run
(results/main_exp_100) and in the oversight run (results/ablation_oversight).
Test statistic: mean paired difference (oversight - control). Null distribution: random sign
flips of the 50 paired differences. Two-sided p = share of flips with |mean| >= |observed|.
"""
import argparse, json, os
from collections import defaultdict
import numpy as np

FAV = {"grant", "amend_suggested"}
ORDER = ["broad_interpreter", "easy_combiner", "senior_examiner", "domain_specialist",
         "strict_gatekeeper", "veteran_strict", "lenient_advocate"]

def favorable_by_archetype(case):
    d = defaultdict(list)
    for er in case["examiner_results"]:
        d[er["archetype"]].append(1.0 if er["decision"] in FAV else 0.0)
    return {a: float(np.mean(v)) for a, v in d.items()}

def signflip_p(diffs, n_flips, seed):
    rng = np.random.default_rng(seed)
    d = np.asarray(diffs); obs = abs(d.mean()); count = 0
    for start in range(0, n_flips, 200_000):
        m = min(200_000, n_flips - start)
        signs = rng.choice([-1, 1], size=(m, len(d)))
        count += int((np.abs((signs * d).mean(axis=1)) >= obs - 1e-12).sum())
    return count / n_flips

def main():
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    ap = argparse.ArgumentParser()
    ap.add_argument("--subset", default=os.path.join(root, "data", "ablation_50_cases.json"))
    ap.add_argument("--control", default=os.path.join(root, "results", "main_exp_100", "per_case"))
    ap.add_argument("--treatment", default=os.path.join(root, "results", "ablation_oversight", "per_case"))
    ap.add_argument("--n-flips", type=int, default=1_000_000)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()

    pns = [str(c["proceeding_number"]) for c in json.load(open(a.subset))["cases"]]
    diffs = defaultdict(list); base = defaultdict(list); treat = defaultdict(list)
    for pn in pns:
        c = favorable_by_archetype(json.load(open(os.path.join(a.control, f"{pn}.json"))))
        t = favorable_by_archetype(json.load(open(os.path.join(a.treatment, f"{pn}_oversight1.json"))))
        for arch in c:
            base[arch].append(c[arch]); treat[arch].append(t[arch]); diffs[arch].append(t[arch] - c[arch])
    print(f"{len(pns)} shared cases; {a.n_flips:,} sign-flips; seed {a.seed}")
    print(f"{'archetype':20s} {'control':>8s} {'oversight':>10s} {'delta(pp)':>10s} {'p':>8s}")
    for arch in ORDER:
        b = 100 * np.mean(base[arch]); t = 100 * np.mean(treat[arch])
        print(f"{arch:20s} {b:8.1f} {t:10.1f} {t - b:+10.1f} {signflip_p(diffs[arch], a.n_flips, a.seed):8.4f}")

if __name__ == "__main__":
    main()
