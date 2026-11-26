#!/usr/bin/env python3
"""Compare two LLM-as-judge coherence scorings of the same traces.

Reproduces the judge-comparison figures in Section 5.3 and Appendix D:
mean totals, score distributions, agreement on the 4-or-5 coherence
threshold, the direction of disagreement, the share of traces on which the
second judge gives an equal or lower total, the Pearson correlation of the
two totals, per-item pass rates and agreement, and per-archetype means.

Totals are the sum of the five item scores for both judges.

Usage:
    python scripts/compare_coherence_judges.py \
        --a results/coherence_scores --b results/coherence_scores_gpt56
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

ITEMS = [f"item{i}" for i in range(1, 6)]
ITEM_NAMES = {
    "item1": "Statute-appropriate framework",
    "item2": "Evidence-linked reasoning",
    "item3": "Stage consistency",
    "item4": "Dimension-consistent reasoning",
    "item5": "Non-arbitrary final decision",
}


def load(root: Path) -> dict[tuple[str, str], list[int]]:
    out = {}
    for f in root.glob("*/*.json"):
        d = json.loads(f.read_text())
        out[(f.parent.name, f.stem)] = [int(d[k]["score"]) for k in ITEMS]
    return out


def pearson(x: list[int], y: list[int]) -> float:
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(x, y))
    vx = sum((a - mx) ** 2 for a in x)
    vy = sum((b - my) ** 2 for b in y)
    return cov / math.sqrt(vx * vy)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="results/coherence_scores", help="first judge (Sonnet 4.6)")
    ap.add_argument("--b", default="results/coherence_scores_gpt56", help="second judge (GPT-5.6 Sol)")
    ap.add_argument("--name-a", default="Sonnet 4.6")
    ap.add_argument("--name-b", default="GPT-5.6 Sol")
    ap.add_argument("--threshold", type=int, default=4, help="total at or above which a trace counts as coherent")
    args = ap.parse_args()

    A, B = load(Path(args.a)), load(Path(args.b))
    keys = sorted(set(A) & set(B))
    if len(keys) != len(A) or len(keys) != len(B):
        print(f"warning: {len(A)} vs {len(B)} traces, {len(keys)} shared")
    n = len(keys)
    ta = [sum(A[k]) for k in keys]
    tb = [sum(B[k]) for k in keys]
    thr = args.threshold

    print(f"n = {n} shared traces\n")
    print(f"{'':28s}{args.name_a:>14s}{args.name_b:>14s}")
    print(f"{'mean total':28s}{sum(ta)/n:14.3f}{sum(tb)/n:14.3f}")
    print(f"{f'pct total >= {thr}':28s}{100*sum(t>=thr for t in ta)/n:13.1f}%{100*sum(t>=thr for t in tb)/n:13.1f}%")
    print(f"{'pct total == 5':28s}{100*sum(t==5 for t in ta)/n:13.1f}%{100*sum(t==5 for t in tb)/n:13.1f}%")
    da, db = Counter(ta), Counter(tb)
    for s in range(5, -1, -1):
        print(f"{f'  total = {s}':28s}{da[s]:8d} ({100*da[s]/n:4.1f}%){db[s]:8d} ({100*db[s]/n:4.1f}%)")

    print("\nper-item pass rate and agreement")
    for i, k in enumerate(ITEMS):
        pa = sum(A[key][i] for key in keys)
        pb = sum(B[key][i] for key in keys)
        ag = sum(A[key][i] == B[key][i] for key in keys)
        print(f"  {k} {ITEM_NAMES[k]:32s}{100*pa/n:6.1f}%{100*pb/n:8.1f}%   agree {100*ag/n:5.1f}%")

    both_pass = sum(a >= thr and b >= thr for a, b in zip(ta, tb))
    both_fail = sum(a < thr and b < thr for a, b in zip(ta, tb))
    a_pass_b_fail = sum(a >= thr and b < thr for a, b in zip(ta, tb))
    b_pass_a_fail = sum(a < thr and b >= thr for a, b in zip(ta, tb))
    print(f"\nthreshold agreement ({thr}-or-5): {100*(both_pass+both_fail)/n:.1f}%")
    print(f"  both pass {both_pass}, both fail {both_fail}")
    print(f"  {args.name_a} pass / {args.name_b} fail: {a_pass_b_fail}")
    print(f"  {args.name_b} pass / {args.name_a} fail: {b_pass_a_fail}")
    print(f"{args.name_b} total <= {args.name_a} total: {100*sum(b<=a for a,b in zip(ta,tb))/n:.1f}%")
    print(f"Pearson r of totals: {pearson(ta, tb):.3f}")
    delta = Counter(b - a for a, b in zip(ta, tb))
    print("delta (second minus first):", dict(sorted(delta.items())))

    arch = defaultdict(lambda: [[], []])
    for k in keys:
        name = k[1].rsplit("_", 1)[0]
        arch[name][0].append(sum(A[k]))
        arch[name][1].append(sum(B[k]))
    print("\nper-archetype mean total")
    for name, (xa, xb) in sorted(arch.items()):
        print(f"  {name:22s} n={len(xa):4d}  {sum(xa)/len(xa):.2f}  {sum(xb)/len(xb):.2f}")


if __name__ == "__main__":
    main()
