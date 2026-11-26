#!/usr/bin/env python3
"""Aggregate per-trace LLM-as-judge coherence scores into _aggregate.json.

Reads every ``<case>/<examiner>.json`` under the coherence-scores directory
and recomputes the summary statistics reported in the paper (Section 5.3 and
Appendix D).  The total for each trace is the sum of the five binary item
scores.  The judge also reports its own ``total`` field; in 8 of 1,400 traces
that field is one point below the item sum, and the paper uses the item sum.
Per-item pass rates are unaffected by this choice.

Fields of the existing _aggregate.json that are not derived from the item
scores (judge_model, trace_source, token_usage) are carried over unchanged.

Usage:
    python scripts/aggregate_coherence_scores.py [--dir results/coherence_scores]
    python scripts/aggregate_coherence_scores.py --dir results/coherence_scores_gpt56 \
        --judge-model gpt-5.6-sol --trace-source "outputs/exp1_full_100/traces (Qwen3-235B-A22B)"
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ITEMS = [f"item{i}" for i in range(1, 6)]
CARRY_OVER = ["judge_model", "trace_source", "token_usage"]


def classify(total: int) -> str:
    if total >= 4:
        return "legally_coherent"
    if total >= 2:
        return "partially_coherent"
    return "weak"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/coherence_scores")
    ap.add_argument("--dry-run", action="store_true", help="print the summary without writing")
    ap.add_argument("--judge-model", default=None, help="recorded as judge_model when no existing aggregate supplies one")
    ap.add_argument("--trace-source", default=None, help="recorded as trace_source when no existing aggregate supplies one")
    args = ap.parse_args()

    root = Path(args.dir)
    out_path = root / "_aggregate.json"
    files = sorted(p for p in root.glob("*/*.json"))
    if not files:
        raise SystemExit(f"no per-trace files under {root}")

    totals: list[int] = []
    item_pass = Counter()
    judge_total_mismatch = 0
    for f in files:
        d = json.loads(f.read_text())
        scores = [int(d[k]["score"]) for k in ITEMS]
        total = sum(scores)
        totals.append(total)
        for k, s in zip(ITEMS, scores):
            item_pass[k] += s
        if "total" in d and int(d["total"]) != total:
            judge_total_mismatch += 1

    n = len(totals)
    dist = Counter(totals)
    agg = {
        "n_traces": n,
        "mean_score": round(sum(totals) / n, 3),
        "pct_4_or_5_of_5": round(100 * sum(1 for t in totals if t >= 4) / n, 1),
        "pct_5_of_5": round(100 * sum(1 for t in totals if t == 5) / n, 1),
        "per_item_pass_rate": {k: round(100 * item_pass[k] / n, 1) for k in ITEMS},
        "score_distribution": {str(k): dist[k] for k in sorted(dist)},
        "classification": dict(Counter(classify(t) for t in totals)),
        "total_basis": "sum of the five item scores",
        "n_judge_total_mismatch": judge_total_mismatch,
    }
    if out_path.exists():
        prev = json.loads(out_path.read_text())
        for k in CARRY_OVER:
            if k in prev:
                agg[k] = prev[k]
    if args.judge_model and "judge_model" not in agg:
        agg["judge_model"] = args.judge_model
    if args.trace_source and "trace_source" not in agg:
        agg["trace_source"] = args.trace_source

    print(json.dumps(agg, indent=2))
    if not args.dry_run:
        out_path.write_text(json.dumps(agg, indent=2) + "\n")
        print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
