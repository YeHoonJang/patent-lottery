#!/usr/bin/env python3
"""
Stratified random sample of 50 cases from eval_100 for ablation experiments.

All ablation experiments (model robustness comparison + oversight ablation)
use the same 50 cases for fair comparison.

Stratification preserves the original outcome distribution:
  100 cases: 45 Affirmed / 35 Reversed / 15 AiP / 5 Other-NGOR
   50 cases: 23 Affirmed / 18 Reversed /  7 AiP / 2 Other-NGOR

Output: data/ablation_50_cases.json
"""
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE = REPO_ROOT / "data/eval_100_enriched_input_v3.json"
OUT = REPO_ROOT / "data/ablation_50_cases.json"

SEED = 42
QUOTAS = {
    "Affirmed": 23,
    "Reversed": 18,
    "Affirmed-in-Part": 7,
    "Other-NGOR": 2,
}

NGOR_VARIANTS = {
    "Reversed with New Ground of Rejection",
    "Affirmed-in-Part with New Ground of Rejection",
    "Affirmed with New Ground of Rejection",
}

def bucket_of(subdecision: str) -> str:
    if subdecision in NGOR_VARIANTS:
        return "Other-NGOR"
    return subdecision

def main():
    src = json.loads(SOURCE.read_text())
    all_cases = src["cases"]
    print(f"Loaded {len(all_cases)} source cases")

    # Group by bucket
    by_bucket = defaultdict(list)
    for c in all_cases:
        sub = c.get("ground_truth", {}).get("subdecision", "UNKNOWN")
        by_bucket[bucket_of(sub)].append(c)

    print("\n=== Source distribution ===")
    for b, lst in by_bucket.items():
        print(f"  {b:25s}  {len(lst)}")

    # Stratified sample
    rng = random.Random(SEED)
    selected = []
    for bucket, quota in QUOTAS.items():
        pool = by_bucket[bucket]
        if len(pool) < quota:
            raise ValueError(f"Not enough cases in {bucket}: {len(pool)} < {quota}")
        rng.shuffle(pool)
        selected.extend(pool[:quota])

    print(f"\n=== Sampled {len(selected)} cases (seed={SEED}) ===")
    sub_counts = Counter(bucket_of(c["ground_truth"]["subdecision"]) for c in selected)
    for b, q in QUOTAS.items():
        actual = sub_counts.get(b, 0)
        ok = "OK" if actual == q else "MISMATCH"
        print(f"  {b:25s}  {actual}/{q}  [{ok}]")

    # Tech Center distribution check
    tc_src = Counter(c.get("tech_center") for c in all_cases)
    tc_sample = Counter(c.get("tech_center") for c in selected)
    print("\n=== Tech Center distribution (source vs sample) ===")
    for tc in sorted(tc_src):
        s, p = tc_src[tc], tc_sample.get(tc, 0)
        print(f"  TC {tc}:  source {s:3d}  ->  sample {p:3d}  ({100*p/s:.0f}%)")

    # Average size sanity
    def avg(field):
        return sum(c.get(field, 0) for c in selected) / len(selected)
    def avg_src(field):
        return sum(c.get(field, 0) for c in all_cases) / len(all_cases)
    print("\n=== Avg case size (source vs sample) ===")
    for field in ("claims_count", "prior_art_count", "description_chars"):
        print(f"  {field:20s}  source {avg_src(field):8.1f}  ->  sample {avg(field):8.1f}")

    # Save
    out = {
        "total": len(selected),
        "seed": SEED,
        "quotas": QUOTAS,
        "source_file": str(SOURCE.relative_to(REPO_ROOT)),
        "note": ("Stratified random sample of 50 cases from eval_100 for ablation "
                 "experiments (model robustness + oversight). All ablations use "
                 "this exact same set."),
        "cases": selected,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT} ({OUT.stat().st_size:,} bytes)")
    print(f"Selected proceeding numbers (first 10): {[c['proceeding_number'] for c in selected[:10]]}")

if __name__ == "__main__":
    main()
