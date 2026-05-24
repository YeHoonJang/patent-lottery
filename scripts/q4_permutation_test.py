#!/usr/bin/env python3
"""
Q4 — Permutation test: are between-archetype grant-rate differences
distinguishable from label-shuffled noise?

H0: archetype labels are exchangeable; observed between-archetype variance
    of grant rates is consistent with random labeling.
H1: archetype identity drives decisions; observed variance exceeds the
    null distribution.

Tests reported:
  (A) Omnibus: between-archetype variance of grant rates over all
      1,260 decisions, K=10,000 label shuffles.
  (B) Pairwise contrasts: difference in grant rates for selected
      archetype pairs (e.g., lenient_advocate vs strict_gatekeeper),
      same K shuffles.

Output: outputs/exp1_main/q4_permutation_results.json (+ console summary)
"""
import json, glob, random
from collections import defaultdict
from pathlib import Path
from statistics import mean, pvariance

REPO_ROOT = Path(__file__).resolve().parents[1]
PER_CASE = REPO_ROOT / "outputs/exp1_full_100/per_case"
OUT = REPO_ROOT / "outputs/exp1_full_100/q4_permutation_results.json"

K = 10_000
SEED = 42
PAIR_CONTRASTS = [
    ("lenient_advocate", "strict_gatekeeper"),
    ("lenient_advocate", "veteran_strict"),
    ("easy_combiner", "strict_gatekeeper"),
    ("broad_interpreter", "strict_gatekeeper"),
]

def load_decisions():
    """Returns list of (archetype, is_grant) tuples."""
    rows = []
    for f in sorted(glob.glob(str(PER_CASE / "*.json"))):
        with open(f) as fh:
            d = json.load(fh)
        for er in d.get("examiner_results", []):
            rows.append((er["archetype"], 1 if er["decision"] == "grant" else 0))
    return rows

def grant_rate_by_arch(rows, arch_labels):
    """Given per-row grant flags and a parallel arch label list, return {arch: rate}."""
    sums = defaultdict(int)
    cnts = defaultdict(int)
    for label, (_, g) in zip(arch_labels, rows):
        sums[label] += g
        cnts[label] += 1
    return {a: sums[a] / cnts[a] for a in cnts}

def between_arch_variance(rates):
    """Variance of per-archetype grant rates (population variance across archetypes)."""
    return pvariance(rates.values())

def main():
    rows = load_decisions()
    n = len(rows)
    arch_labels = [a for a, _ in rows]
    archetypes = sorted(set(arch_labels))
    print(f"Loaded {n} decisions across {len(archetypes)} archetypes.")

    # ── Observed statistics ──────────────────────────
    obs_rates = grant_rate_by_arch(rows, arch_labels)
    obs_var = between_arch_variance(obs_rates)
    print("\n=== Observed grant rates ===")
    for a in archetypes:
        print(f"  {a:25s}  {obs_rates[a]:.4f}")
    print(f"\nObserved between-archetype variance: {obs_var:.6f}")

    obs_pair_diff = {}
    for a, b in PAIR_CONTRASTS:
        if a in obs_rates and b in obs_rates:
            obs_pair_diff[(a, b)] = obs_rates[a] - obs_rates[b]

    # ── Permutation test ────────────────────────────
    rng = random.Random(SEED)
    shuffled = list(arch_labels)
    null_var = []
    null_pair_diff = {p: [] for p in obs_pair_diff}

    for k in range(K):
        rng.shuffle(shuffled)
        rates = grant_rate_by_arch(rows, shuffled)
        null_var.append(between_arch_variance(rates))
        for (a, b) in obs_pair_diff:
            null_pair_diff[(a, b)].append(rates[a] - rates[b])
        if (k + 1) % 2000 == 0:
            print(f"  ...{k+1}/{K} permutations")

    # ── p-values (one-sided for variance, two-sided for pair diffs) ─
    p_omnibus = (sum(1 for v in null_var if v >= obs_var) + 1) / (K + 1)
    pair_results = {}
    for (a, b), d_obs in obs_pair_diff.items():
        null = null_pair_diff[(a, b)]
        # two-sided
        p_two = (sum(1 for v in null if abs(v) >= abs(d_obs)) + 1) / (K + 1)
        pair_results[f"{a}__vs__{b}"] = {
            "obs_diff": round(d_obs, 4),
            "null_mean": round(mean(null), 4),
            "p_two_sided": round(p_two, 5),
        }

    print("\n=== Omnibus permutation test ===")
    print(f"  Observed between-arch variance : {obs_var:.6f}")
    print(f"  Null mean (K={K}) variance     : {mean(null_var):.6f}")
    print(f"  Null max variance              : {max(null_var):.6f}")
    print(f"  p-value (one-sided)            : {p_omnibus:.5f}")

    print("\n=== Pairwise contrasts ===")
    for k, v in pair_results.items():
        print(f"  {k:50s}  Δ={v['obs_diff']:+.4f}  p={v['p_two_sided']:.5f}")

    out = {
        "n_decisions": n,
        "archetypes": archetypes,
        "K_permutations": K,
        "seed": SEED,
        "observed_grant_rates": {a: round(r, 4) for a, r in obs_rates.items()},
        "omnibus": {
            "observed_between_arch_variance": round(obs_var, 6),
            "null_mean_variance": round(mean(null_var), 6),
            "null_max_variance": round(max(null_var), 6),
            "p_value_one_sided": round(p_omnibus, 5),
        },
        "pairwise": pair_results,
    }
    OUT.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT}")

if __name__ == "__main__":
    main()
