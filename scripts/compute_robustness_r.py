#!/usr/bin/env python3
"""Compute Pearson r for paper §5.5 Robustness — matching the paper's method.

Method:
  - Baseline: per-archetype grant rate from FULL 100-case main experiment
  - Ablation model: per-archetype grant rate from its 50-case run
  - Pearson r computed over 7 archetypes
  - Also compute G+A (favorable rate) r

This is "loosely paired" (different cases between baseline and ablation), matching
how the paper currently reports it.
"""
import json
from pathlib import Path
from collections import defaultdict
from statistics import mean
import math

REPO = Path(__file__).resolve().parents[1]

BASELINE = REPO / "outputs/exp1_full_100/per_case"   # 100 cases
MODELS = {
    "Qwen3-8B":   REPO / "outputs/ablation_qwen3_8b_local/per_case",
    "Qwen3-32B":  REPO / "outputs/ablation_qwen3_32b/per_case",
    "Sonnet 4.5": REPO / "outputs/ablation_sonnet45/per_case",
    "DeepSeek-V3":REPO / "outputs/ablation_deepseek_v3/per_case",
    "Mistral-L":  REPO / "outputs/ablation_mistral_large/per_case",
}

def archetype_rates(per_case_dir):
    """Returns (grant_rate dict, GA_rate dict) per archetype, averaged across cases."""
    g, ga = defaultdict(list), defaultdict(list)
    for f in sorted(Path(per_case_dir).glob("*.json")):
        try:
            r = json.loads(f.read_text())
        except Exception:
            continue
        by_g, by_ga = defaultdict(list), defaultdict(list)
        for er in r.get("examiner_results", []):
            arch = er.get("archetype") or er.get("examiner", "").rsplit("_", 1)[0]
            dec = er.get("decision")
            if dec not in {"grant", "amend_suggested", "reject"}:
                continue
            by_g[arch].append(1 if dec == "grant" else 0)
            by_ga[arch].append(1 if dec in {"grant", "amend_suggested"} else 0)
        for arch, vals in by_g.items():
            if vals: g[arch].append(sum(vals) / len(vals))
        for arch, vals in by_ga.items():
            if vals: ga[arch].append(sum(vals) / len(vals))
    return ({a: mean(v) for a, v in g.items()},
            {a: mean(v) for a, v in ga.items()})

def pearson(xs, ys):
    n = len(xs)
    mx, my = mean(xs), mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx) ** 2 for x in xs))
    dy = math.sqrt(sum((y - my) ** 2 for y in ys))
    return num / (dx * dy) if dx and dy else float("nan")

baseline_g, baseline_ga = archetype_rates(BASELINE)
n_base = len(list(BASELINE.glob("*.json")))
print(f"Baseline (Qwen-235B, {n_base} cases):")
for a in sorted(baseline_g):
    print(f"  {a:24s}  grant={baseline_g[a]:.3f}  G+A={baseline_ga[a]:.3f}")

print(f"\n{'Model':<14s} {'n':>4} {'Grant r':>10} {'G+A r':>10}")
print("-" * 44)
for name, p in MODELS.items():
    n = len(list(p.glob("*.json")))
    m_g, m_ga = archetype_rates(p)
    archs = sorted(set(baseline_g) & set(m_g))
    xs_g = [baseline_g[a] for a in archs]
    ys_g = [m_g[a] for a in archs]
    xs_ga = [baseline_ga[a] for a in archs]
    ys_ga = [m_ga[a] for a in archs]
    r_g = pearson(xs_g, ys_g)
    r_ga = pearson(xs_ga, ys_ga)
    print(f"{name:<14s} {n:>4} {r_g:>10.3f} {r_ga:>10.3f}")
