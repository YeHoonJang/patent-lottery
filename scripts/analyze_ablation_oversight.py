#!/usr/bin/env python3
"""
Analyze ablation oversight results.

Produces:
  - summary.csv:          archetype x condition x grant/amend/reject rates
  - mcnemar_results.json: paired McNemar's test on grant decisions
  - section_breakdown.csv: §102/§103/§112 citation rate change
  - ptab_alignment.csv:   Reversed-agreement preservation check
"""
import json, csv, sys
from pathlib import Path
from collections import defaultdict, Counter
from statistics import mean

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "outputs/ablation_oversight"
PER_CASE = OUT / "per_case"

# ─── Load ─────────────────────────────────────────
control = []   # oversight=0
treatment = [] # oversight=1
for f in sorted(PER_CASE.glob("*_oversight0.json")):
    with open(f) as fh:
        control.append(json.load(fh))
for f in sorted(PER_CASE.glob("*_oversight1.json")):
    with open(f) as fh:
        treatment.append(json.load(fh))

print(f"Loaded: control={len(control)}, treatment={len(treatment)}")

# Index by (pn, archetype, instance) for paired analysis
def index_by_key(case_list):
    """Returns: {(pn, examiner_name): result_dict}"""
    idx = {}
    for case in case_list:
        pn = case["proceeding_number"]
        for er in case["examiner_results"]:
            idx[(pn, er["examiner"])] = {
                "decision": er["decision"],
                "archetype": er["archetype"],
                "weighing": er.get("weighing", []),
                "pred": er.get("pred", []),
                "gt_outcome": case["gt_outcome"],
            }
    return idx

ctrl_idx = index_by_key(control)
trt_idx = index_by_key(treatment)
shared_keys = sorted(set(ctrl_idx) & set(trt_idx))
print(f"Shared (pn, examiner) pairs: {len(shared_keys)}")

# ─── Summary table ────────────────────────────────
def rate_table(case_list, condition_label):
    arch_counts = defaultdict(Counter)
    for case in case_list:
        for er in case["examiner_results"]:
            arch_counts[er["archetype"]][er["decision"]] += 1
    rows = []
    for arch, c in arch_counts.items():
        total = sum(c.values())
        rows.append({
            "archetype": arch,
            "condition": condition_label,
            "n": total,
            "grant_rate": round(c.get("grant", 0) / total, 3) if total else 0,
            "amend_rate": round(c.get("amend_suggested", 0) / total, 3) if total else 0,
            "reject_rate": round(c.get("reject", 0) / total, 3) if total else 0,
            "error": c.get("error", 0),
        })
    return rows

summary_rows = rate_table(control, "control_oversight0") + rate_table(treatment, "treatment_oversight1")
with open(OUT / "summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["archetype", "condition", "n", "grant_rate", "amend_rate", "reject_rate", "error"])
    w.writeheader()
    w.writerows(summary_rows)
print(f"Wrote {OUT / 'summary.csv'}")

# Print summary
print("\n=== Grant rates by archetype × condition ===")
for arch in sorted(set(r["archetype"] for r in summary_rows)):
    ctrl = next((r for r in summary_rows if r["archetype"] == arch and "control" in r["condition"]), None)
    trt = next((r for r in summary_rows if r["archetype"] == arch and "treatment" in r["condition"]), None)
    if ctrl and trt:
        delta = trt["grant_rate"] - ctrl["grant_rate"]
        print(f"  {arch:25s}  ctrl={ctrl['grant_rate']:.3f}  trt={trt['grant_rate']:.3f}  Δ={delta:+.3f}")

# ─── McNemar's test (paired binary: grant vs not-grant) ─────
def is_grant(d): return 1 if d == "grant" else 0

mcnemar_results = {}
by_arch = defaultdict(list)
for k in shared_keys:
    by_arch[ctrl_idx[k]["archetype"]].append(k)

for arch, keys in by_arch.items():
    b = c_count = a_count = d_count = 0  # b: ctrl_grant & trt_notgrant; c: ctrl_notgrant & trt_grant
    for k in keys:
        cg = is_grant(ctrl_idx[k]["decision"])
        tg = is_grant(trt_idx[k]["decision"])
        if cg == 1 and tg == 1: a_count += 1
        elif cg == 1 and tg == 0: b += 1
        elif cg == 0 and tg == 1: c_count += 1
        else: d_count += 1
    # McNemar's exact test approximation (chi-square with continuity correction)
    if (b + c_count) > 0:
        chi2 = (abs(b - c_count) - 1) ** 2 / (b + c_count)
    else:
        chi2 = 0.0
    # p-value approx via chi-square 1 df survival fn
    try:
        from math import erfc, sqrt
        p_value = erfc(sqrt(chi2 / 2))
    except Exception:
        p_value = None
    mcnemar_results[arch] = {
        "n_pairs": len(keys),
        "ctrl_grant_trt_grant": a_count,
        "ctrl_grant_trt_notgrant": b,
        "ctrl_notgrant_trt_grant": c_count,
        "ctrl_notgrant_trt_notgrant": d_count,
        "chi2_continuity_corrected": round(chi2, 4),
        "p_value_approx": round(p_value, 6) if p_value is not None else None,
        "grant_rate_change": round((a_count + c_count - a_count - b) / len(keys), 4) if keys else 0,
    }

with open(OUT / "mcnemar_results.json", "w") as f:
    json.dump(mcnemar_results, f, indent=2)
print(f"\nWrote {OUT / 'mcnemar_results.json'}")
for arch, r in mcnemar_results.items():
    print(f"  {arch:25s}  χ²={r['chi2_continuity_corrected']:.3f}  p≈{r['p_value_approx']}")

# ─── Section breakdown (§102/§103/§112 reject citation rate) ──
def section_cite_rate(case_list, sections=("102", "103", "112")):
    out = defaultdict(lambda: defaultdict(int))
    totals = defaultdict(int)
    for case in case_list:
        for er in case["examiner_results"]:
            arch = er["archetype"]
            totals[arch] += 1
            for s in sections:
                if s in er.get("pred", []):
                    out[arch][s] += 1
    rates = {}
    for arch, t in totals.items():
        rates[arch] = {s: round(out[arch].get(s, 0) / t, 3) if t else 0 for s in sections}
    return rates

ctrl_sec = section_cite_rate(control)
trt_sec = section_cite_rate(treatment)
sec_rows = []
for arch in sorted(set(ctrl_sec) | set(trt_sec)):
    for s in ("102", "103", "112"):
        c = ctrl_sec.get(arch, {}).get(s, 0)
        t = trt_sec.get(arch, {}).get(s, 0)
        sec_rows.append({
            "archetype": arch,
            "section": f"§{s}",
            "control_rate": c,
            "treatment_rate": t,
            "delta": round(t - c, 3),
        })

with open(OUT / "section_breakdown.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["archetype", "section", "control_rate", "treatment_rate", "delta"])
    w.writeheader()
    w.writerows(sec_rows)
print(f"Wrote {OUT / 'section_breakdown.csv'}")

# ─── PTAB alignment: Reversed-case agreement preservation ──
def reversed_agreement(case_list):
    """For Reversed cases, fraction of examiner decisions that are NOT reject (=agree with PTAB reverse)."""
    out = defaultdict(lambda: {"n": 0, "agree": 0})
    for case in case_list:
        if case["gt_outcome"] != "Reversed":
            continue
        for er in case["examiner_results"]:
            arch = er["archetype"]
            out[arch]["n"] += 1
            if er["decision"] in ("grant", "amend_suggested"):
                out[arch]["agree"] += 1
    return {a: {"n": v["n"], "agree_rate": round(v["agree"] / v["n"], 3) if v["n"] else 0} for a, v in out.items()}

ctrl_ptab = reversed_agreement(control)
trt_ptab = reversed_agreement(treatment)
ptab_rows = []
for arch in sorted(set(ctrl_ptab) | set(trt_ptab)):
    c = ctrl_ptab.get(arch, {"n": 0, "agree_rate": 0})
    t = trt_ptab.get(arch, {"n": 0, "agree_rate": 0})
    ptab_rows.append({
        "archetype": arch,
        "n_reversed_cases": c["n"],
        "control_agree": c["agree_rate"],
        "treatment_agree": t["agree_rate"],
        "delta": round(t["agree_rate"] - c["agree_rate"], 3),
    })

with open(OUT / "ptab_alignment.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["archetype", "n_reversed_cases", "control_agree", "treatment_agree", "delta"])
    w.writeheader()
    w.writerows(ptab_rows)
print(f"Wrote {OUT / 'ptab_alignment.csv'}")

print("\n=== PTAB Reversed agreement preservation ===")
for r in ptab_rows:
    print(f"  {r['archetype']:25s}  ctrl={r['control_agree']:.3f}  trt={r['treatment_agree']:.3f}  Δ={r['delta']:+.3f}")
