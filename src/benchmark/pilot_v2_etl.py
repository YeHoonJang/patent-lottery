"""
PILOT-Bench v2 ETL — integrates prior art mapping data from the companion corpus.

Key change: each case now has ACTUAL cited prior art patent numbers.
Combined with our existing BQ-enriched data (claims + abstract), we can provide
the system with the exact same prior art the examiner used.

Task redefinition:
  Before: "look at claims, guess what to reject with"
  After:  "given these claims AND these prior art docs, is the rejection valid?"
"""

import csv
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("pilot_v2")

BASE = str(Path(__file__).resolve().parents[2] / "data" / "pilot_bench_v2")


def load_prior_art_mapping():
    """Load cited prior art for 102/103 cases."""
    mapping = {}
    with open(f"{BASE}/prior_art_mapping/cited_prior_art_102_103_summary.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            proc = row["proceedingNumber"]
            mapping[proc] = {
                "proceedingNumber": proc,
                "applicant_patent": row["applicant_patent"],
                "prior_art_patents": row["prior_art_patents"].split("|") if row["prior_art_patents"] else [],
                "prior_art_count": int(row["prior_art_count"]) if row["prior_art_count"] else 0,
                "issue_type": row["issueType_label"],
                "subdecision": row["subdecisionTypeCoarse_label"],
                "art_unit": row["respondentGroupArtUnitNumber"],
                "tech_center": row["respondentTechnologyCenterNumber"],
                "appellant_patent_number": row.get("appellantPatentNumber", ""),
            }
    logger.info("Loaded prior art mapping: %d cases", len(mapping))
    return mapping


def load_artunit_stats():
    """Load art unit and tech center statistics."""
    stats = {}

    # Art unit stats
    with open(f"{BASE}/artunit_statistics/art_unit_stats.csv") as f:
        reader = csv.DictReader(f)
        au_stats = list(reader)
    stats["art_units"] = au_stats
    logger.info("Art unit stats: %d units", len(au_stats))

    # Tech center stats
    with open(f"{BASE}/artunit_statistics/tech_center_stats.csv") as f:
        reader = csv.DictReader(f)
        stats["tech_centers"] = list(reader)

    # Issue type stats
    with open(f"{BASE}/artunit_statistics/issue_type_stats.csv") as f:
        reader = csv.DictReader(f)
        stats["issue_types"] = list(reader)

    # Cross stats
    with open(f"{BASE}/artunit_statistics/tech_center_x_issue_cross.csv") as f:
        reader = csv.DictReader(f)
        stats["cross"] = list(reader)

    return stats


def merge_with_pilot_bench(prior_art_mapping: dict, opinion_split_path: str = None):
    """
    Merge prior art mapping with existing PILOT-Bench opinion split data.
    Returns cases ready for benchmarking.
    """
    # Load existing PILOT-Bench benchmark data
    benchmark_path = str(Path(__file__).resolve().parents[2] / "data" / "pilot_bench" / "benchmark" / "benchmark_dataset.json")
    with open(benchmark_path) as f:
        existing = json.load(f)

    merged = []
    matched = 0

    for case in existing["cases"]:
        proc = str(case["proceeding_number"])
        pa_data = prior_art_mapping.get(proc)

        if pa_data:
            case["cited_prior_art"] = pa_data["prior_art_patents"]
            case["applicant_patent"] = pa_data["applicant_patent"]
            case["prior_art_count"] = pa_data["prior_art_count"]
            case["has_cited_prior_art"] = True
            matched += 1
        else:
            case["cited_prior_art"] = []
            case["applicant_patent"] = ""
            case["prior_art_count"] = 0
            case["has_cited_prior_art"] = False

        merged.append(case)

    logger.info("Merged: %d/%d cases have cited prior art", matched, len(merged))
    return merged


def build_v2_benchmark():
    """Build the v2 benchmark dataset with cited prior art."""
    # Load new data
    pa_mapping = load_prior_art_mapping()
    au_stats = load_artunit_stats()

    # Merge
    cases = merge_with_pilot_bench(pa_mapping)

    # Filter: only cases with cited prior art (102/103)
    cases_with_pa = [c for c in cases if c["has_cited_prior_art"]]

    # Stats
    issue_dist = {}
    decision_dist = {}
    pa_count_dist = {}
    for c in cases_with_pa:
        for it in c["ground_truth"]["issue_type"]:
            issue_dist[it] = issue_dist.get(it, 0) + 1
        sd = c["ground_truth"]["subdecision_coarse"]
        decision_dist[sd] = decision_dist.get(sd, 0) + 1
        pac = c["prior_art_count"]
        pa_count_dist[pac] = pa_count_dist.get(pac, 0) + 1

    stats = {
        "total_with_prior_art": len(cases_with_pa),
        "total_all": len(cases),
        "issue_distribution": dict(sorted(issue_dist.items(), key=lambda x: -x[1])),
        "decision_distribution": dict(sorted(decision_dist.items(), key=lambda x: -x[1])),
        "prior_art_count_distribution": dict(sorted(pa_count_dist.items())),
    }

    # Save
    output = {
        "stats": stats,
        "artunit_stats": au_stats,
        "cases": cases_with_pa,
    }

    output_path = f"{BASE}/benchmark_v2.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"PILOT-Bench v2 ETL Complete")
    print(f"{'='*60}")
    print(f"Total cases with cited prior art: {len(cases_with_pa)}")
    print(f"Total all cases: {len(cases)}")
    print(f"Avg prior art per case: {sum(c['prior_art_count'] for c in cases_with_pa)/max(len(cases_with_pa),1):.1f}")
    print(f"\nIssue distribution:")
    for k, v in sorted(issue_dist.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    print(f"\nDecision distribution:")
    for k, v in sorted(decision_dist.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")
    print(f"\nSaved to: {output_path}")

    return output


if __name__ == "__main__":
    build_v2_benchmark()
