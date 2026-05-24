"""
PILOT-Bench ETL v2 — correctly handles opinion_split JSON files.

Data structure:
- data/ptab_metadata.json: dict keyed by index, each with case metadata + issue_type etc.
- data/opinion_split/<proceeding>_DECISION.json: {appellant_arguments, examiner_findings, ptab_opinion}
  Each field is a list of sentences.
"""

import json
import logging
import tarfile
from pathlib import Path

logger = logging.getLogger("vpatent.benchmark")


def load_pilot_bench(tar_path: str, max_records: int = None) -> list[dict]:
    """Load and merge PILOT-Bench metadata + opinion splits."""
    tar_path = Path(tar_path)
    if not tar_path.exists():
        raise FileNotFoundError(f"Not found: {tar_path}")

    metadata = {}
    opinion_splits = {}

    with tarfile.open(tar_path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile() or not member.name.endswith(".json"):
                continue
            # Skip cache/lock/metadata files
            if ".cache" in member.name or member.name.endswith(".lock") or member.name.endswith(".metadata"):
                continue

            f = tar.extractfile(member)
            if f is None:
                continue
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue

            if member.name.endswith("ptab_metadata.json"):
                metadata = data
                logger.info("Loaded metadata: %d entries", len(metadata))
            elif "opinion_split/" in member.name:
                # Extract proceeding number from filename
                fname = Path(member.name).stem  # e.g. "2013001169_DECISION"
                proc_num = fname.split("_")[0] if "_" in fname else fname
                # Also try "Decision_PROCNUM_date" format
                if fname.startswith("Decision_"):
                    parts = fname.split("_")
                    if len(parts) >= 2:
                        proc_num = parts[1]
                opinion_splits[proc_num] = data

    logger.info("Loaded %d opinion splits, %d metadata entries",
                len(opinion_splits), len(metadata))

    # Build proceeding_number -> metadata lookup
    meta_by_proc = {}
    for key, meta in metadata.items():
        proc = str(meta.get("proceedingNumber", ""))
        if proc:
            meta_by_proc[proc] = meta

    # Merge
    merged = []
    for proc_num, opinion in opinion_splits.items():
        meta = meta_by_proc.get(proc_num, {})
        if not meta:
            continue

        combined = {
            **meta,
            "examiner_findings": opinion.get("examiner_findings", []),
            "appellant_arguments": opinion.get("appellant_arguments", []),
            "ptab_opinion": opinion.get("ptab_opinion", []),
        }

        # Must have issue_type
        issue_type = combined.get("issue_type", [])
        if not issue_type or (len(issue_type) == 1 and issue_type[0] == ""):
            continue

        merged.append(combined)
        if max_records and len(merged) >= max_records:
            break

    logger.info("Merged: %d records with issue_type + opinion_split", len(merged))
    return merged


def record_to_benchmark(record: dict) -> dict:
    """Convert one record to benchmark format."""
    # Join sentence lists into text
    examiner_text = "\n".join(record.get("examiner_findings", []))
    appellant_text = "\n".join(record.get("appellant_arguments", []))
    ptab_text = "\n".join(record.get("ptab_opinion", []))

    return {
        "case_id": f"PILOT-{record.get('proceedingNumber', 'unknown')}",
        "proceeding_number": record.get("proceedingNumber", ""),
        "patent_number": record.get("respondentPatentNumber", ""),
        "tech_center": record.get("respondentTechnologyCenterNumber", ""),
        "art_unit": record.get("respondentGroupArtUnitNumber", ""),
        "decision_date": record.get("decisionDate", ""),
        "examiner_findings": examiner_text,
        "appellant_arguments": appellant_text,
        "ground_truth": {
            "issue_type": record.get("issue_type", []),
            "board_rulings": record.get("board_rulings", []),
            "subdecision": record.get("subdecision", ""),
            "subdecision_coarse": record.get("subdecision_coarse", ""),
        },
        "ptab_opinion": ptab_text,
        "examiner_findings_sentences": record.get("examiner_findings", []),
        "appellant_arguments_sentences": record.get("appellant_arguments", []),
    }


def build_benchmark_dataset(
    tar_path: str,
    output_dir: str,
    max_records: int = None,
    require_examiner_findings: bool = True,
) -> str:
    """Build benchmark dataset from PILOT-Bench tar.gz."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    records = load_pilot_bench(tar_path, max_records=None)

    cases = []
    for record in records:
        case = record_to_benchmark(record)
        if require_examiner_findings and not case["examiner_findings"].strip():
            continue
        if not case["ground_truth"]["issue_type"]:
            continue
        cases.append(case)
        if max_records and len(cases) >= max_records:
            break

    # Stats
    issue_dist = {}
    decision_dist = {}
    tech_dist = {}
    for case in cases:
        for it in case["ground_truth"]["issue_type"]:
            issue_dist[it] = issue_dist.get(it, 0) + 1
        sd = case["ground_truth"]["subdecision_coarse"]
        decision_dist[sd] = decision_dist.get(sd, 0) + 1
        tc = case["tech_center"]
        tech_dist[tc] = tech_dist.get(tc, 0) + 1

    avg_examiner_len = sum(len(c["examiner_findings"]) for c in cases) / max(len(cases), 1)
    avg_appellant_len = sum(len(c["appellant_arguments"]) for c in cases) / max(len(cases), 1)

    stats = {
        "total_cases": len(cases),
        "with_examiner_findings": sum(1 for c in cases if c["examiner_findings"]),
        "with_appellant_arguments": sum(1 for c in cases if c["appellant_arguments"]),
        "avg_examiner_findings_chars": round(avg_examiner_len),
        "avg_appellant_arguments_chars": round(avg_appellant_len),
        "issue_type_distribution": dict(sorted(issue_dist.items(), key=lambda x: -x[1])),
        "subdecision_distribution": dict(sorted(decision_dist.items(), key=lambda x: -x[1])),
        "tech_center_top10": dict(sorted(tech_dist.items(), key=lambda x: -x[1])[:10]),
    }

    output_path = output_dir / "benchmark_dataset.json"
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "cases": cases}, f, ensure_ascii=False, indent=2)

    with open(output_dir / "benchmark_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)

    logger.info("Saved %d cases -> %s", len(cases), output_path)
    print(f"\n{'='*60}")
    print(f"PILOT-Bench ETL Complete")
    print(f"{'='*60}")
    print(f"Total cases:      {len(cases)}")
    print(f"Avg examiner text: {avg_examiner_len:.0f} chars")
    print(f"Avg appellant text: {avg_appellant_len:.0f} chars")
    print(f"\nIssue distribution:")
    for k, v in sorted(issue_dist.items(), key=lambda x: -x[1]):
        print(f"  {k:>10}: {v:>5} ({v/len(cases)*100:.1f}%)")
    print(f"\nDecision distribution:")
    for k, v in sorted(decision_dist.items(), key=lambda x: -x[1]):
        print(f"  {k:>30}: {v:>5} ({v/len(cases)*100:.1f}%)")
    print(f"\nSaved to: {output_path}")

    return str(output_path)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    tar_path = sys.argv[1] if len(sys.argv) > 1 else "data/pilot_bench/pilot-bench.tar.gz"
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "data/pilot_bench/benchmark"
    max_n = int(sys.argv[3]) if len(sys.argv) > 3 else None

    build_benchmark_dataset(tar_path, output_dir, max_records=max_n)
