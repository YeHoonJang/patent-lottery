"""
WARNING: This module uses BigQuery which costs money.
All queries MUST go through src.utils.bigquery_safety.safe_query().
NEVER call client.query() directly.

Incident 2026-03-24: $48 cost from uncontrolled batch queries.

BigQuery Patent Enricher — fetches full patent text (claims + description + citations)
from Google Patents Public Dataset and enriches our benchmark/KB data.

Three modes:
1. Enrich PILOT-Bench cases (add claims, description, cited refs)
2. Enrich Tomocube KB (add claims, description to existing 120+276 patents)
3. Fetch cited references recursively (get prior art full text)
"""

import json
import logging
import os
import time
from pathlib import Path

os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS",
                      "/data1/patent-kb/config/tomocube-patent-intel-675e4e2b9540.json")

logger = logging.getLogger("vpatent.bigquery")


def get_bq_client():
    from google.cloud import bigquery
    return bigquery.Client()


def normalize_patent_number(num: str) -> str:
    """Convert '8655637' or 'US8655637B2' to 'US-8655637-B2' format."""
    num = num.strip().replace(",", "")
    if num.startswith("US-"):
        return num
    # Strip country/kind codes if present
    import re
    m = re.match(r"(?:US)?(\d+)(?:([AB]\d?))?", num)
    if m:
        number = m.group(1)
        kind = m.group(2) or "B2"
        return f"US-{number}-{kind}"
    return f"US-{num}-B2"


def fetch_patents_batch(client, patent_numbers: list[str], batch_size: int = 50) -> dict:
    """
    Fetch full patent data from BigQuery for a batch of patent numbers.
    Returns dict keyed by publication_number.
    """
    results = {}

    for i in range(0, len(patent_numbers), batch_size):
        batch = patent_numbers[i:i + batch_size]
        nums_str = ", ".join(f"'{n}'" for n in batch)

        query = f"""
        SELECT
            p.publication_number,
            p.filing_date,
            p.grant_date,
            p.family_id,
            (SELECT t.text FROM UNNEST(p.title_localized) t WHERE t.language = 'en' LIMIT 1) as title,
            (SELECT a.text FROM UNNEST(p.abstract_localized) a WHERE a.language = 'en' LIMIT 1) as abstract_text,
            (SELECT c.text FROM UNNEST(p.claims_localized) c WHERE c.language = 'en' LIMIT 1) as claims_text,
            SUBSTR((SELECT d.text FROM UNNEST(p.description_localized) d WHERE d.language = 'en' LIMIT 1), 1, 10000) as desc_summary,
            LENGTH((SELECT d.text FROM UNNEST(p.description_localized) d WHERE d.language = 'en' LIMIT 1)) as desc_full_len,
            ARRAY_TO_STRING(ARRAY(SELECT DISTINCT a.name FROM UNNEST(p.assignee_harmonized) a), '|') as assignees,
            ARRAY_TO_STRING(ARRAY(SELECT DISTINCT c.code FROM UNNEST(p.ipc) c LIMIT 10), '|') as ipc_codes,
            ARRAY_TO_STRING(ARRAY(SELECT ct.publication_number FROM UNNEST(p.citation) ct
                WHERE ct.publication_number IS NOT NULL LIMIT 30), '|') as cited_refs
        FROM `patents-public-data.patents.publications` p
        WHERE p.publication_number IN ({nums_str})
        """

        rows = client.query(query).result()
        for row in rows:
            results[row.publication_number] = {
                "publication_number": row.publication_number,
                "title": row.title or "",
                "abstract": row.abstract_text or "",
                "claims_text": row.claims_text or "",
                "desc_summary": row.desc_summary or "",
                "desc_full_len": row.desc_full_len or 0,
                "filing_date": str(row.filing_date or ""),
                "grant_date": str(row.grant_date or ""),
                "assignees": row.assignees or "",
                "ipc_codes": row.ipc_codes or "",
                "cited_refs": [r for r in (row.cited_refs or "").split("|") if r],
            }

        logger.info("Fetched batch %d-%d: %d results", i, i + len(batch), len(results))

    return results


def enrich_pilot_bench(
    benchmark_path: str,
    output_path: str,
    max_cases: int = 100,
):
    """Enrich PILOT-Bench cases with full patent text from BigQuery."""
    with open(benchmark_path) as f:
        data = json.load(f)

    cases = [c for c in data["cases"] if c.get("patent_number")][:max_cases]
    logger.info("Enriching %d PILOT-Bench cases with BigQuery data", len(cases))

    # Normalize patent numbers
    patent_nums = []
    num_map = {}
    for c in cases:
        raw = c["patent_number"]
        normalized = normalize_patent_number(raw)
        patent_nums.append(normalized)
        num_map[normalized] = raw
        # Also try B1 variant
        if normalized.endswith("-B2"):
            patent_nums.append(normalized.replace("-B2", "-B1"))

    client = get_bq_client()
    bq_data = fetch_patents_batch(client, list(set(patent_nums)))

    # Merge back
    enriched = []
    not_found = []
    for c in cases:
        raw = c["patent_number"]
        norm = normalize_patent_number(raw)
        bq = bq_data.get(norm) or bq_data.get(norm.replace("-B2", "-B1"))

        if bq and bq["claims_text"]:
            c["bq_data"] = bq
            enriched.append(c)
        else:
            not_found.append(raw)

    # Stats
    avg_claims_len = sum(len(c["bq_data"]["claims_text"]) for c in enriched) / max(len(enriched), 1)
    avg_desc_len = sum(c["bq_data"]["desc_full_len"] for c in enriched) / max(len(enriched), 1)
    avg_citations = sum(len(c["bq_data"]["cited_refs"]) for c in enriched) / max(len(enriched), 1)

    output = {
        "stats": {
            "total_attempted": len(cases),
            "enriched": len(enriched),
            "not_found": len(not_found),
            "avg_claims_chars": round(avg_claims_len),
            "avg_desc_chars": round(avg_desc_len),
            "avg_citations": round(avg_citations, 1),
        },
        "cases": enriched,
        "not_found": not_found,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"PILOT-Bench BigQuery Enrichment")
    print(f"{'='*60}")
    print(f"Attempted:     {len(cases)}")
    print(f"Enriched:      {len(enriched)}")
    print(f"Not found:     {len(not_found)}")
    print(f"Avg claims:    {avg_claims_len:.0f} chars")
    print(f"Avg desc:      {avg_desc_len:.0f} chars")
    print(f"Avg citations: {avg_citations:.1f}")
    print(f"Saved to:      {output_path}")

    return output


def enrich_tomocube_kb(output_path: str):
    """Enrich Tomocube KB similar patents with full claims + description."""
    import sqlite3
    db = sqlite3.connect("/data1/patent-kb/data/patent_kb.db")
    db.row_factory = sqlite3.Row
    cursor = db.cursor()

    cursor.execute("SELECT publication_number, title, abstract FROM similar_patent_bq WHERE abstract IS NOT NULL")
    patents = [dict(row) for row in cursor.fetchall()]
    db.close()

    logger.info("Enriching %d Tomocube KB patents", len(patents))

    patent_nums = [p["publication_number"] for p in patents]
    client = get_bq_client()
    bq_data = fetch_patents_batch(client, patent_nums)

    enriched = []
    for p in patents:
        bq = bq_data.get(p["publication_number"])
        if bq and bq["claims_text"]:
            enriched.append({**p, "bq_data": bq})

    avg_claims = sum(len(e["bq_data"]["claims_text"]) for e in enriched) / max(len(enriched), 1)

    output = {
        "stats": {
            "total": len(patents),
            "enriched_with_claims": len(enriched),
            "avg_claims_chars": round(avg_claims),
        },
        "patents": enriched,
    }

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Tomocube KB Enrichment")
    print(f"{'='*60}")
    print(f"Total KB patents:       {len(patents)}")
    print(f"Enriched with claims:   {len(enriched)}")
    print(f"Avg claims length:      {avg_claims:.0f} chars")
    print(f"Saved to:               {output_path}")

    return output


def fetch_cited_prior_art(patent_numbers: list[str], output_path: str):
    """Fetch full text of cited prior art references."""
    client = get_bq_client()
    bq_data = fetch_patents_batch(client, patent_numbers)

    enriched = [v for v in bq_data.values() if v["claims_text"]]

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"count": len(enriched), "patents": enriched}, f, ensure_ascii=False, indent=2)

    print(f"Fetched {len(enriched)} cited prior art patents -> {output_path}")
    return enriched


if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["pilot", "tomocube", "citations"],
                        help="pilot=enrich PILOT-Bench, tomocube=enrich KB, citations=fetch cited refs")
    parser.add_argument("--input", default="data/pilot_bench/benchmark/benchmark_dataset.json")
    parser.add_argument("--output", default=None)
    parser.add_argument("--max", type=int, default=100)
    args = parser.parse_args()

    if args.mode == "pilot":
        out = args.output or "data/pilot_bench/enriched_bq/pilot_bench_bq.json"
        enrich_pilot_bench(args.input, out, max_cases=args.max)
    elif args.mode == "tomocube":
        out = args.output or "data/tomocube_kb/enriched_kb.json"
        enrich_tomocube_kb(out)
    elif args.mode == "citations":
        # Read patent numbers from input file
        with open(args.input) as f:
            nums = json.load(f)
        out = args.output or "data/cited_prior_art.json"
        fetch_cited_prior_art(nums, out)
