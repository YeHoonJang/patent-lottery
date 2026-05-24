"""
BigQuery Safety Wrapper — MANDATORY dry_run before any query execution.
ALL BigQuery calls in this project MUST go through this module.

INCIDENT: 2026-03-24, 8,698GB scanned ($48 cost) from batch citation queries.
"""

import json
import logging
import os

os.environ.setdefault("GOOGLE_APPLICATION_CREDENTIALS",
    "/data1/patent-kb/config/tomocube-patent-intel-675e4e2b9540.json")

logger = logging.getLogger("vpatent.bigquery")

# Hard limit: refuse to run queries scanning more than this without explicit override
MAX_SCAN_GB = 10.0
COST_PER_TB = 6.25  # USD


def safe_query(query: str, max_scan_gb: float = MAX_SCAN_GB, force: bool = False) -> list[dict]:
    """
    Execute BigQuery query with MANDATORY dry_run cost check.

    Args:
        query: SQL query string
        max_scan_gb: Max allowed scan size in GB (default 10GB)
        force: Set True only if user explicitly approved the cost

    Returns:
        List of row dicts

    Raises:
        RuntimeError if estimated scan exceeds max_scan_gb and force=False
    """
    from google.cloud import bigquery

    client = bigquery.Client()

    # STEP 1: MANDATORY dry run
    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    dry_job = client.query(query, job_config=job_config)
    scan_bytes = dry_job.total_bytes_processed
    scan_gb = scan_bytes / 1e9
    cost_usd = scan_gb / 1000 * COST_PER_TB

    logger.info("BigQuery dry_run: %.2f GB ($%.3f)", scan_gb, cost_usd)

    if scan_gb > max_scan_gb and not force:
        msg = (f"BLOCKED: Query would scan {scan_gb:.1f}GB (${cost_usd:.2f}). "
               f"Limit is {max_scan_gb}GB. Use force=True with user approval.")
        logger.error(msg)
        raise RuntimeError(msg)

    # STEP 2: Execute
    logger.info("Executing query: %.2f GB scan", scan_gb)
    result = client.query(query).result()
    rows = [dict(row) for row in result]
    logger.info("Query returned %d rows", len(rows))

    return rows


def check_cost(query: str) -> dict:
    """Check estimated cost of a query WITHOUT executing it."""
    from google.cloud import bigquery

    client = bigquery.Client()
    job_config = bigquery.QueryJobConfig(dry_run=True, use_query_cache=False)
    dry_job = client.query(query, job_config=job_config)
    scan_bytes = dry_job.total_bytes_processed
    scan_gb = scan_bytes / 1e9
    cost_usd = scan_gb / 1000 * COST_PER_TB

    return {
        "scan_bytes": scan_bytes,
        "scan_gb": round(scan_gb, 2),
        "estimated_cost_usd": round(cost_usd, 3),
        "within_limit": scan_gb <= MAX_SCAN_GB,
    }


if __name__ == "__main__":
    # Test: check cost of a simple query
    logging.basicConfig(level=logging.INFO)
    q = "SELECT publication_number FROM `patents-public-data.patents.publications` WHERE publication_number = 'US-8655637-B2'"
    result = check_cost(q)
    print(f"Cost check: {json.dumps(result, indent=2)}")
