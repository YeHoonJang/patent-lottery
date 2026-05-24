"""
Fetch patent claims from Google Patents for PILOT-Bench cases.
Uses Google Patents URL pattern: https://patents.google.com/patent/US{number}/en

Falls back to LLM extraction if HTML parsing fails.
"""

import json
import logging
import os
import re
import time
import urllib.request
from pathlib import Path

logger = logging.getLogger("vpatent.fetch")

HEADERS = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
}


def fetch_google_patent(patent_number: str) -> dict:
    """Fetch patent data from Google Patents."""
    # Clean patent number
    num = patent_number.strip().replace(",", "")
    if not num:
        return {}

    url = f"https://patents.google.com/patent/US{num}/en"
    try:
        req = urllib.request.Request(url, headers=HEADERS)
        with urllib.request.urlopen(req, timeout=20) as resp:
            html = resp.read().decode("utf-8", errors="ignore")
    except Exception as e:
        logger.warning("Failed to fetch US%s: %s", num, e)
        return {}

    result = {"patent_number": num, "url": url}

    # Extract title
    title_match = re.search(r'<meta name="DC\.title" content="([^"]*)"', html)
    if title_match:
        result["title"] = title_match.group(1)

    # Extract abstract
    abs_match = re.search(r'<meta name="DC\.description" content="([^"]*)"', html)
    if abs_match:
        result["abstract"] = abs_match.group(1)

    # Extract claims from the claims section
    # Google Patents uses <section itemprop="claims"> or class="claims"
    claims_section = re.search(
        r'<section[^>]*itemprop="claims"[^>]*>(.*?)</section>',
        html, re.DOTALL
    )
    if not claims_section:
        claims_section = re.search(
            r'class="claims"[^>]*>(.*?)</div>\s*</div>\s*</div>',
            html, re.DOTALL
        )

    claims = []
    if claims_section:
        claim_text = claims_section.group(1)
        # Split by claim divs
        claim_divs = re.findall(
            r'<div[^>]*class="claim"[^>]*>(.*?)</div>',
            claim_text, re.DOTALL
        )
        if not claim_divs:
            # Try another pattern
            claim_divs = re.findall(
                r'<claim[^>]*>(.*?)</claim>',
                claim_text, re.DOTALL
            )
        if not claim_divs:
            # Fallback: split by claim numbers
            claim_divs = re.split(r'\n\s*\d+\.\s+', claim_text)

        for i, cd in enumerate(claim_divs):
            text = re.sub(r'<[^>]+>', ' ', cd).strip()
            text = re.sub(r'\s+', ' ', text).strip()
            if len(text) > 20:
                # Determine if independent or dependent
                is_dep = bool(re.search(r'claim \d+|according to claim', text, re.I))
                dep_on = None
                if is_dep:
                    dep_match = re.search(r'claim (\d+)', text, re.I)
                    if dep_match:
                        dep_on = int(dep_match.group(1))
                claims.append({
                    "claim_number": i + 1,
                    "claim_type": "dependent" if is_dep else "independent",
                    "text": text,
                    "depends_on": dep_on,
                })

    result["claims"] = claims
    result["n_claims"] = len(claims)

    # Extract IPC codes
    ipc_matches = re.findall(r'<meta[^>]*scheme="ipc"[^>]*content="([^"]*)"', html)
    result["ipc_codes"] = ipc_matches[:10]

    return result


def fetch_claims_batch(
    benchmark_path: str,
    output_dir: str,
    max_patents: int = 100,
    delay: float = 1.0,
) -> str:
    """
    Fetch claims for PILOT-Bench cases that have patent numbers.

    Args:
        benchmark_path: Path to benchmark_dataset.json
        output_dir: Where to save enriched dataset
        max_patents: Max patents to fetch
        delay: Seconds between requests (be nice to Google)
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(benchmark_path) as f:
        data = json.load(f)

    # Filter cases with patent numbers
    cases_with_patents = [c for c in data["cases"] if c.get("patent_number")]
    logger.info("Cases with patent_number: %d / %d", len(cases_with_patents), len(data["cases"]))

    enriched = []
    failed = []
    cache_path = output_dir / "patent_cache.json"

    # Load cache
    cache = {}
    if cache_path.exists():
        with open(cache_path) as f:
            cache = json.load(f)
        logger.info("Loaded cache: %d patents", len(cache))

    for i, case in enumerate(cases_with_patents[:max_patents]):
        pnum = case["patent_number"]

        if pnum in cache:
            patent_data = cache[pnum]
            logger.debug("[%d/%d] Cache hit: US%s", i+1, max_patents, pnum)
        else:
            logger.info("[%d/%d] Fetching US%s...", i+1, max_patents, pnum)
            patent_data = fetch_google_patent(pnum)
            cache[pnum] = patent_data
            time.sleep(delay)

            # Save cache periodically
            if (i + 1) % 20 == 0:
                with open(cache_path, "w") as f:
                    json.dump(cache, f, ensure_ascii=False, indent=2)

        if patent_data.get("claims"):
            enriched_case = {
                **case,
                "patent_data": patent_data,
            }
            enriched.append(enriched_case)
        else:
            failed.append({"case_id": case["case_id"], "patent_number": pnum,
                          "reason": "no claims extracted"})

        if (i + 1) % 10 == 0:
            logger.info("Progress: %d/%d fetched, %d with claims, %d failed",
                        i+1, max_patents, len(enriched), len(failed))

    # Final cache save
    with open(cache_path, "w") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)

    # Save enriched dataset
    output_path = output_dir / "enriched_benchmark.json"
    stats = {
        "total_attempted": min(max_patents, len(cases_with_patents)),
        "with_claims": len(enriched),
        "failed": len(failed),
        "avg_claims": sum(len(c["patent_data"]["claims"]) for c in enriched) / max(len(enriched), 1),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump({"stats": stats, "cases": enriched, "failed": failed}, f,
                  ensure_ascii=False, indent=2)

    print(f"\n{'='*60}")
    print(f"Claims Fetch Complete")
    print(f"{'='*60}")
    print(f"Attempted:   {stats['total_attempted']}")
    print(f"With claims: {stats['with_claims']}")
    print(f"Failed:      {stats['failed']}")
    print(f"Avg claims:  {stats['avg_claims']:.1f}")
    print(f"Saved to:    {output_path}")

    return str(output_path)


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

    benchmark = sys.argv[1] if len(sys.argv) > 1 else "data/pilot_bench/benchmark/benchmark_dataset.json"
    output = sys.argv[2] if len(sys.argv) > 2 else "data/pilot_bench/enriched"
    max_n = int(sys.argv[3]) if len(sys.argv) > 3 else 100

    fetch_claims_batch(benchmark, output, max_patents=max_n, delay=1.5)
