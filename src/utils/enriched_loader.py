"""
Load enriched patent input into PatentCase schema.

Handles the richer data format from eval_100_enriched_input.json:
- Full description text
- PA with claims and description
- All claim types (independent + dependent)
"""

from src.models.schema import PatentCase, Claim, ClaimType, PriorArtReference


# Token budget constants (chars, ~4 chars/token for mixed en/ko)
MAX_PA_CLAIMS_CHARS = 8000       # ~2K tokens per PA
MAX_PA_DESC_CHARS = 12000        # ~3K tokens per PA
MAX_CLAIM_TEXT_CHARS = 3000      # per individual claim
MAX_DESCRIPTION_CHARS = 200000   # full spec for SpecRetriever (not sent raw to LLM)


def dict_to_case_enriched(d: dict) -> PatentCase:
    """Convert enriched input dict to PatentCase with full data."""

    # --- Claims ---
    claims = []
    for c in d.get("claims", []):
        if isinstance(c, dict):
            text = c.get("text", "")
            if isinstance(text, list):
                text = " ".join(str(t) for t in text)
            text = str(text)[:MAX_CLAIM_TEXT_CHARS]

            claim_type = ClaimType.INDEPENDENT
            depends_on = c.get("depends_on")
            ct = c.get("claim_type", "")
            if ct == "dependent" or depends_on:
                claim_type = ClaimType.DEPENDENT

            claims.append(Claim(
                claim_number=c.get("claim_number", len(claims) + 1),
                claim_type=claim_type,
                text=text,
                depends_on=depends_on,
            ))
        elif isinstance(c, str):
            claims.append(Claim(
                claim_number=len(claims) + 1,
                claim_type=ClaimType.INDEPENDENT,
                text=c[:MAX_CLAIM_TEXT_CHARS],
            ))

    if not claims:
        claims = [Claim(claim_number=1, claim_type=ClaimType.INDEPENDENT, text="(no claims)")]

    # --- Prior Art ---
    prior_art = []
    for pa in d.get("prior_art", []):
        # Build claims_text from PA claims list
        pa_claims_text = ""
        pa_claims = pa.get("claims", [])
        if isinstance(pa_claims, list):
            parts = []
            for pc in pa_claims:
                if isinstance(pc, dict):
                    parts.append("Claim %s: %s" % (pc.get("claim_number", ""), pc.get("text", "")))
                elif isinstance(pc, str):
                    parts.append(pc)
            pa_claims_text = "\n".join(parts)[:MAX_PA_CLAIMS_CHARS]
        elif isinstance(pa_claims, str):
            pa_claims_text = pa_claims[:MAX_PA_CLAIMS_CHARS]

        # PA description
        pa_desc = pa.get("description", "")
        if isinstance(pa_desc, list):
            pa_desc = "\n".join(str(x) for x in pa_desc)
        pa_desc = str(pa_desc)[:MAX_PA_DESC_CHARS]

        # PA abstract
        pa_abstract = pa.get("abstract", "")
        if isinstance(pa_abstract, list):
            pa_abstract = " ".join(str(x) for x in pa_abstract)
        pa_abstract = str(pa_abstract)[:2000]

        prior_art.append(PriorArtReference(
            ref_id=pa.get("ref_id", ""),
            title=pa.get("title", ""),
            abstract=pa_abstract,
            relevant_text=pa.get("relevant_text", ""),
            claims_text=pa_claims_text,
            description_text=pa_desc,
            source=pa.get("source", "HF-fulltext"),
        ))

    # --- Description ---
    description = d.get("description", "")
    if isinstance(description, list):
        description = "\n\n".join(str(x) for x in description)
    description = str(description)[:MAX_DESCRIPTION_CHARS]

    # description_summary: first ~2000 chars for quick reference
    desc_summary = d.get("description_summary", "")
    if not desc_summary and description:
        desc_summary = description[:2000]

    # --- Abstract ---
    abstract = d.get("abstract", "")
    if isinstance(abstract, list):
        abstract = " ".join(str(x) for x in abstract)
    abstract = str(abstract)[:2000]

    return PatentCase(
        case_id=d.get("case_id", ""),
        title=d.get("title", ""),
        abstract=abstract,
        description_summary=desc_summary,
        description_full=description,
        claims=claims,
        prior_art=prior_art,
        ipc_codes=d.get("ipc_codes", []),
        applicant=d.get("applicant", d.get("applicant_pub_number", "")),
        filing_date=d.get("filing_date", ""),
        jurisdiction=d.get("jurisdiction", "USPTO"),
    )
