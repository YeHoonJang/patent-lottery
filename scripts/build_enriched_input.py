#!/usr/bin/env python3
"""Build enriched eval input by merging PILOT-Bench cases with HF patent full-text data."""

import json
import os
import re
import glob

from pathlib import Path
DATA_DIR = str(Path(__file__).resolve().parents[1] / "data")
HF_DIR = os.path.join(DATA_DIR, "ptab_patents_hf/extracted")
OUTPUT = os.path.join(DATA_DIR, "eval_100_enriched_input.json")

def find_hf_case_dir(proceeding_number):
    pattern = os.path.join(HF_DIR, "%s_*" % proceeding_number)
    matches = glob.glob(pattern)
    return matches[0] if matches else None

def load_json(filepath):
    with open(filepath) as f:
        return json.load(f)

def parse_claims_text(text):
    """Parse a single claims string into individual claims."""
    if not text or not isinstance(text, str):
        return []
    # Try splitting by claim number patterns like "1.", "2.", etc.
    parts = re.split(r"\n\s*(\d+)\s*\.\s+", text)
    claims = []
    if len(parts) > 1:
        # parts[0] is before first claim number, then alternating: number, text
        for i in range(1, len(parts), 2):
            num = int(parts[i])
            body = parts[i+1].strip() if i+1 < len(parts) else ""
            claims.append({"claim_number": num, "text": body})
    else:
        # Fallback: whole text as one claim
        claims.append({"claim_number": 1, "text": text.strip()})
    return claims

def format_claims(claims_data):
    if isinstance(claims_data, str):
        return parse_claims_text(claims_data)
    elif isinstance(claims_data, list):
        result = []
        for i, c in enumerate(claims_data):
            if isinstance(c, dict):
                result.append(c)
            elif isinstance(c, str):
                # Could be individual claim or full text
                if len(claims_data) == 1:
                    return parse_claims_text(c)
                result.append({"claim_number": i+1, "text": c.strip()})
        return result
    return []

def join_desc(desc):
    if isinstance(desc, list):
        return "\n\n".join(str(d) for d in desc)
    return str(desc) if desc else ""

def build_enriched_case(ready_case, hf_case_dir):
    pn = str(ready_case.get("proceeding_number",
             ready_case.get("case_id","").replace("PILOT-","")))

    case = {
        "case_id": "PILOT-%s" % pn,
        "proceeding_number": pn,
        "ground_truth": ready_case.get("ground_truth", {}),
        "tech_center": ready_case.get("tech_center", ""),
        "jurisdiction": "USPTO",
    }

    # Applicant patent - prefer original
    orig_dir = os.path.join(hf_case_dir, "ApplicantPatent", "original")
    parsed_dir = os.path.join(hf_case_dir, "ApplicantPatent")

    app_data = None
    if os.path.exists(orig_dir):
        jsons = [f for f in os.listdir(orig_dir) if f.endswith(".json")]
        if jsons:
            app_data = load_json(os.path.join(orig_dir, jsons[0]))
            case["applicant_source"] = "original"
    if not app_data and os.path.exists(parsed_dir):
        jsons = [f for f in os.listdir(parsed_dir) if f.endswith(".json")]
        if jsons:
            app_data = load_json(os.path.join(parsed_dir, jsons[0]))
            case["applicant_source"] = "parsed"

    if app_data:
        case["title"] = app_data.get("title", "")
        case["applicant_pub_number"] = app_data.get("publication_number", app_data.get("filename", ""))
        case["application_number"] = app_data.get("application_number", "")
        case["filing_date"] = app_data.get("pub_date", "")
        case["abstract"] = app_data.get("abstract", "")
        claims = format_claims(app_data.get("claims", []))
        case["claims"] = claims
        case["claims_count"] = len(claims)
        case["description"] = join_desc(app_data.get("description", []))
        case["description_chars"] = len(case["description"])

    # Prior art patents
    prior_arts = []
    for subdir_name in ["PriorArtPatent"]:
        # Try original first
        pa_orig = os.path.join(hf_case_dir, subdir_name, "original")
        pa_base = os.path.join(hf_case_dir, subdir_name)

        if os.path.exists(pa_orig):
            pa_files = [os.path.join(pa_orig, f) for f in os.listdir(pa_orig) if f.endswith(".json")]
        elif os.path.exists(pa_base):
            pa_files = [os.path.join(pa_base, f) for f in os.listdir(pa_base) if f.endswith(".json")]
        else:
            pa_files = []

        for pf in pa_files:
            pa = load_json(pf)
            pa_claims = format_claims(pa.get("claims", []))
            pa_entry = {
                "ref_id": pa.get("publication_number", os.path.basename(pf).replace(".json","")),
                "title": pa.get("title", ""),
                "abstract": pa.get("abstract", ""),
                "claims": pa_claims,
                "claims_count": len(pa_claims),
                "description": join_desc(pa.get("description", [])),
                "description_chars": len(join_desc(pa.get("description", []))),
            }
            prior_arts.append(pa_entry)

    case["prior_art"] = prior_arts
    case["prior_art_count"] = len(prior_arts)

    return case

def main():
    with open(os.path.join(DATA_DIR, "experiment_eval_100_v2.json")) as f:
        eval100 = json.load(f)

    with open(os.path.join(DATA_DIR, "experiment_ready_cases.json")) as f:
        ready = json.load(f)
    ready_by_pn = {}
    for c in ready["cases"]:
        pn = str(c.get("proceeding_number", c.get("case_id","").replace("PILOT-","")))
        ready_by_pn[pn] = c

    enriched_cases = []
    errors = []

    for pn in eval100["proceeding_numbers"]:
        pn = str(pn)
        hf_dir = find_hf_case_dir(pn)
        if not hf_dir:
            errors.append({"id": pn, "error": "no HF dir"})
            continue

        ready_case = ready_by_pn.get(pn, {"proceeding_number": pn})
        try:
            enriched = build_enriched_case(ready_case, hf_dir)
            enriched_cases.append(enriched)
            print("  OK %s: claims=%d PA=%d desc=%dK abs=%d" % (
                pn, enriched["claims_count"], enriched["prior_art_count"],
                enriched.get("description_chars",0)//1000,
                1 if enriched.get("abstract") else 0))
        except Exception as e:
            errors.append({"id": pn, "error": str(e)})
            print("  ERR %s: %s" % (pn, e))

    output = {
        "total": len(enriched_cases),
        "cases": enriched_cases,
        "errors": errors,
        "note": "Enriched with HF full-text (companion patent corpus released with this paper). Includes abstract, full description, all claims, PA full text."
    }

    with open(OUTPUT, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print("\n=== Summary ===")
    print("Total enriched: %d" % len(enriched_cases))
    print("Errors: %d" % len(errors))
    if enriched_cases:
        avg_cl = sum(c["claims_count"] for c in enriched_cases) / len(enriched_cases)
        avg_pa = sum(c["prior_art_count"] for c in enriched_cases) / len(enriched_cases)
        avg_desc = sum(c.get("description_chars",0) for c in enriched_cases) / len(enriched_cases)
        has_ab = sum(1 for c in enriched_cases if c.get("abstract"))
        print("Avg claims: %.1f" % avg_cl)
        print("Avg PA refs: %.1f" % avg_pa)
        print("Avg description: %.1fK chars" % (avg_desc/1000))
        print("Has abstract: %d/%d" % (has_ab, len(enriched_cases)))
    print("\nSaved: %s" % OUTPUT)

if __name__ == "__main__":
    main()
