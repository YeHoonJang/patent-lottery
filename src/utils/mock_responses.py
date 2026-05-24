"""
Mock response generator for testing the pipeline without LLM calls.

Generates deterministic but examiner-persona-aware mock responses.
The mock results vary based on examiner parameters so that population
analysis still produces meaningful distributions.
"""

import random
from src.models.schema import ExaminerProfile, PatentCase
from src.utils.llm_adapter import register_mock


def _decide(examiner: ExaminerProfile, base_prob: float, seed_offset: int = 0) -> bool:
    """Stochastic decision influenced by examiner strictness."""
    rng = random.Random((examiner.noise_seed or 0) + seed_offset)
    threshold = examiner.strictness
    return rng.random() > threshold * base_prob


def register_all_mocks(examiner: ExaminerProfile, case: PatentCase):
    """Register mock responses for all roles, tailored to this examiner."""
    rng = random.Random(examiner.noise_seed or 42)

    # ─── Claim Interpretation ─────────────────────────
    breadth_label = (
        "narrow" if examiner.claim_breadth < 0.35 else
        "broad" if examiner.claim_breadth > 0.65 else
        "moderate"
    )
    interpretations = []
    for c in case.claims:
        elements = [{"element_id": f"{c.claim_number}{chr(97+i)}", "text": part.strip()}
                     for i, part in enumerate(c.text.split(";")[:5]) if part.strip()]
        if not elements:
            elements = [{"element_id": f"{c.claim_number}a", "text": c.text[:200]}]
        interpretations.append({
            "claim_number": c.claim_number,
            "interpreted_scope": f"[{breadth_label} BRI] {c.text[:150]}...",
            "key_elements": elements,
            "breadth_assessment": breadth_label,
            "ambiguities": ["Term scope unclear"] if rng.random() > 0.6 else [],
        })
    register_mock("claim_interpretation", {"interpretations": interpretations})

    # ─── Prior Art Mapping ────────────────────────────
    mappings = []
    for c in case.claims:
        elem_mappings = []
        for elem in interpretations[c.claim_number - 1]["key_elements"][:4]:
            matched = []
            for pa in case.prior_art[:3]:
                score = round(rng.uniform(0.3, 0.95), 2)
                # Broader BRI → higher coverage scores
                score = min(1.0, score + (examiner.claim_breadth - 0.5) * 0.2)
                match_type = "explicit" if score > 0.8 else "implicit" if score > 0.6 else "obvious_variant"
                matched.append({
                    "ref_id": pa.ref_id,
                    "relevant_passage": f"[Mock] Relevant passage from {pa.title}",
                    "similarity_score": round(score, 2),
                    "match_type": match_type,
                })
            is_covered = any(m["similarity_score"] > 0.7 for m in matched)
            elem_mappings.append({
                "element_id": elem["element_id"],
                "element_text": elem["text"][:100],
                "matched_refs": matched,
                "is_covered": is_covered,
            })
        coverage = sum(1 for e in elem_mappings if e["is_covered"]) / max(len(elem_mappings), 1)
        mappings.append({
            "claim_number": c.claim_number,
            "element_mappings": elem_mappings,
            "closest_ref_id": case.prior_art[0].ref_id if case.prior_art else "",
            "coverage_score": round(coverage, 2),
        })
    register_mock("prior_art_mapping", {"mappings": mappings})

    # ─── Novelty Assessment ───────────────────────────
    nov_assessments = []
    for m in mappings:
        # Higher strictness + higher coverage → more likely to find anticipation
        anticipation_prob = m["coverage_score"] * examiner.strictness
        is_novel = rng.random() > anticipation_prob
        nov_assessments.append({
            "claim_number": m["claim_number"],
            "is_novel": is_novel,
            "reasoning": f"[Mock] {'Novel' if is_novel else 'Anticipated'} — coverage={m['coverage_score']:.0%}",
            "anticipating_refs": [m["closest_ref_id"]] if not is_novel else [],
            "confidence": round(rng.uniform(0.5, 0.9), 2),
        })
    register_mock("novelty_assessment", {"assessments": nov_assessments})

    # ─── Obviousness Analysis ─────────────────────────
    obv_assessments = []
    for m in mappings:
        # Low combination_threshold → easier to find obvious
        obvious_prob = (1 - examiner.combination_threshold) * m["coverage_score"] * 1.2
        is_obvious = rng.random() < min(obvious_prob, 0.95)
        obv_assessments.append({
            "claim_number": m["claim_number"],
            "is_obvious": is_obvious,
            "reasoning": f"[Mock] {'Obvious' if is_obvious else 'Non-obvious'} — threshold={examiner.combination_threshold:.2f}",
            "combination_refs": [pa.ref_id for pa in case.prior_art[:2]] if is_obvious else [],
            "motivation_to_combine": "[Mock] Common field, analogous art" if is_obvious else "",
            "graham_factors": {
                "scope_and_content": "[Mock] Prior art covers similar domain",
                "differences": "[Mock] Claim adds specific technical feature",
                "phosita_level": "[Mock] Skilled in optical imaging and cell biology",
                "secondary_considerations": "[Mock] Commercial success noted" if examiner.seniority > 0.5 else "[Mock] None significant",
            },
            "confidence": round(rng.uniform(0.5, 0.9), 2),
        })
    register_mock("obviousness_analysis", {"assessments": obv_assessments})


    # ─── Written Description Check ────────────────────
    wd_assessments = []
    for c in case.claims:
        # Higher rigor → more likely to find issues
        issue_prob = examiner.tech_specialization * 0.6
        has_issues = rng.random() < issue_prob
        issues = []
        unsupported = []
        enablement = []
        if has_issues:
            if rng.random() < 0.5:
                unsupported.append("[Mock] Term lacks explicit definition in specification")
                issues.append("Indefiniteness: key term not defined")
            if rng.random() < 0.4:
                enablement.append("[Mock] Insufficient detail for reproduction")
                issues.append("Enablement: insufficient implementation detail")
            if rng.random() < 0.3:
                issues.append("Written description: claim broader than specification support")
            if not issues:
                issues.append("[Mock] Minor definiteness concern")
        wd_assessments.append({
            "claim_number": c.claim_number,
            "has_issues": has_issues,
            "issues": issues,
            "unsupported_terms": unsupported,
            "enablement_concerns": enablement,
            "reasoning": f"[Mock] {'Issues found' if has_issues else 'No issues'} — rigor={examiner.tech_specialization:.2f}",
            "confidence": round(rng.uniform(0.5, 0.9), 2),
        })
    register_mock("written_desc_check", {"assessments": wd_assessments})

    # ─── Office Action ────────────────────────────────
    rejection_grounds = []
    allowable = []
    for i, (nov, obv) in enumerate(zip(nov_assessments, obv_assessments)):
        cn = nov["claim_number"]
        wd = wd_assessments[i] if i < len(wd_assessments) else {"has_issues": False}
        if wd.get("has_issues"):
            rejection_grounds.append({
                "claim_numbers": [cn],
                "basis": "112_written_description",
                "reasoning": f"[Mock] Claim {cn}: {'; '.join(wd.get('issues', []))}",
                "cited_refs": [],
            })
        if not nov["is_novel"]:
            rejection_grounds.append({
                "claim_numbers": [cn],
                "basis": "102_novelty",
                "reasoning": f"[Mock] Claim {cn} anticipated by {nov['anticipating_refs']}",
                "cited_refs": nov["anticipating_refs"],
            })
        elif obv["is_obvious"]:
            rejection_grounds.append({
                "claim_numbers": [cn],
                "basis": "103_obviousness",
                "reasoning": f"[Mock] Claim {cn} obvious over combination",
                "cited_refs": obv["combination_refs"],
            })
        else:
            allowable.append(cn)

    register_mock("oa_drafter", {
        "rejection_grounds": rejection_grounds,
        "objections": [],
        "allowable_claims": allowable,
        "full_text": f"[Mock Office Action] Rejections: {len(rejection_grounds)}, Allowable: {allowable}",
    })

    # ─── Final Decision ───────────────────────────────
    n_claims = len(case.claims)
    n_allowable = len(allowable)
    grant_ratio = n_allowable / max(n_claims, 1)

    if grant_ratio >= 0.8:
        decision = "grant"
    elif grant_ratio >= 0.3:
        decision = "amend_suggested"
    else:
        decision = "reject"

    # Score influenced by policy stance
    score = grant_ratio * 0.7 + examiner.policy_stance * 0.3
    score = max(0.0, min(1.0, score))

    claim_decisions = []
    for nov, obv in zip(nov_assessments, obv_assessments):
        cn = nov["claim_number"]
        if cn in allowable:
            claim_decisions.append({"claim_number": cn, "decision": "grant", "primary_basis": None, "reasoning": "[Mock] Patentable"})
        elif not nov["is_novel"]:
            claim_decisions.append({"claim_number": cn, "decision": "reject", "primary_basis": "102_novelty", "reasoning": "[Mock] Anticipated"})
        else:
            claim_decisions.append({"claim_number": cn, "decision": "reject", "primary_basis": "103_obviousness", "reasoning": "[Mock] Obvious"})

    register_mock("final_decision", {
        "decision": decision,
        "decision_score": round(score, 3),
        "summary": f"[Mock] {decision.upper()} — {n_allowable}/{n_claims} claims allowable",
        "claim_decisions": claim_decisions,
        "key_issues": ["Prior art coverage", "Combination motivation"],
        "examiner_notes": f"[Mock] Examiner {examiner.display_name} ({examiner.archetype_name})",
    })
