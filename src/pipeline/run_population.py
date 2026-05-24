"""
Population pipeline: runs one case through N examiners and aggregates.
"""

import json
import logging
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from src.models.schema import ExaminerProfile, ExaminerRunTrace, PatentCase, PopulationSummary
from src.examiner.population import generate_population
from src.pipeline.run_case import run_single_examiner
from concurrent.futures import ThreadPoolExecutor, as_completed
from src.utils.llm_adapter import LLMAdapter

logger = logging.getLogger("vpatent.population_run")


def run_population(
    case: PatentCase,
    population: list[ExaminerProfile],
    llm: LLMAdapter,
    output_dir: str,
    verbose: bool = False,
) -> tuple[list[ExaminerRunTrace], PopulationSummary]:
    """
    Run a patent case through all examiners in the population.

    Returns:
        Tuple of (list of traces, population summary)
    """
    run_dir = Path(output_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save case
    with open(run_dir / "case.json", "w", encoding="utf-8") as f:
        json.dump(case.model_dump(), f, ensure_ascii=False, indent=2)

    traces: list[ExaminerRunTrace] = []
    start = time.time()
    max_parallel = getattr(llm, "max_parallel", 1)

    def _run_one(idx_examiner):
        i, examiner = idx_examiner
        logger.info("Examiner %d/%d: %s (%s)",
                     i, len(population), examiner.display_name, examiner.archetype_name)
        examiner_llm = LLMAdapter(
            mode=llm.mode, model=llm.model,
            timeout=llm.timeout, retries=llm.retries,
        )
        return run_single_examiner(
            case=case,
            examiner=examiner,
            llm=examiner_llm,
            output_dir=str(run_dir / "examiners"),
            verbose=verbose,
        )

    if max_parallel > 1:
        logger.info("Running %d examiners in parallel (max_workers=%d)", len(population), max_parallel)
        with ThreadPoolExecutor(max_workers=max_parallel) as executor:
            futures = {executor.submit(_run_one, (i, ex)): i
                       for i, ex in enumerate(population, 1)}
            for future in as_completed(futures):
                traces.append(future.result())
    else:
        for i, examiner in enumerate(population, 1):
            traces.append(_run_one((i, examiner)))

    total_elapsed = time.time() - start
    logger.info("Population run complete: %d examiners in %.1fs", len(population), total_elapsed)

    # Aggregate
    summary = aggregate_results(case, traces)
    summary.era_context = population[0].era_context_name if population else ""

    # Save summary
    with open(run_dir / "population_summary.json", "w", encoding="utf-8") as f:
        json.dump(summary.model_dump(), f, ensure_ascii=False, indent=2)

    # Save readable report
    report = generate_report(case, traces, summary)
    with open(run_dir / "report.md", "w", encoding="utf-8") as f:
        f.write(report)

    return traces, summary


def aggregate_results(
    case: PatentCase,
    traces: list[ExaminerRunTrace],
) -> PopulationSummary:
    """Compute population-level statistics from individual traces."""
    summary = PopulationSummary(
        case_id=case.case_id,
        n_examiners=len(traces),
    )

    decisions = []
    rejection_basis_counts: dict[str, int] = {}
    claim_grants: dict[int, int] = {c.claim_number: 0 for c in case.claims}
    claim_total: dict[int, int] = {c.claim_number: 0 for c in case.claims}

    for trace in traces:
        summary.examiner_run_ids.append(trace.run_id)
        fd = trace.final_decision
        if not fd:
            continue

        dec = fd.get("decision", fd) if isinstance(fd, dict) else getattr(fd, "decision", None)
        if dec:
            decisions.append(dec)

        # Count rejection bases from OA
        oa = trace.office_action
        if isinstance(oa, dict):
            for ground in oa.get("rejection_grounds", []):
                basis = ground.get("basis", "unknown")
                rejection_basis_counts[basis] = rejection_basis_counts.get(basis, 0) + 1

        # Per-claim decisions
        claim_decs = fd.get("claim_decisions", []) if isinstance(fd, dict) else []
        for cd in claim_decs:
            cn = cd.get("claim_number")
            if cn in claim_grants:
                claim_total[cn] += 1
                if cd.get("decision") == "grant":
                    claim_grants[cn] += 1

    # Decision distribution
    summary.grant_count = sum(1 for d in decisions if d == "grant")
    summary.reject_count = sum(1 for d in decisions if d == "reject")
    summary.amend_count = sum(1 for d in decisions if d == "amend_suggested")
    summary.grant_rate = summary.grant_count / max(len(decisions), 1)

    summary.rejection_basis_distribution = rejection_basis_counts

    # Per-claim grant rates
    for cn in claim_grants:
        total = claim_total.get(cn, 0)
        summary.claim_grant_rates[cn] = claim_grants[cn] / max(total, 1)

    # Hotspot claims: high variance (grant rate between 0.3 and 0.7)
    summary.hotspot_claims = [
        cn for cn, rate in summary.claim_grant_rates.items()
        if 0.2 < rate < 0.8
    ]

    # Dimension correlations (simplified: strictness vs decision)
    strict_grant = []
    for trace in traces:
        fd = trace.final_decision
        if not fd:
            continue
        score = fd.get("decision_score", 0.5) if isinstance(fd, dict) else 0.5
        strict_grant.append((trace.examiner.strictness, score))

    if len(strict_grant) >= 3:
        try:
            xs = [s for s, _ in strict_grant]
            ys = [g for _, g in strict_grant]
            n = len(xs)
            mean_x = sum(xs) / n
            mean_y = sum(ys) / n
            cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / n
            std_x = (sum((x - mean_x) ** 2 for x in xs) / n) ** 0.5
            std_y = (sum((y - mean_y) ** 2 for y in ys) / n) ** 0.5
            if std_x > 0 and std_y > 0:
                summary.dimension_correlations["strictness_vs_score"] = round(cov / (std_x * std_y), 3)
        except Exception:
            pass

    # Policy stance correlation
    policy_grant = []
    for trace in traces:
        fd = trace.final_decision
        if not fd:
            continue
        score = fd.get("decision_score", 0.5) if isinstance(fd, dict) else 0.5
        policy_grant.append((trace.examiner.policy_stance, score))

    if len(policy_grant) >= 3:
        try:
            xs = [s for s, _ in policy_grant]
            ys = [g for _, g in policy_grant]
            n = len(xs)
            mean_x = sum(xs) / n
            mean_y = sum(ys) / n
            cov = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys)) / n
            std_x = (sum((x - mean_x) ** 2 for x in xs) / n) ** 0.5
            std_y = (sum((y - mean_y) ** 2 for y in ys) / n) ** 0.5
            if std_x > 0 and std_y > 0:
                summary.dimension_correlations["policy_vs_score"] = round(cov / (std_x * std_y), 3)
        except Exception:
            pass

    return summary


def generate_report(
    case: PatentCase,
    traces: list[ExaminerRunTrace],
    summary: PopulationSummary,
) -> str:
    """Generate a human-readable markdown report."""
    lines = [
        f"# Virtual Patent Office — Examination Report",
        f"",
        f"**Case**: {case.title} (`{case.case_id}`)",
        f"**Era Context**: {summary.era_context or 'None'}",
        f"**Examiners**: {summary.n_examiners}",
        f"**Timestamp**: {summary.timestamp}",
        f"",
        f"---",
        f"",
        f"## Decision Distribution",
        f"",
        f"| Decision | Count | Rate |",
        f"|----------|-------|------|",
        f"| Grant | {summary.grant_count} | {summary.grant_rate:.0%} |",
        f"| Reject | {summary.reject_count} | {summary.reject_count/max(summary.n_examiners,1):.0%} |",
        f"| Amend | {summary.amend_count} | {summary.amend_count/max(summary.n_examiners,1):.0%} |",
        f"",
    ]

    # Rejection basis
    if summary.rejection_basis_distribution:
        lines.extend([
            f"## Rejection Basis Distribution",
            f"",
            f"| Basis | Count |",
            f"|-------|-------|",
        ])
        for basis, count in sorted(summary.rejection_basis_distribution.items(), key=lambda x: -x[1]):
            lines.append(f"| {basis} | {count} |")
        lines.append("")

    # Per-claim
    if summary.claim_grant_rates:
        lines.extend([
            f"## Per-Claim Grant Rates",
            f"",
            f"| Claim | Grant Rate | Status |",
            f"|-------|-----------|--------|",
        ])
        for cn, rate in sorted(summary.claim_grant_rates.items()):
            status = "HOTSPOT" if cn in summary.hotspot_claims else ("high" if rate > 0.7 else "low" if rate < 0.3 else "mixed")
            lines.append(f"| Claim {cn} | {rate:.0%} | {status} |")
        lines.append("")

    # Correlations
    if summary.dimension_correlations:
        lines.extend([
            f"## Dimension Correlations",
            f"",
        ])
        for dim, corr in summary.dimension_correlations.items():
            direction = "↑ positive" if corr > 0.3 else "↓ negative" if corr < -0.3 else "→ weak"
            lines.append(f"- **{dim}**: r = {corr:.3f} ({direction})")
        lines.append("")

    # Individual examiner results
    lines.extend([
        f"## Individual Examiner Results",
        f"",
        f"| Examiner | Archetype | Strictness | Policy | Decision | Score |",
        f"|----------|-----------|-----------|--------|----------|-------|",
    ])
    for trace in traces:
        e = trace.examiner
        fd = trace.final_decision
        dec = fd.get("decision", "?") if isinstance(fd, dict) else "?"
        score = fd.get("decision_score", "?") if isinstance(fd, dict) else "?"
        lines.append(f"| {e.display_name} | {e.archetype_name} | {e.strictness:.2f} | {e.policy_stance:.2f} | {dec} | {score} |")
    lines.append("")

    # Hotspots
    if summary.hotspot_claims:
        lines.extend([
            f"## Disagreement Hotspots",
            f"",
            f"Claims where examiners significantly disagree:",
            f"",
        ])
        for cn in summary.hotspot_claims:
            rate = summary.claim_grant_rates.get(cn, 0)
            lines.append(f"- **Claim {cn}**: grant rate = {rate:.0%} — high variance across examiners")
        lines.append("")

    lines.append("---")
    lines.append(f"*Generated by Virtual Patent Office v0.1*")

    return "\n".join(lines)
