"""
Virtual Patent Office — CLI Entry Point

Usage:
  python -m src.main --case data/sample_cases/sample_01.json --era configs/era_contexts/pro_innovation_2025.yaml
  python -m src.main --case data/sample_cases/sample_01.json --mode mock --verbose
  python -m src.main --case data/sample_cases/sample_01.json --instances 3 --seed 100
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path

from src.models.schema import PatentCase
from src.examiner.population import generate_population
from src.pipeline.run_population import run_population
from src.utils.llm_adapter import LLMAdapter


def setup_logging(verbose: bool = False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def main():
    parser = argparse.ArgumentParser(
        description="Virtual Patent Office — Examiner Population Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Mock mode (no LLM calls, test architecture)
  python -m src.main --case data/sample_cases/sample_01.json --mode mock

  # With era context
  python -m src.main --case data/sample_cases/sample_01.json --era configs/era_contexts/pro_innovation_2025.yaml

  # Custom population size
  python -m src.main --case data/sample_cases/sample_01.json --instances 3 --seed 100

  # Auto mode (Codex CLI → OpenAI API fallback)
  python -m src.main --case data/sample_cases/sample_01.json --mode auto
        """,
    )

    parser.add_argument("--case", required=True, help="Path to patent case JSON")
    parser.add_argument("--archetypes", default="configs/archetypes",
                        help="Directory containing archetype YAML files")
    parser.add_argument("--era", default=None,
                        help="Path to era context YAML file")
    parser.add_argument("--instances", type=int, default=2,
                        help="Number of examiner instances per archetype (default: 2)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Base random seed for stochastic noise (default: 42)")
    parser.add_argument("--mode", choices=["auto", "codex", "openai", "deepseek", "mock"],
                        default="mock",
                        help="LLM mode (default: mock)")
    parser.add_argument("--model", default="gpt-5.4-mini",
                        help="Codex CLI model (default: gpt-5.4-mini)")
    parser.add_argument("--parallel", type=int, default=1,
                        help="Max parallel examiners (default: 1, try 4 for speed)")
    parser.add_argument("--output_dir", default=None,
                        help="Output directory (default: outputs/<timestamp>)")
    parser.add_argument("--timeout", type=int, default=120,
                        help="LLM call timeout in seconds (default: 120)")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--dry_run", action="store_true",
                        help="Show population without running examination")

    args = parser.parse_args()
    setup_logging(args.verbose)
    logger = logging.getLogger("vpatent.main")

    # ─── Load case ────────────────────────────────────
    case_path = Path(args.case)
    if not case_path.exists():
        logger.error("Case file not found: %s", case_path)
        sys.exit(1)

    with open(case_path, encoding="utf-8") as f:
        case_data = json.load(f)
    case = PatentCase(**case_data)
    logger.info("Loaded case: %s (%d claims, %d prior art refs)",
                case.title, len(case.claims), len(case.prior_art))

    # ─── Generate population ──────────────────────────
    population = generate_population(
        archetype_dir=args.archetypes,
        era_context_path=args.era,
        instances_per_archetype=args.instances,
        base_seed=args.seed,
    )
    logger.info("Population: %d examiners", len(population))

    if args.dry_run:
        print(f"\n{'='*70}")
        print(f"DRY RUN — Population Preview")
        print(f"{'='*70}")
        print(f"Case: {case.title}")
        print(f"Era: {population[0].era_context_name if population else 'None'}")
        print(f"\n{'Examiner':<35} {'Archetype':<25} {'Strict':>7} {'Policy':>7} {'Combo':>7} {'Breadth':>7}")
        print("-" * 95)
        for e in population:
            print(f"{e.display_name:<35} {e.archetype_name:<25} {e.strictness:>7.2f} {e.policy_stance:>7.2f} {e.combination_threshold:>7.2f} {e.claim_breadth:>7.2f}")
        print(f"\nTo run: remove --dry_run flag")
        return

    # ─── Setup output ─────────────────────────────────
    if args.output_dir:
        output_dir = args.output_dir
    else:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        era_tag = Path(args.era).stem if args.era else "no_era"
        output_dir = f"outputs/run_{ts}_{era_tag}"

    logger.info("Output directory: %s", output_dir)

    # ─── Run ──────────────────────────────────────────
    llm = LLMAdapter(mode=args.mode, model=args.model, timeout=args.timeout, max_parallel=args.parallel)
    traces, summary = run_population(
        case=case,
        population=population,
        llm=llm,
        output_dir=output_dir,
        verbose=args.verbose,
    )

    # ─── Print summary ────────────────────────────────
    print(f"\n{'='*70}")
    print(f"EXAMINATION COMPLETE")
    print(f"{'='*70}")
    print(f"Case: {case.title}")
    print(f"Era: {summary.era_context or 'None'}")
    print(f"Examiners: {summary.n_examiners}")
    print(f"")
    print(f"Decision Distribution:")
    print(f"  Grant:  {summary.grant_count} ({summary.grant_rate:.0%})")
    print(f"  Reject: {summary.reject_count} ({summary.reject_count/max(summary.n_examiners,1):.0%})")
    print(f"  Amend:  {summary.amend_count} ({summary.amend_count/max(summary.n_examiners,1):.0%})")
    print(f"")

    if summary.hotspot_claims:
        print(f"Disagreement Hotspots: Claims {summary.hotspot_claims}")

    if summary.dimension_correlations:
        print(f"\nDimension Correlations:")
        for dim, corr in summary.dimension_correlations.items():
            print(f"  {dim}: r = {corr:.3f}")

    print(f"\nResults saved to: {output_dir}/")
    print(f"  - report.md (readable summary)")
    print(f"  - population_summary.json")
    print(f"  - examiners/<run_id>/ (per-examiner traces)")


if __name__ == "__main__":
    main()
