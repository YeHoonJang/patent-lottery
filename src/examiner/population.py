"""
Examiner population generator.

Creates examiner instances from archetypes + stochastic noise + era context.
"""

import logging
import random
from pathlib import Path
from typing import Optional

import yaml

from src.models.schema import ExaminerProfile

logger = logging.getLogger("vpatent.population")

# Dimensions that accept stochastic noise (must be float 0~1)
NOISY_DIMENSIONS = [
    "strictness", "policy_stance", "claim_breadth",
    "combination_threshold", "seniority", "tech_specialization",
]


def load_archetype(yaml_path: str) -> dict:
    """Load a single archetype YAML file."""
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def load_era_context(yaml_path: str) -> dict:
    """Load an era context YAML file."""
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def instantiate_examiner(
    archetype: dict,
    era_context: Optional[dict] = None,
    noise_seed: Optional[int] = None,
) -> ExaminerProfile:
    """
    Create one ExaminerProfile from archetype + noise + era modifiers.

    Args:
        archetype: Parsed archetype YAML dict.
        era_context: Parsed era context YAML dict (optional).
        noise_seed: Random seed for stochastic perturbation.
    """
    rng = random.Random(noise_seed)
    base = archetype.get("base_profile", {})
    noise_cfg = archetype.get("noise_config", {})

    # Start from base values
    params = {}
    noise_applied = {}
    for dim in NOISY_DIMENSIONS:
        base_val = base.get(dim, 0.5)
        # Apply noise
        noise_std = noise_cfg.get(dim, 0.05)
        noise_val = rng.gauss(0, noise_std)
        noise_applied[dim] = round(noise_val, 4)
        params[dim] = _clamp(base_val + noise_val)

    # Apply era context modifiers (additive)
    if era_context:
        modifiers = era_context.get("examiner_bias_modifiers", {})
        for key, offset in modifiers.items():
            # Map modifier keys to profile dimensions
            dim_map = {
                "strictness_offset": "strictness",
                "combination_threshold_offset": "combination_threshold",
                "claim_breadth_offset": "claim_breadth",
                "policy_stance_offset": "policy_stance",
                "tech_specialization_offset": "tech_specialization",
            }
            dim = dim_map.get(key)
            if dim and dim in params:
                params[dim] = _clamp(params[dim] + offset)

    # Build profile
    arch_name = archetype.get("archetype_name", "unknown")
    idx = noise_seed if noise_seed is not None else rng.randint(0, 9999)

    profile = ExaminerProfile(
        archetype_name=arch_name,
        display_name=f"{arch_name}_{idx:04d}",
        strictness=round(params["strictness"], 3),
        policy_stance=round(params["policy_stance"], 3),
        claim_breadth=round(params["claim_breadth"], 3),
        combination_threshold=round(params["combination_threshold"], 3),
        tech_specialization=round(params["tech_specialization"], 3),
        seniority=round(params["seniority"], 3),
        oversight_awareness=float(base.get("oversight_awareness", 0.0)),
        tech_domains=base.get("tech_domains", []),
        legal_style=base.get("legal_style", "pragmatist"),
        noise_seed=noise_seed,
        noise_applied=noise_applied,
    )

    # Era context injection
    if era_context:
        profile.era_context_name = era_context.get("name", "")
        profile.era_policy_summary = era_context.get("policy_environment", {}).get("summary", "")

    return profile


def generate_population(
    archetype_dir: str,
    era_context_path: Optional[str] = None,
    instances_per_archetype: int = 2,
    base_seed: int = 42,
    archetype_filter: Optional[list] = None,
    oversight_override: Optional[float] = None,
) -> list:
    """
    Generate a population of examiners from all archetypes in a directory.

    Args:
        archetype_dir: Directory containing archetype YAML files.
        era_context_path: Path to era context YAML (optional).
        instances_per_archetype: Number of instances per archetype.
        base_seed: Starting seed for reproducibility.
        archetype_filter: If provided, only include archetypes whose archetype_name
            (from yaml) is in this list.
        oversight_override: If provided, set oversight_awareness on every examiner
            to this value (overrides yaml).

    Returns:
        List of ExaminerProfile instances.
    """
    archetype_dir = Path(archetype_dir)
    era_context = None
    if era_context_path:
        era_context = load_era_context(era_context_path)

    population = []
    seed = base_seed

    for yaml_file in sorted(archetype_dir.glob("*.yaml")):
        archetype = load_archetype(str(yaml_file))
        arch_name = archetype.get("archetype_name", yaml_file.stem)

        if archetype_filter is not None and arch_name not in archetype_filter:
            continue

        logger.info("Loading archetype: %s", arch_name)

        for i in range(instances_per_archetype):
            examiner = instantiate_examiner(archetype, era_context, noise_seed=seed)
            if oversight_override is not None:
                examiner.oversight_awareness = float(oversight_override)
            population.append(examiner)
            seed += 1

    logger.info("Generated population: %d examiners (filter=%s, oversight=%s)",
                len(population), archetype_filter, oversight_override)
    return population
