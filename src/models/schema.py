"""
Core data models for Virtual Patent Office.

All structured data flows through these Pydantic schemas.
Every intermediate result is serializable to JSON for inspection.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ─── Enums ───────────────────────────────────────────────

class DecisionLabel(str, Enum):
    GRANT = "grant"
    REJECT = "reject"
    AMEND = "amend_suggested"


class RejectionBasis(str, Enum):
    SEC_102 = "102_novelty"
    SEC_103 = "103_obviousness"
    SEC_112 = "112_written_description"
    SEC_101 = "101_eligibility"


class ClaimType(str, Enum):
    INDEPENDENT = "independent"
    DEPENDENT = "dependent"


# ─── Patent Case Input ───────────────────────────────────

class ClaimElement(BaseModel):
    element_id: str = ""
    text: str
    is_novel: Optional[bool] = None
    mapped_prior_art: list[str] = Field(default_factory=list)


class Claim(BaseModel):
    claim_number: int
    claim_type: ClaimType = ClaimType.INDEPENDENT
    text: str
    depends_on: Optional[int] = None
    elements: list[ClaimElement] = Field(default_factory=list)


class PriorArtReference(BaseModel):
    ref_id: str
    title: str
    abstract: str = ""
    relevant_text: str = ""
    claims_text: str = ""        # Full claims text from patent
    description_text: str = ""   # Full description text from patent
    publication_date: str = ""
    source: str = ""  # USPTO, KIPRIS, CNIPA, NPL


class PatentCase(BaseModel):
    case_id: str = Field(default_factory=lambda: f"case_{uuid.uuid4().hex[:8]}")
    title: str
    abstract: str = ""
    description_summary: str = ""
    description_full: str = ""     # Full specification text for SpecRetriever
    claims: list[Claim]
    prior_art: list[PriorArtReference] = Field(default_factory=list)
    ipc_codes: list[str] = Field(default_factory=list)
    applicant: str = ""
    filing_date: str = ""
    jurisdiction: str = "USPTO"


# ─── Examiner Profile ────────────────────────────────────

class ExaminerProfile(BaseModel):
    examiner_id: str = Field(default_factory=lambda: f"ex_{uuid.uuid4().hex[:6]}")
    archetype_name: str
    display_name: str = ""

    # Core dimensions (0.0 ~ 1.0)
    strictness: float = 0.5
    policy_stance: float = 0.5       # 0=conservative, 1=innovation-friendly
    claim_breadth: float = 0.5       # 0=narrow, 1=broad BRI
    combination_threshold: float = 0.5  # lower → easier to find obvious
    tech_specialization: float = 0.5  # 0=generalist, 1=domain expert
    seniority: float = 0.5           # 0=junior, 1=senior (time pressure)
    oversight_awareness: float = 0.0  # 0=none, 1=full SPE+OPQA+asymmetric risk

    # Categorical
    tech_domains: list[str] = Field(default_factory=list)
    legal_style: str = "pragmatist"  # textualist, purposivist, pragmatist

    # Era context (injected at runtime)
    era_context_name: str = ""
    era_policy_summary: str = ""

    # Noise tracking
    noise_seed: Optional[int] = None
    noise_applied: dict[str, float] = Field(default_factory=dict)

    # Persona baseline (if set, overrides full persona block)
    persona_label: str = ""


# ─── Role Outputs ────────────────────────────────────────
# Ordered so that referenced types are defined before use.

class ClaimInterpretation(BaseModel):
    claim_number: int
    interpreted_scope: str = ""
    key_elements: list[ClaimElement] = Field(default_factory=list)
    breadth_assessment: str = ""  # narrow / moderate / broad
    ambiguities: list[str] = Field(default_factory=list)


class RefMatch(BaseModel):
    ref_id: str
    relevant_passage: str = ""
    similarity_score: float = 0.0
    match_type: str = ""  # explicit, implicit, obvious_variant


class ElementMapping(BaseModel):
    element_id: str
    element_text: str
    matched_refs: list[RefMatch] = Field(default_factory=list)
    is_covered: bool = False


class PriorArtMapping(BaseModel):
    claim_number: int
    mappings: list[ElementMapping] = Field(default_factory=list)
    closest_ref_id: str = ""
    coverage_score: float = 0.0  # 0~1


class NoveltyAssessment(BaseModel):
    claim_number: int
    is_novel: bool = True
    basis: Optional[RejectionBasis] = None
    reasoning: str = ""
    anticipating_refs: list[str] = Field(default_factory=list)
    confidence: float = 0.5


class ObviousnessAssessment(BaseModel):
    claim_number: int
    is_obvious: bool = False
    basis: Optional[RejectionBasis] = None
    reasoning: str = ""
    combination_refs: list[str] = Field(default_factory=list)
    motivation_to_combine: str = ""
    graham_factors: dict[str, str] = Field(default_factory=dict)
    confidence: float = 0.5




class WrittenDescAssessment(BaseModel):
    claim_number: int
    has_issues: bool = False
    basis: Optional[RejectionBasis] = None  # SEC_112
    issues: list[str] = Field(default_factory=list)
    unsupported_terms: list[str] = Field(default_factory=list)
    enablement_concerns: list[str] = Field(default_factory=list)
    reasoning: str = ""
    confidence: float = 0.5

class RejectionGround(BaseModel):
    claim_numbers: list[int]
    basis: RejectionBasis
    reasoning: str
    cited_refs: list[str] = Field(default_factory=list)


class OfficeAction(BaseModel):
    rejection_grounds: list[RejectionGround] = Field(default_factory=list)
    objections: list[str] = Field(default_factory=list)
    allowable_claims: list[int] = Field(default_factory=list)
    full_text: str = ""


class ClaimDecision(BaseModel):
    claim_number: int
    decision: DecisionLabel
    primary_basis: Optional[RejectionBasis] = None
    reasoning: str = ""


class FinalDecision(BaseModel):
    decision: DecisionLabel
    decision_score: float = 0.5  # 0=strong reject, 1=strong grant
    summary: str = ""
    claim_decisions: list[ClaimDecision] = Field(default_factory=list)
    key_issues: list[str] = Field(default_factory=list)
    examiner_notes: str = ""


# ─── Run Trace ────────────────────────────────────────────

class ExaminerRunTrace(BaseModel):
    run_id: str = Field(default_factory=lambda: f"run_{uuid.uuid4().hex[:8]}")
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())
    case_id: str
    examiner: ExaminerProfile

    claim_interpretations: list[ClaimInterpretation] = Field(default_factory=list)
    prior_art_mappings: list[PriorArtMapping] = Field(default_factory=list)
    novelty_assessments: list[NoveltyAssessment] = Field(default_factory=list)
    obviousness_assessments: list[ObviousnessAssessment] = Field(default_factory=list)
    written_desc_assessments: list[WrittenDescAssessment] = Field(default_factory=list)
    office_action: Optional[OfficeAction] = None
    final_decision: Optional[FinalDecision] = None

    llm_calls: int = 0
    total_tokens: int = 0
    elapsed_seconds: float = 0.0
    errors: list[str] = Field(default_factory=list)


# ─── Population Summary ──────────────────────────────────

class PopulationSummary(BaseModel):
    case_id: str
    era_context: str = ""
    n_examiners: int = 0
    timestamp: str = Field(default_factory=lambda: datetime.now().isoformat())

    grant_count: int = 0
    reject_count: int = 0
    amend_count: int = 0
    grant_rate: float = 0.0

    rejection_basis_distribution: dict[str, int] = Field(default_factory=dict)
    claim_grant_rates: dict[int, float] = Field(default_factory=dict)
    hotspot_claims: list[int] = Field(default_factory=list)
    dimension_correlations: dict[str, float] = Field(default_factory=dict)
    examiner_run_ids: list[str] = Field(default_factory=list)
