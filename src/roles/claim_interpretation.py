"""Role 1: Claim Interpretation — BRI-based claim analysis."""

import logging
from src.roles.base import BaseRole
from src.models.schema import (
    PatentCase, ClaimInterpretation, ClaimElement, ExaminerProfile
)
from src.utils.prompts import claim_interpretation_prompt

logger = logging.getLogger("vpatent.roles.claim_interp")


class ClaimInterpreterRole(BaseRole):
    role_name = "claim_interpretation"

    def run(self, case: PatentCase, **prev_outputs) -> list[dict]:
        system, user = claim_interpretation_prompt(self.examiner, case)
        result = self.llm.call_json(system, user, role_name=self.role_name)

        if not result:
            logger.error("ClaimInterpreter: LLM returned None")
            return self._fallback(case)

        return result.get("interpretations", [])

    def _fallback(self, case: PatentCase) -> list[dict]:
        """Minimal fallback if LLM fails."""
        return [
            {
                "claim_number": c.claim_number,
                "interpreted_scope": c.text,
                "key_elements": [{"element_id": f"{c.claim_number}a", "text": c.text}],
                "breadth_assessment": "moderate",
                "ambiguities": [],
            }
            for c in case.claims
        ]
