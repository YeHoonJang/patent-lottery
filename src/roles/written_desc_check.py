"""Role 5b: Written Description / Enablement Check — 35 USC 112."""

import logging
from src.roles.base import BaseRole
from src.models.schema import PatentCase
from src.utils.prompts import written_desc_prompt

logger = logging.getLogger("vpatent.roles.written_desc")


class WrittenDescCheckRole(BaseRole):
    role_name = "written_desc_check"

    def run(self, case: PatentCase, **prev_outputs) -> list[dict]:
        interpretations = prev_outputs.get("claim_interpretations", [])
        spec_retriever = prev_outputs.get("spec_retriever")

        # If we have a spec retriever, provide structured evidence per element
        spec_context = ""
        if spec_retriever and interpretations:
            all_elements = []
            for interp in interpretations:
                for elem in interp.get("key_elements", []):
                    all_elements.append(elem)
            if all_elements:
                spec_context = spec_retriever.build_context_for_role(
                    "written_desc_check", claim_elements=all_elements
                )

        system, user = written_desc_prompt(
            self.examiner, case, interpretations, spec_context=spec_context
        )
        result = self.llm.call_json(system, user, role_name=self.role_name)

        if not result:
            logger.error("WrittenDescCheck: LLM returned None")
            return self._fallback(case)

        return result.get("assessments", [])

    def _fallback(self, case: PatentCase) -> list[dict]:
        return [
            {
                "claim_number": c.claim_number,
                "has_issues": False,
                "issues": [],
                "unsupported_terms": [],
                "enablement_concerns": [],
                "reasoning": "LLM unavailable — default to no issues",
                "confidence": 0.0,
            }
            for c in case.claims
        ]
