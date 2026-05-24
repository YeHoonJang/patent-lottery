"""Role 2: Prior Art Mapping — maps claim elements to prior art."""

import logging
from src.roles.base import BaseRole
from src.models.schema import PatentCase
from src.utils.prompts import prior_art_mapping_prompt

logger = logging.getLogger("vpatent.roles.prior_art")


class PriorArtMappingRole(BaseRole):
    role_name = "prior_art_mapping"

    def run(self, case: PatentCase, **prev_outputs) -> list[dict]:
        interpretations = prev_outputs.get("claim_interpretations", [])
        system, user = prior_art_mapping_prompt(self.examiner, case, interpretations)
        result = self.llm.call_json(system, user, role_name=self.role_name)

        if not result:
            logger.error("PriorArtMapping: LLM returned None")
            return self._fallback(case)

        return result.get("mappings", [])

    def _fallback(self, case: PatentCase) -> list[dict]:
        return [
            {
                "claim_number": c.claim_number,
                "element_mappings": [],
                "closest_ref_id": case.prior_art[0].ref_id if case.prior_art else "",
                "coverage_score": 0.0,
            }
            for c in case.claims
        ]
