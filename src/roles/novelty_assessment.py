"""Role 3: Novelty Assessment — 35 USC 102."""

import logging
from src.roles.base import BaseRole
from src.models.schema import PatentCase
from src.utils.prompts import novelty_prompt

logger = logging.getLogger("vpatent.roles.novelty")


class NoveltyAssessorRole(BaseRole):
    role_name = "novelty_assessment"

    def run(self, case: PatentCase, **prev_outputs) -> list[dict]:
        mappings = prev_outputs.get("prior_art_mappings", [])
        system, user = novelty_prompt(self.examiner, case, mappings)
        result = self.llm.call_json(system, user, role_name=self.role_name)

        if not result:
            logger.error("NoveltyAssessor: LLM returned None")
            return self._fallback(case)

        return result.get("assessments", [])

    def _fallback(self, case: PatentCase) -> list[dict]:
        return [
            {
                "claim_number": c.claim_number,
                "is_novel": True,
                "reasoning": "LLM unavailable — default to novel",
                "anticipating_refs": [],
                "confidence": 0.0,
            }
            for c in case.claims
        ]
