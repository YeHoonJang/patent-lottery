"""Role 4: Obviousness Analysis — 35 USC 103 (Graham factors)."""

import logging
from src.roles.base import BaseRole
from src.models.schema import PatentCase
from src.utils.prompts import obviousness_prompt

logger = logging.getLogger("vpatent.roles.obviousness")


class ObviousnessAnalyzerRole(BaseRole):
    role_name = "obviousness_analysis"

    def run(self, case: PatentCase, **prev_outputs) -> list[dict]:
        mappings = prev_outputs.get("prior_art_mappings", [])
        novelty = prev_outputs.get("novelty_assessments", [])
        system, user = obviousness_prompt(self.examiner, case, mappings, novelty)
        result = self.llm.call_json(system, user, role_name=self.role_name)

        if not result:
            logger.error("ObviousnessAnalyzer: LLM returned None")
            return self._fallback(case)

        return result.get("assessments", [])

    def _fallback(self, case: PatentCase) -> list[dict]:
        return [
            {
                "claim_number": c.claim_number,
                "is_obvious": False,
                "reasoning": "LLM unavailable — default to non-obvious",
                "combination_refs": [],
                "motivation_to_combine": "",
                "graham_factors": {},
                "confidence": 0.0,
            }
            for c in case.claims
        ]
