"""Role 6: Final Decision Synthesizer."""

import logging
from src.roles.base import BaseRole
from src.models.schema import PatentCase
from src.utils.prompts import final_decision_prompt

logger = logging.getLogger("vpatent.roles.final_decision")


class FinalDecisionRole(BaseRole):
    role_name = "final_decision"

    def run(self, case: PatentCase, **prev_outputs) -> dict:
        novelty = prev_outputs.get("novelty_assessments", [])
        obviousness = prev_outputs.get("obviousness_assessments", [])
        office_action = prev_outputs.get("office_action", {})
        system, user = final_decision_prompt(
            self.examiner, case, novelty, obviousness, office_action
        )
        result = self.llm.call_json(system, user, role_name=self.role_name)

        if not result:
            logger.error("FinalDecision: LLM returned None")
            return {
                "decision": "reject",
                "decision_score": 0.0,
                "summary": "LLM unavailable",
                "claim_decisions": [],
                "key_issues": [],
                "examiner_notes": "",
            }

        return result
