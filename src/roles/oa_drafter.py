"""Role 5: Office Action Drafter."""

import logging
from src.roles.base import BaseRole
from src.models.schema import PatentCase
from src.utils.prompts import office_action_prompt

logger = logging.getLogger("vpatent.roles.oa_drafter")


class OADrafterRole(BaseRole):
    role_name = "oa_drafter"

    def run(self, case: PatentCase, **prev_outputs) -> dict:
        novelty = prev_outputs.get("novelty_assessments", [])
        obviousness = prev_outputs.get("obviousness_assessments", [])
        written_desc = prev_outputs.get("written_desc_assessments", [])
        system, user = office_action_prompt(self.examiner, case, novelty, obviousness, written_desc)
        result = self.llm.call_json(system, user, role_name=self.role_name)

        if not result:
            logger.error("OADrafter: LLM returned None")
            return {"rejection_grounds": [], "objections": [], "allowable_claims": [], "full_text": ""}

        return result
