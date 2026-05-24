"""Role 7: Rejection Verifier — validates each rejection ground has concrete evidence.

This implements the Chain-of-Verification (CoVe) pattern:
1. OA Drafter generates rejections
2. Verifier independently checks each rejection for grounding
3. Ungrounded rejections are removed
"""

import logging
from src.roles.base import BaseRole
from src.models.schema import PatentCase
from src.utils.prompts import verification_prompt

logger = logging.getLogger("vpatent.roles.verifier")


class RejectionVerifierRole(BaseRole):
    role_name = "rejection_verifier"

    def run(self, case: PatentCase, **prev_outputs) -> dict:
        office_action = prev_outputs.get("office_action", {})

        # Skip verification if no rejections
        if not office_action.get("rejection_grounds"):
            return {
                "verified_rejections": [],
                "removed_rejections": [],
                "verified_allowable_claims": office_action.get("allowable_claims", []),
                "verification_summary": "No rejections to verify",
            }

        system, user = verification_prompt(self.examiner, case, office_action)
        result = self.llm.call_json(system, user, role_name=self.role_name)

        if not result:
            logger.error("RejectionVerifier: LLM returned None, keeping original OA")
            return {
                "verified_rejections": office_action.get("rejection_grounds", []),
                "removed_rejections": [],
                "verified_allowable_claims": office_action.get("allowable_claims", []),
                "verification_summary": "Verification failed, original OA kept",
            }

        # Update the office_action with verified results
        verified_oa = dict(office_action)
        if "verified_rejections" in result:
            verified_oa["rejection_grounds"] = result["verified_rejections"]
        if "verified_allowable_claims" in result:
            verified_oa["allowable_claims"] = result["verified_allowable_claims"]

        # Store verification metadata
        result["original_rejection_count"] = len(office_action.get("rejection_grounds", []))
        result["verified_rejection_count"] = len(result.get("verified_rejections", []))
        result["removed_count"] = len(result.get("removed_rejections", []))

        n_orig = result["original_rejection_count"]
        n_verified = result["verified_rejection_count"]
        n_removed = result["removed_count"]
        logger.info("Verification: %d -> %d rejections (%d removed)",
                     n_orig, n_verified, n_removed)

        return result
