"""Base class for all examiner roles."""

import logging
from typing import Optional

from src.models.schema import ExaminerProfile, PatentCase
from src.utils.llm_adapter import LLMAdapter

logger = logging.getLogger("vpatent.roles")


class BaseRole:
    """Base class providing LLM adapter and examiner context."""

    role_name: str = "base"

    def __init__(self, llm: LLMAdapter, examiner: ExaminerProfile):
        self.llm = llm
        self.examiner = examiner

    def run(self, case: PatentCase, **prev_outputs) -> dict:
        """Execute this role. Subclasses must override."""
        raise NotImplementedError
