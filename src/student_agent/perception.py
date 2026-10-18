"""Student-owned interpretation of visible tutor messages.

Perception is the boundary between free-form tutor text and the deterministic student
policy. A perception adapter may misunderstand ambiguous wording, just as a learner can,
but it must never receive an authoritative grade or other hidden tutor state.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum
from typing import TYPE_CHECKING, Protocol

from pydantic import BaseModel, ConfigDict, Field

from .conversation import ConversationMessage

if TYPE_CHECKING:
    from .policy import SimulatedStudentState


class MessageType(StrEnum):
    """The conversational job the latest tutor message appears to perform."""

    QUESTION = "question"
    FEEDBACK = "feedback"
    FEEDBACK_AND_QUESTION = "feedback_and_question"
    EXPLANATION = "explanation"
    HINT = "hint"
    INSTRUCTION = "instruction"
    OTHER = "other"


class PerceivedFeedback(StrEnum):
    """How the learner interprets feedback about its preceding answer."""

    CORRECT = "correct"
    PARTIAL = "partial"
    INCORRECT = "incorrect"
    NONE = "none"


class StudentObservation(BaseModel):
    """The learner's structured, fallible reading of one tutor message.

    An observation is temporary turn input. The engine passes it to the state update and
    policy, which decide what the learner remembers and how it responds. Topic and task
    keys belong to the learner's own registry; they are not course-database identifiers.
    """

    model_config = ConfigDict(extra="forbid")

    # What the learner thinks the conversation is currently about. Human-readable labels
    # are kept separately from stable local keys so paraphrases can reuse the same memory.
    tutor_message: str = Field(min_length=1)
    perceived_topic: str = "general discussion"
    topic_key: str = "general"
    perceived_task: str = "general task"
    task_key: str = "general-task"

    # The learner's interpretation of the tutor's intent and assessment.
    message_type: MessageType = MessageType.OTHER
    perceived_feedback: PerceivedFeedback = PerceivedFeedback.NONE
    feedback_confidence: float = Field(0.0, ge=0.0, le=1.0)
    correction_present: bool = False
    explanation_present: bool = False
    demonstrated_understanding: bool = False

    # Content extracted from the visible exchange. These fields let state track taught
    # propositions and specific mistaken beliefs without consulting an answer key.
    knowledge_components: list[str] = Field(default_factory=list)
    misconception_statement: str = ""
    matched_misconception_id: str = ""
    evidence: str = ""

    # Operational metadata records how the observation was produced and where this turn
    # sits in the learner's own conversation. It is useful for debugging and evaluation.
    perception_source: str = "semantic"
    perception_error: str = ""
    response_expected: bool = False
    perceived_difficulty: float = Field(0.5, ge=0.0, le=1.0)
    mentioned_terms: list[str] = Field(default_factory=list)
    turn_number: int = Field(1, ge=1)
    attempt_number: int = Field(0, ge=0)
    experiment_seed: str = ""


class StudentPerception(Protocol):
    """Semantically interpret visible conversation without tutor internals."""

    async def observe(
        self,
        *,
        tutor_message: str,
        history: Sequence[ConversationMessage],
        state: SimulatedStudentState,
    ) -> StudentObservation:
        """Return a private interpretation using only the supplied student-owned data."""
