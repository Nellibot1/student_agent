"""Messages that belong to the student's visible conversation.

Only text represented by this module may be treated as conversation history. Private
tutor data—such as grades, answer keys, or lesson-plan state—must not be converted into a
``ConversationMessage`` just to make it available to the simulator.
"""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ConversationRole(StrEnum):
    """The two speakers whose messages a simulated learner is allowed to see."""

    TUTOR = "tutor"
    STUDENT = "student"


class ConversationMessage(BaseModel):
    """One message visible to the student."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: ConversationRole
    content: str = Field(min_length=1)


def render_history(history: Sequence[ConversationMessage], *, limit: int = 8) -> str:
    """Render the newest visible messages as a compact prompt block.

    ``history`` remains complete in the simulator. The limit controls only how much is
    copied into this particular model prompt, which prevents prompts from growing forever.
    A non-positive limit intentionally renders the complete supplied sequence.
    """
    recent = history[-limit:] if limit > 0 else history
    if not recent:
        return "(no earlier visible messages)"
    return "\n".join(f"{message.role.value}: {message.content}" for message in recent)
