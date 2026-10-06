"""Visible conversation types owned by the simulated student."""

from __future__ import annotations

from collections.abc import Sequence
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ConversationRole(StrEnum):
    TUTOR = "tutor"
    STUDENT = "student"


class ConversationMessage(BaseModel):
    """One message visible to the student."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    role: ConversationRole
    content: str = Field(min_length=1)


def render_history(history: Sequence[ConversationMessage], *, limit: int = 8) -> str:
    """Render recent visible messages for a generation prompt."""
    recent = history[-limit:] if limit > 0 else history
    if not recent:
        return "(no earlier visible messages)"
    return "\n".join(f"{message.role.value}: {message.content}" for message in recent)
