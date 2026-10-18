"""JSON persistence for standalone simulated-student conversations.

The engine itself knows how to create a portable learner snapshot. This module adds the
experiment-facing transcript, model names, validation details, and token-usage records
needed to resume or inspect a CLI conversation.
"""

from __future__ import annotations

import json

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .engine import StudentReply, StudentSnapshot, ValidationStatus
from .perception import StudentObservation
from .policy import PolicyDecision, SimulatedStudentState, StudentResponseMode


class SavedStudentTurn(BaseModel):
    """One visible exchange plus the private evidence behind that student reply."""

    model_config = ConfigDict(extra="forbid")

    tutor_message: str
    student_message: str
    observation: StudentObservation
    decision: PolicyDecision
    state: SimulatedStudentState
    generation_attempts: int = Field(ge=1)
    validation_status: ValidationStatus = ValidationStatus.SKIPPED
    validation_reason: str = ""
    observed_mode: StudentResponseMode | None = None

    @classmethod
    def from_reply(cls, reply: StudentReply) -> SavedStudentTurn:
        """Copy a committed engine reply into the JSON persistence schema."""
        return cls(
            tutor_message=reply.observation.tutor_message,
            student_message=reply.message,
            observation=reply.observation,
            decision=reply.decision,
            state=reply.state,
            generation_attempts=reply.generation_attempts,
            validation_status=reply.validation_status,
            validation_reason=reply.validation_reason,
            observed_mode=reply.observed_mode,
        )


class SavedConversation(BaseModel):
    """Portable transcript, usage log, and the latest resumable learner snapshot.

    ``turns`` is useful for analysis; ``snapshot`` is the authoritative state used to
    continue the learner. Keeping both avoids reconstructing state by replaying model calls.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(1, ge=1)
    saved_at: datetime
    model: str
    perception_model: str = ""
    validator_model: str = ""
    snapshot: StudentSnapshot
    turns: list[SavedStudentTurn] = Field(default_factory=list)
    generation_usage: list[dict[str, Any]] = Field(default_factory=list)
    perception_usage: list[dict[str, Any]] = Field(default_factory=list)
    validation_usage: list[dict[str, Any]] = Field(default_factory=list)


def default_conversation_path(persona_id: str, *, directory: Path | None = None) -> Path:
    """Return a unique default JSON path under data/conversations."""
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    root = directory or Path.cwd() / "data" / "conversations"
    return root / f"{persona_id}-{timestamp}.json"


def save_conversation(path: str | Path, conversation: SavedConversation) -> Path:
    """Atomically write a conversation JSON file."""
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    # Write a complete temporary file first. Replacing the destination in one operation
    # prevents a crash halfway through writing from corrupting the last good checkpoint.
    temporary.write_text(conversation.model_dump_json(indent=2), encoding="utf-8")
    temporary.replace(destination)
    return destination


def load_conversation(path: str | Path) -> SavedConversation:
    """Load and validate a previously saved standalone conversation."""
    source = Path(path).expanduser().resolve()
    # Validate on load instead of trusting arbitrary JSON. This makes incompatible or
    # manually damaged save files fail before a simulator is partially restored.
    payload = json.loads(source.read_text(encoding="utf-8"))
    conversation = SavedConversation.model_validate(payload)
    if conversation.schema_version != 1:
        raise ValueError(f"unsupported conversation schema version: {conversation.schema_version}")
    return conversation


def new_saved_conversation(
    *,
    model: str,
    perception_model: str = "",
    validator_model: str = "",
    snapshot: StudentSnapshot,
    turns: list[SavedStudentTurn],
    generation_usage: list[dict[str, Any]],
    perception_usage: list[dict[str, Any]] | None = None,
    validation_usage: list[dict[str, Any]] | None = None,
) -> SavedConversation:
    """Build a fresh persistence record at the current time."""
    return SavedConversation(
        saved_at=datetime.now(UTC),
        model=model,
        perception_model=perception_model,
        validator_model=validator_model,
        snapshot=snapshot,
        turns=turns,
        generation_usage=generation_usage,
        perception_usage=perception_usage or [],
        validation_usage=validation_usage or [],
    )
