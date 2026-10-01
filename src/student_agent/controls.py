"""Normalized experimental controls for simulated students.

These are the public knobs for experiments: each value is in [0, 1] and maps to the
lower-level policy scalars used by the simulator.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class StudentPolicyScalars(BaseModel):
    """Lower-level probabilities/rates consumed by the deterministic policy."""

    model_config = ConfigDict(extra="forbid")

    correct_probability: float = Field(0.55, ge=0.0, le=1.0)
    partial_probability: float = Field(0.25, ge=0.0, le=1.0)
    learning_gain: float = Field(0.08, ge=0.0, le=1.0)
    misconception_probability: float = Field(0.20, ge=0.0, le=1.0)
    clarification_probability: float = Field(0.15, ge=0.0, le=1.0)
    topic_jump_probability: float = Field(0.03, ge=0.0, le=1.0)


class StudentControlProfile(BaseModel):
    """
    All values are normalized to [0, 1]:
    - mastery: baseline concept knowledge
    - confidence: how certain the student sounds/feels
    - learning_rate: how quickly feedback improves future behavior
    - misconception_rate: how likely wrong/partial answers are misconception-shaped
    - help_seeking: tendency to ask clarification questions
    - topic_drift: tendency to jump to nearby topics
    """

    model_config = ConfigDict(extra="forbid")

    mastery: float = Field(0.55, ge=0.0, le=1.0)
    confidence: float = Field(0.55, ge=0.0, le=1.0)
    learning_rate: float = Field(0.40, ge=0.0, le=1.0)
    misconception_rate: float = Field(0.20, ge=0.0, le=1.0)
    help_seeking: float = Field(0.15, ge=0.0, le=1.0)
    topic_drift: float = Field(0.03, ge=0.0, le=1.0)

    def to_policy_scalars(self) -> StudentPolicyScalars:
        """Translate into the simulator's current probability knobs."""
        partial_probability = _clamp01(0.10 + (1.0 - self.mastery) * 0.35 + self.misconception_rate * 0.15)
        return StudentPolicyScalars(
            correct_probability=self.mastery,
            partial_probability=partial_probability,
            learning_gain=0.02 + self.learning_rate * 0.18,
            misconception_probability=self.misconception_rate,
            clarification_probability=self.help_seeking,
            topic_jump_probability=self.topic_drift,
        )

    def prompt_block(self) -> str:
        return (
            f"mastery={self.mastery:.2f}, "
            f"confidence={self.confidence:.2f}, "
            f"learning_rate={self.learning_rate:.2f}, "
            f"misconception_rate={self.misconception_rate:.2f}, "
            f"help_seeking={self.help_seeking:.2f}, "
            f"topic_drift={self.topic_drift:.2f}"
        )


def _clamp01(x: float) -> float:
    return min(1.0, max(0.0, x))
