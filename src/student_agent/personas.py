"""Application-neutral persona schema for the simulated student agent."""

from __future__ import annotations

from importlib.resources import files
from pathlib import Path
from typing import Literal

import yaml

from pydantic import BaseModel, ConfigDict, Field

from .controls import StudentControlProfile, StudentPolicyScalars

AnswerStyle = Literal["terse", "concise", "hesitant", "verbose"]
ConfidenceStyle = Literal["low", "calibrated", "overconfident"]
PersistenceStyle = Literal["fragile", "steady", "stubborn"]


class StudentPersona(BaseModel):
    """Reusable behavior profile for a simulated learner.

    Probabilities are intentionally explicit so experiment runs can be compared across
    personas.
    """

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    name: str = ""
    description: str = ""
    control: StudentControlProfile = Field(default_factory=StudentControlProfile)
    answer_style: AnswerStyle = "concise"
    confidence_style: ConfidenceStyle = "calibrated"
    persistence_style: PersistenceStyle = "steady"
    language: str = "match_tutor"
    traits: list[str] = Field(default_factory=list)
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    def control_profile(self) -> StudentControlProfile:
        """Public 0..1 control profile for experiments."""
        return self.control

    def policy_scalars(self) -> StudentPolicyScalars:
        """Lower-level derived values used by the deterministic policy."""
        return self.control.to_policy_scalars()

    def prompt_block(self) -> str:
        """Compact persona text for the LLM instruction."""
        lines = [
            f"id: {self.id}",
            f"name: {self.name or '(unnamed simulated student)'}",
            f"description: {self.description or '(none)'}",
            f"answer_style: {self.answer_style}",
            f"confidence_style: {self.confidence_style}",
            f"persistence_style: {self.persistence_style}",
            f"language: {self.language}",
            f"control_profile: {self.control_profile().prompt_block()}",
        ]
        if self.traits:
            lines.append("traits: " + " | ".join(self.traits))
        return "\n".join(lines)


DEFAULT_PERSONA = StudentPersona(
    id="balanced_student",
    name="Balanced simulated student",
    description=(
        "A realistic learner who sometimes knows the answer, sometimes has a partial "
        "mental model, and asks for clarification when an explanation moves too quickly."
    ),
    control=StudentControlProfile(
        mastery=0.55,
        confidence=0.55,
        learning_rate=0.40,
        misconception_rate=0.20,
        help_seeking=0.15,
        topic_drift=0.03,
    ),
    answer_style="concise",
    confidence_style="calibrated",
    persistence_style="steady",
)


def load_persona(path: str | Path) -> StudentPersona:
    """Load one persona YAML file."""
    return _parse_persona_yaml(Path(path).read_text(encoding="utf-8"), source=str(path))


def available_builtin_personas() -> tuple[str, ...]:
    """Return the IDs of personas distributed with the package."""
    directory = files("student_agent").joinpath("builtin_personas")
    return tuple(sorted(item.name.removesuffix(".yaml") for item in directory.iterdir() if item.name.endswith(".yaml")))


def load_builtin_persona(persona_id: str) -> StudentPersona:
    """Load a packaged persona by ID without relying on repository paths."""
    available = available_builtin_personas()
    if persona_id not in available:
        choices = ", ".join(available)
        raise ValueError(f"unknown built-in persona {persona_id!r}; choose one of: {choices}")
    resource = files("student_agent").joinpath("builtin_personas", f"{persona_id}.yaml")
    return _parse_persona_yaml(resource.read_text(encoding="utf-8"), source=f"built-in persona {persona_id!r}")


def _parse_persona_yaml(text: str, *, source: str) -> StudentPersona:
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise TypeError(f"{source} must contain a YAML mapping")
    return StudentPersona.model_validate(data)
