"""Reusable learner profiles and the helpers that load or save them.

A persona describes stable starting tendencies, while ``SimulatedStudentState`` changes
during a conversation. Keeping those concepts separate allows the same persona to start
many independent and reproducible learner sessions.
"""

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
LearningPurpose = Literal["realistic", "tutor_stress_test"]


class LearningDynamics(BaseModel):
    """How quickly repeated correction changes the learner's behavior.

    ``purpose`` keeps deliberately adversarial tutor stress tests distinguishable from
    personas intended to approximate ordinary learning.  Resistance controls how much
    misconception repetition remains after the configured adaptation threshold.
    """

    model_config = ConfigDict(extra="forbid")

    # ``realistic`` favors adaptation; ``tutor_stress_test`` may deliberately retain a
    # misconception longer in order to exercise the tutor's remediation behavior.
    purpose: LearningPurpose = "realistic"
    misconception_resistance: float = Field(0.65, ge=0.0, le=1.0)
    corrections_before_adaptation: int = Field(3, ge=0, le=50)


class StudentPersona(BaseModel):
    """Reusable behavior profile for a simulated learner.

    Probabilities are intentionally explicit so experiment runs can be compared across
    personas.
    """

    model_config = ConfigDict(extra="forbid")

    # Identity and description are for humans, saved files, and experiment reporting.
    id: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]*$")
    name: str = ""
    description: str = ""

    # Controls influence policy probabilities. Style fields influence the surface wording
    # produced after the policy has already selected a response behavior.
    control: StudentControlProfile = Field(default_factory=StudentControlProfile)
    answer_style: AnswerStyle = "concise"
    confidence_style: ConfidenceStyle = "calibrated"
    persistence_style: PersistenceStyle = "steady"
    learning_dynamics: LearningDynamics = Field(default_factory=LearningDynamics)
    language: str = "match_tutor"
    # Applications may attach serializable labels here. Core policy code does not read
    # metadata, so it cannot silently turn application-specific fields into behavior.
    metadata: dict[str, str | int | float | bool | None] = Field(default_factory=dict)

    def control_profile(self) -> StudentControlProfile:
        """Public 0..1 control profile for experiments."""
        return self.control

    def policy_scalars(self) -> StudentPolicyScalars:
        """Lower-level derived values used by the deterministic policy."""
        return self.control.to_policy_scalars()

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
        motivation=0.53,
        misconception_rate=0.20,
        help_seeking=0.15,
        topic_drift=0.03,
    ),
    answer_style="concise",
    confidence_style="calibrated",
    persistence_style="steady",
)


def load_persona(path: str | Path) -> StudentPersona:
    """Load and validate one user-provided persona YAML file."""
    return _parse_persona_yaml(Path(path).read_text(encoding="utf-8"), source=str(path))


def save_persona(path: str | Path, persona: StudentPersona, *, overwrite: bool = False) -> Path:
    """Save a fully materialized persona as reusable, human-readable YAML."""
    destination = Path(path).expanduser()
    if destination.exists() and not overwrite:
        raise FileExistsError(f"persona file already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = yaml.safe_dump(
        persona.model_dump(mode="python"),
        sort_keys=False,
        allow_unicode=True,
    )
    destination.write_text(payload, encoding="utf-8")
    return destination


def available_builtin_personas() -> tuple[str, ...]:
    """Return the IDs of personas distributed with the package."""
    # ``importlib.resources`` also works after the package is installed in a wheel, unlike
    # paths calculated relative to a source checkout.
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
    """Parse YAML and let Pydantic report missing, unknown, or invalid fields."""
    data = yaml.safe_load(text)
    if not isinstance(data, dict):
        raise TypeError(f"{source} must contain a YAML mapping")
    return StudentPersona.model_validate(data)
