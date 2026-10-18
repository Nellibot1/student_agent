"""Provider-neutral entry point for natural-language persona generation.

This module owns validation and identity guarantees. A provider integration owns the model
call. The resulting persona is ordinary data and can be saved once, reviewed by a human,
and reused without asking a model to reinterpret the description on every experiment run.
"""

from __future__ import annotations

from typing import Protocol

from .personas import StudentPersona


class PersonaGenerator(Protocol):
    """Interpret a free-form learner description as a concrete persona."""

    async def generate(
        self,
        *,
        description: str,
        persona_id: str,
        name: str,
        language: str,
    ) -> StudentPersona:
        """Return one validated, fully materialized persona."""


async def generate_persona_from_description(
    description: str,
    *,
    generator: PersonaGenerator,
    persona_id: str = "generated_student",
    name: str = "",
    language: str = "match_tutor",
) -> StudentPersona:
    """Convert a natural-language description into reusable behavior controls.

    The model/provider boundary is injected so the core package remains independent of
    LiteLLM or any other SDK. The result is a normal :class:`StudentPersona` and should be
    saved and reused when reproducible experiment runs are required.
    """
    source = description.strip()
    if not source:
        raise ValueError("description cannot be empty")
    if not language.strip():
        raise ValueError("language cannot be empty")

    # Validate caller-owned identity fields before spending a model request. Validate them
    # again afterward so a provider adapter cannot silently change the requested identity.
    StudentPersona(id=persona_id, name=name, language=language)
    persona = await generator.generate(
        description=source,
        persona_id=persona_id,
        name=name,
        language=language,
    )
    if not isinstance(persona, StudentPersona):
        raise TypeError("persona generator must return StudentPersona")
    if persona.id != persona_id:
        raise ValueError("persona generator changed the requested persona_id")
    if name and persona.name != name:
        raise ValueError("persona generator changed the requested name")
    if persona.language != language:
        raise ValueError("persona generator changed the requested language")
    return persona
