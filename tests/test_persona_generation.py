from __future__ import annotations

import inspect
import json

from argparse import Namespace

import pytest

from student_agent import (
    PersonaGenerationError,
    StudentControlProfile,
    StudentPersona,
    generate_persona_from_description,
    load_persona,
    save_persona,
)
from student_agent.cli import build_parser, run_persona_generation
from student_agent.integrations import DEFAULT_STUDENT_GENERATION_MAX_TOKENS, LiteLLMPersonaGenerator
from student_agent.integrations.litellm import LiteLLMStudentGenerator
from student_agent.policy import initial_simulated_state


def generated_payload(**control_overrides: float) -> str:
    control = {
        "mastery": 0.55,
        "confidence": 0.42,
        "learning_rate": 0.48,
        "motivation": 0.88,
        "misconception_rate": 0.20,
        "help_seeking": 0.32,
        "topic_drift": 0.02,
    }
    control.update(control_overrides)
    return json.dumps(
        {
            "name": "Cautious engaged learner",
            "description": "An engaged learner who checks their reasoning before answering.",
            "control": control,
            "answer_style": "hesitant",
            "confidence_style": "calibrated",
            "persistence_style": "steady",
        }
    )


@pytest.mark.asyncio
async def test_litellm_persona_generator_maps_description_to_validated_controls() -> None:
    requests: list[dict] = []

    async def completion(**request):
        requests.append(request)
        return {
            "model": "test/model",
            "choices": [{"message": {"content": generated_payload()}}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 30, "total_tokens": 70},
        }

    adapter = LiteLLMPersonaGenerator("test/model", completion=completion)
    persona = await generate_persona_from_description(
        "This student is very interested but cautious and eager to learn.",
        generator=adapter,
        persona_id="cautious_learner",
    )

    assert persona.id == "cautious_learner"
    assert persona.control.motivation == 0.88
    assert persona.control.mastery == 0.55
    assert persona.answer_style == "hesitant"
    assert persona.metadata["source_description"].startswith("This student")
    assert requests[0]["response_format"] == {"type": "json_object"}
    assert "motivation: willingness" in requests[0]["messages"][0]["content"]
    assert json.loads(requests[0]["messages"][1]["content"])["learner_description"].endswith("learn.")
    assert adapter.usage[0].operation == "persona_generation"


@pytest.mark.asyncio
async def test_persona_generator_retries_schema_failure() -> None:
    calls = 0

    async def completion(**request):
        nonlocal calls
        calls += 1
        content = generated_payload(motivation=1.4) if calls == 1 else generated_payload()
        return {"choices": [{"message": {"content": content}}]}

    adapter = LiteLLMPersonaGenerator("test/model", completion=completion)
    persona = await adapter.generate(
        description="Interested but cautious.",
        persona_id="retry_student",
        name="",
        language="match_tutor",
    )

    assert calls == 2
    assert persona.control.motivation == 0.88
    assert adapter.last_error == ""


@pytest.mark.asyncio
async def test_persona_generator_reports_exhausted_validation() -> None:
    async def completion(**request):
        del request
        return {"choices": [{"message": {"content": generated_payload(mastery=-0.5)}}]}

    adapter = LiteLLMPersonaGenerator("test/model", max_attempts=1, completion=completion)
    with pytest.raises(PersonaGenerationError, match="failed after 1 attempts"):
        await adapter.generate(
            description="An impossible generated value should fail.",
            persona_id="invalid_student",
            name="",
            language="match_tutor",
        )


@pytest.mark.asyncio
async def test_provider_neutral_function_validates_identity_before_calling_generator() -> None:
    class NeverCalled:
        async def generate(self, **kwargs):
            raise AssertionError(kwargs)

    with pytest.raises(ValueError, match="description cannot be empty"):
        await generate_persona_from_description("   ", generator=NeverCalled())
    with pytest.raises(ValueError, match="id"):
        await generate_persona_from_description("cautious", generator=NeverCalled(), persona_id="Not Valid")


def test_motivation_is_an_independent_control() -> None:
    eager = StudentPersona(
        id="eager",
        control=StudentControlProfile(learning_rate=0.20, motivation=0.90),
    )
    quick_but_disengaged = StudentPersona(
        id="quick_but_disengaged",
        control=StudentControlProfile(learning_rate=0.90, motivation=0.10),
    )

    assert initial_simulated_state(eager).motivation == 0.90
    assert initial_simulated_state(quick_but_disengaged).motivation == 0.10
    assert eager.policy_scalars().learning_gain < quick_but_disengaged.policy_scalars().learning_gain


def test_persona_yaml_round_trip_does_not_overwrite_by_default(tmp_path) -> None:
    path = tmp_path / "personas" / "cautious.yaml"
    persona = StudentPersona(id="cautious", control=StudentControlProfile(motivation=0.8))

    assert save_persona(path, persona) == path
    assert load_persona(path) == persona
    with pytest.raises(FileExistsError):
        save_persona(path, persona)


def test_reasoning_safe_generation_budget_is_the_system_default() -> None:
    constructor_default = inspect.signature(LiteLLMStudentGenerator).parameters["max_tokens"].default
    cli_default = build_parser().parse_args(["chat", "--model", "test/model"]).max_tokens

    assert DEFAULT_STUDENT_GENERATION_MAX_TOKENS == 1000
    assert constructor_default == DEFAULT_STUDENT_GENERATION_MAX_TOKENS
    assert cli_default == DEFAULT_STUDENT_GENERATION_MAX_TOKENS


@pytest.mark.asyncio
async def test_generate_persona_cli_command_saves_the_materialized_result(tmp_path) -> None:
    class StaticGenerator:
        async def generate(self, *, description, persona_id, name, language):
            return StudentPersona(
                id=persona_id,
                name=name,
                description=description,
                language=language,
                control=StudentControlProfile(motivation=0.8),
            )

    output = tmp_path / "generated.yaml"
    args = Namespace(
        model="test/model",
        reasoning_effort="low",
        temperature=None,
        max_tokens=900,
        timeout=30.0,
        attempts=3,
        description="Interested and careful.",
        id="careful_student",
        name="Careful Student",
        language="match_tutor",
        output=output,
        force=False,
    )

    destination, persona = await run_persona_generation(args, generator=StaticGenerator())

    assert destination == output
    assert load_persona(output) == persona
    parsed = build_parser().parse_args(
        [
            "generate-persona",
            "Interested and careful.",
            "--id",
            "careful_student",
            "--output",
            str(output),
            "--model",
            "test/model",
        ]
    )
    assert parsed.command == "generate-persona"
    assert parsed.description == "Interested and careful."
