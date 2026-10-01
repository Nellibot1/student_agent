from __future__ import annotations

from pathlib import Path

import pytest

from pydantic import ValidationError

from student_agent import (
    Grade,
    InvalidStudentResponseError,
    SimulatedStudentState,
    SimulationContext,
    StudentSimulator,
    StudentSnapshot,
    ValidationResult,
)


class RecordingGenerator:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies or ("Could you explain that last step again?",))
        self.calls: list[tuple[str, str]] = []

    async def generate(self, *, instruction: str, tutor_message: str) -> str:
        self.calls.append((instruction, tutor_message))
        return self.replies.pop(0)


@pytest.mark.asyncio
async def test_responds_without_application_or_provider_sdk() -> None:
    generator = RecordingGenerator()
    simulator = StudentSimulator(generator, seed=42)

    reply = await simulator.respond(
        "What does this result mean?",
        context=SimulationContext(
            topic_id="result-meaning",
            task_type="explanation",
            expects_answer=True,
            turn_number=1,
        ),
    )

    assert reply.message == "Could you explain that last step again?"
    assert reply.context.experiment_seed == "42"
    assert generator.calls[0][1] == "What does this result mean?"
    assert "latest_tutor_message" in generator.calls[0][0]


@pytest.mark.asyncio
async def test_state_evolves_from_typed_grade() -> None:
    simulator = StudentSimulator(
        RecordingGenerator("I understand it now."),
        state=SimulatedStudentState(mastery_by_topic={"topic-a": 0.4}),
    )

    reply = await simulator.respond(
        "Good. Now explain it once more.",
        context=SimulationContext(
            topic_id="topic-a",
            task_type="explanation",
            expects_answer=True,
            turn_number=2,
            previous_grade=Grade.CORRECT,
        ),
    )

    assert reply.state.turns_seen == 1
    assert reply.state.mastery_by_topic["topic-a"] > 0.4
    assert simulator.state == reply.state


def test_unknown_grade_is_rejected() -> None:
    with pytest.raises(ValidationError):
        SimulationContext(previous_grade="Correct")


def test_snapshot_round_trip() -> None:
    simulator = StudentSimulator(
        RecordingGenerator(),
        state=SimulatedStudentState(mastery_by_topic={"topic-a": 0.65}),
        seed="experiment-7",
    )

    payload = simulator.snapshot().model_dump_json()
    snapshot = StudentSnapshot.model_validate_json(payload)
    restored = StudentSimulator.from_snapshot(snapshot, generator=RecordingGenerator())

    assert restored.seed == "experiment-7"
    assert restored.state.mastery_by_topic == {"topic-a": 0.65}
    assert restored.state is not simulator.state


class RejectOnceValidator:
    def __init__(self) -> None:
        self.calls = 0

    async def validate(self, *, message, decision, context) -> ValidationResult:
        del message, decision, context
        self.calls += 1
        if self.calls == 1:
            return ValidationResult(
                accepted=False,
                reason="first response does not match the intended behavior",
                repair_hint="Make the uncertainty explicit.",
            )
        return ValidationResult(accepted=True)


@pytest.mark.asyncio
async def test_validator_can_repair_and_retry() -> None:
    generator = RecordingGenerator("first response", "second response")
    validator = RejectOnceValidator()
    simulator = StudentSimulator(
        generator,
        validator=validator,
        max_generation_attempts=2,
    )

    reply = await simulator.respond("Please answer.")

    assert reply.message == "second response"
    assert reply.generation_attempts == 2
    assert "Make the uncertainty explicit." in generator.calls[1][0]


@pytest.mark.asyncio
async def test_empty_response_raises_package_error() -> None:
    simulator = StudentSimulator(RecordingGenerator(""))

    with pytest.raises(InvalidStudentResponseError):
        await simulator.respond("Please answer.")


def test_package_has_no_repository_or_provider_imports() -> None:
    package_root = Path(__file__).parents[1] / "src" / "student_agent"
    source = "\n".join(path.read_text(encoding="utf-8") for path in package_root.rglob("*.py"))
    forbidden_imports = (
        "from app.",
        "import app.",
        "from google.adk",
        "import google.adk",
        "from openai",
        "from anthropic",
    )
    assert not [item for item in forbidden_imports if item in source]
