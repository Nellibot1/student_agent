from __future__ import annotations

import inspect

from pathlib import Path

import pytest

from student_agent import (
    ConversationMessage,
    ConversationRole,
    InvalidStudentResponseError,
    MessageType,
    PerceivedFeedback,
    SimulatedStudentState,
    SimulationMode,
    StudentObservation,
    StudentSimulator,
    StudentSnapshot,
    ValidationResult,
    available_builtin_personas,
    load_builtin_persona,
)


class RecordingGenerator:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies or ("Could you explain that last step again?",))
        self.calls: list[tuple[str, str]] = []

    async def generate(self, *, instruction: str, tutor_message: str) -> str:
        self.calls.append((instruction, tutor_message))
        return self.replies.pop(0)


@pytest.mark.asyncio
async def test_responds_from_visible_tutor_text_only() -> None:
    generator = RecordingGenerator()
    simulator = StudentSimulator(generator, seed=42)

    reply = await simulator.respond("What does this result mean?")

    assert reply.message == "Could you explain that last step again?"
    assert reply.observation.experiment_seed == "42"
    assert reply.observation.response_expected is True
    assert generator.calls[0][1] == "What does this result mean?"
    assert "latest_tutor_message" in generator.calls[0][0]
    assert simulator.history == [
        ConversationMessage(role=ConversationRole.TUTOR, content="What does this result mean?"),
        ConversationMessage(role=ConversationRole.STUDENT, content=reply.message),
    ]


def test_public_respond_api_has_no_tutor_metadata_parameter() -> None:
    parameters = inspect.signature(StudentSimulator.respond).parameters

    assert tuple(parameters) == ("self", "tutor_message", "repair_hint")
    assert "context" not in parameters


@pytest.mark.asyncio
async def test_topic_guess_ignores_question_scaffolding() -> None:
    simulator = StudentSimulator(RecordingGenerator("I am unsure."))

    reply = await simulator.respond("Can you explain probability density?")

    assert reply.observation.perceived_topic == "probability density"


@pytest.mark.asyncio
async def test_visible_feedback_is_inferred_and_updates_private_state() -> None:
    generator = RecordingGenerator("My first answer.", "I will include the interval this time.")
    simulator = StudentSimulator(generator)

    first = await simulator.respond("What is the domain of this function?")
    before_mastery = first.state.mastery_by_topic[first.observation.topic_key]
    second = await simulator.respond("That is almost correct, but you missed the interval. Can you try again?")

    assert second.observation.perceived_feedback is PerceivedFeedback.PARTIAL
    assert second.observation.message_type is MessageType.FEEDBACK_AND_QUESTION
    assert second.observation.topic_key == first.observation.topic_key
    assert second.observation.attempt_number == 2
    assert second.state.turns_seen == 2
    assert second.state.mastery_by_topic[second.observation.topic_key] > before_mastery


class FixedPerception:
    def __init__(self) -> None:
        self.inputs: list[tuple[str, tuple[ConversationMessage, ...]]] = []

    async def observe(self, *, tutor_message, history, state) -> StudentObservation:
        del state
        self.inputs.append((tutor_message, tuple(history)))
        return StudentObservation(
            tutor_message=tutor_message,
            perceived_topic="linear equations",
            topic_key="linear-equations",
            message_type=MessageType.QUESTION,
            response_expected=True,
            turn_number=1 + sum(item.role is ConversationRole.TUTOR for item in history),
        )


@pytest.mark.asyncio
async def test_custom_perception_can_replace_default_heuristics() -> None:
    perception = FixedPerception()
    simulator = StudentSimulator(RecordingGenerator("x is three"), perception=perception)

    reply = await simulator.respond("Solve 2x = 6.")

    assert reply.observation.perceived_topic == "linear equations"
    assert perception.inputs == [("Solve 2x = 6.", ())]


@pytest.mark.asyncio
async def test_state_prompted_mode_sees_latest_visible_message() -> None:
    generator = RecordingGenerator("My answer.")
    simulator = StudentSimulator(generator, mode=SimulationMode.STATE_PROMPTED_STUDENT)

    await simulator.respond("Why is the derivative zero here?")

    instruction, _ = generator.calls[0]
    assert "<latest_tutor_message>" in instruction
    assert "Why is the derivative zero here?" in instruction


@pytest.mark.asyncio
async def test_snapshot_round_trip_includes_private_history() -> None:
    simulator = StudentSimulator(
        RecordingGenerator("My answer."),
        state=SimulatedStudentState(mastery_by_topic={"topic-a": 0.65}),
        seed="experiment-7",
    )
    await simulator.respond("Can you explain this example?")

    payload = simulator.snapshot().model_dump_json()
    snapshot = StudentSnapshot.model_validate_json(payload)
    restored = StudentSimulator.from_snapshot(snapshot, generator=RecordingGenerator())

    assert snapshot.schema_version == 2
    assert restored.seed == "experiment-7"
    assert restored.state is not simulator.state
    assert restored.history == simulator.history
    assert restored.history is not simulator.history


def test_all_migrated_personas_are_packaged_and_valid() -> None:
    expected = ("luna", "matej", "nika", "rok", "stress_zero", "tom")

    assert available_builtin_personas() == expected
    for persona_id in expected:
        persona = load_builtin_persona(persona_id)
        assert persona.id == persona_id
        assert 0.0 <= persona.control.mastery <= 1.0

    assert load_builtin_persona("luna").metadata["prior_student_id"] == "luna"
    assert load_builtin_persona("tom").metadata["prior_student_id"] == "tom"


class RejectOnceValidator:
    def __init__(self) -> None:
        self.calls = 0

    async def validate(self, *, message, decision, observation) -> ValidationResult:
        del message, decision, observation
        self.calls += 1
        if self.calls == 1:
            return ValidationResult(
                accepted=False,
                reason="first response does not match the intended behavior",
                repair_hint="Make the uncertainty explicit.",
            )
        return ValidationResult(accepted=True)


@pytest.mark.asyncio
async def test_validator_can_repair_and_retry_without_reprocessing_turn() -> None:
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
    assert reply.state.turns_seen == 1
    assert "Make the uncertainty explicit." in generator.calls[1][0]


@pytest.mark.asyncio
async def test_failed_generation_does_not_commit_state_or_history() -> None:
    simulator = StudentSimulator(RecordingGenerator(""))

    with pytest.raises(InvalidStudentResponseError):
        await simulator.respond("Please answer.")

    assert simulator.state.turns_seen == 0
    assert simulator.history == []


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
