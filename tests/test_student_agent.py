from __future__ import annotations

import inspect

from pathlib import Path

import pytest

from student_agent import (
    ConversationMessage,
    ConversationRole,
    ExpectedObservation,
    InvalidStudentResponseError,
    LearningDynamics,
    MessageType,
    MisconceptionState,
    PerceivedFeedback,
    PerceptionEvaluationCase,
    PerceptionFailurePolicy,
    SimulatedStudentState,
    StudentObservation,
    StudentPerceptionError,
    StudentPersona,
    StudentResponseMode,
    StudentSimulator,
    StudentSnapshot,
    ValidationFailurePolicy,
    ValidationPolicy,
    ValidationResult,
    ValidationStatus,
    available_builtin_personas,
    evaluate_perception,
    load_builtin_persona,
    response_requires_validation,
)
from student_agent.policy import (
    _misconception_repeat_probability,
    choose_response_mode,
    evolve_simulated_state,
)


class RecordingGenerator:
    def __init__(self, *replies: str) -> None:
        self.replies = list(replies or ("Could you explain that last step again?",))
        self.calls: list[tuple[str, str]] = []

    async def generate(self, *, instruction: str, tutor_message: str) -> str:
        self.calls.append((instruction, tutor_message))
        return self.replies.pop(0)


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
async def test_responds_from_visible_tutor_text_only() -> None:
    generator = RecordingGenerator()
    simulator = StudentSimulator(generator, perception=FixedPerception(), seed=42)

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
    constructor_parameters = inspect.signature(StudentSimulator).parameters

    assert tuple(parameters) == ("self", "tutor_message", "repair_hint")
    assert "context" not in parameters
    assert constructor_parameters["perception"].default is inspect.Parameter.empty


@pytest.mark.asyncio
async def test_visible_feedback_is_inferred_and_updates_private_state() -> None:
    class FeedbackPerception:
        async def observe(self, *, tutor_message, history, state) -> StudentObservation:
            is_partial = "almost correct" in tutor_message
            topic_key = "function-domain"
            return StudentObservation(
                tutor_message=tutor_message,
                perceived_topic="function domain",
                topic_key=topic_key,
                message_type=MessageType.FEEDBACK_AND_QUESTION if is_partial else MessageType.QUESTION,
                perceived_feedback=PerceivedFeedback.PARTIAL if is_partial else PerceivedFeedback.NONE,
                feedback_confidence=0.95 if is_partial else 0.0,
                correction_present=is_partial,
                explanation_present=is_partial,
                response_expected=True,
                turn_number=1 + sum(item.role is ConversationRole.TUTOR for item in history),
                attempt_number=state.attempts_by_topic.get(topic_key, 0) + 1,
            )

    generator = RecordingGenerator("My first answer.", "I will include the interval this time.")
    simulator = StudentSimulator(generator, perception=FeedbackPerception())

    first = await simulator.respond("What is the domain of this function?")
    before_mastery = first.state.mastery_by_topic[first.observation.topic_key]
    second = await simulator.respond("That is almost correct, but you missed the interval. Can you try again?")

    assert second.observation.perceived_feedback is PerceivedFeedback.PARTIAL
    assert second.observation.message_type is MessageType.FEEDBACK_AND_QUESTION
    assert second.observation.topic_key == first.observation.topic_key
    assert second.observation.attempt_number == 2
    assert second.state.turns_seen == 2
    assert second.state.mastery_by_topic[second.observation.topic_key] > before_mastery


@pytest.mark.asyncio
async def test_incorrect_assessment_without_instruction_does_not_erase_mastery() -> None:
    class IncorrectPerception:
        async def observe(self, *, tutor_message, history, state) -> StudentObservation:
            del history, state
            return StudentObservation(
                tutor_message=tutor_message,
                perceived_topic="linear equations",
                topic_key="linear-equations",
                message_type=MessageType.FEEDBACK,
                perceived_feedback=PerceivedFeedback.INCORRECT,
                feedback_confidence=1.0,
                correction_present=False,
                explanation_present=False,
            )

    simulator = StudentSimulator(
        RecordingGenerator("I will try again."),
        perception=IncorrectPerception(),
        state=SimulatedStudentState(mastery_by_topic={"linear-equations": 0.60}),
    )

    reply = await simulator.respond("No, try again.")

    assert reply.state.mastery_by_topic["linear-equations"] == 0.60
    assert reply.state.confidence < 0.50
    assert reply.state.confusion > 0.30


def test_incorrect_explanation_records_exposure_without_inflating_mastery() -> None:
    persona = load_builtin_persona("tom")
    state = SimulatedStudentState(
        mastery_by_topic={"probability": 0.40},
        confusion=0.70,
    )
    observation = StudentObservation(
        tutor_message="That is incorrect. Disjoint union probabilities are added.",
        perceived_topic="probability",
        topic_key="probability",
        perceived_task="add disjoint probabilities",
        task_key="disjoint-union",
        message_type=MessageType.FEEDBACK_AND_QUESTION,
        perceived_feedback=PerceivedFeedback.INCORRECT,
        feedback_confidence=1.0,
        correction_present=True,
        explanation_present=True,
        misconception_statement="Disjoint union probabilities should be multiplied.",
        response_expected=True,
        attempt_number=4,
    )

    updated = evolve_simulated_state(state, persona, observation)

    assert updated.mastery_by_topic["probability"] == pytest.approx(0.40)
    assert updated.instructional_exposure_by_topic["probability"] > 0.0
    assert updated.confusion > 0.68


def test_correct_answer_after_extensive_scaffolding_is_weaker_mastery_evidence() -> None:
    persona = load_builtin_persona("tom")
    state = SimulatedStudentState(mastery_by_topic={"probability": 0.20})

    def correct_observation(attempt_number: int) -> StudentObservation:
        return StudentObservation(
            tutor_message="Correct.",
            perceived_topic="probability",
            topic_key="probability",
            perceived_task="apply the rule",
            task_key="apply-rule",
            message_type=MessageType.FEEDBACK,
            perceived_feedback=PerceivedFeedback.CORRECT,
            feedback_confidence=1.0,
            demonstrated_understanding=True,
            attempt_number=attempt_number,
        )

    independent = evolve_simulated_state(state, persona, correct_observation(1))
    scaffolded = evolve_simulated_state(state, persona, correct_observation(10))

    assert independent.mastery_by_topic["probability"] > scaffolded.mastery_by_topic["probability"]


def test_correct_feedback_only_decays_misconceptions_for_the_demonstrated_task() -> None:
    persona = load_builtin_persona("tom")
    state = SimulatedStudentState(
        misconceptions_by_id={
            "probability:first": MisconceptionState(
                topic_key="probability",
                task_key="first-task",
                belief="First mistaken belief.",
                strength=0.8,
            ),
            "probability:second": MisconceptionState(
                topic_key="probability",
                task_key="second-task",
                belief="Second mistaken belief.",
                strength=0.8,
            ),
        },
        misconception_strength_by_id={"probability:first": 0.8, "probability:second": 0.8},
        active_misconceptions=["probability:first", "probability:second"],
    )
    observation = StudentObservation(
        tutor_message="Correct.",
        perceived_topic="probability",
        topic_key="probability",
        perceived_task="first task",
        task_key="first-task",
        message_type=MessageType.FEEDBACK,
        perceived_feedback=PerceivedFeedback.CORRECT,
        feedback_confidence=1.0,
        demonstrated_understanding=True,
        attempt_number=1,
    )

    updated = evolve_simulated_state(state, persona, observation)

    assert updated.misconceptions_by_id["probability:first"].strength < 0.8
    assert updated.misconceptions_by_id["probability:second"].strength == pytest.approx(0.8)


def test_stress_persona_stays_persistent_but_adapts_after_many_corrections() -> None:
    realistic = StudentPersona(
        id="realistic",
        learning_dynamics=LearningDynamics(
            purpose="realistic",
            misconception_resistance=0.65,
            corrections_before_adaptation=3,
        ),
    )
    stress = load_builtin_persona("tom")

    realistic_early = _misconception_repeat_probability(
        realistic,
        misconception_probability=0.88,
        misconception_strength=1.0,
        attempt_number=2,
    )
    realistic_late = _misconception_repeat_probability(
        realistic,
        misconception_probability=0.88,
        misconception_strength=1.0,
        attempt_number=10,
    )
    stress_early = _misconception_repeat_probability(
        stress,
        misconception_probability=0.88,
        misconception_strength=1.0,
        attempt_number=2,
    )
    stress_late = _misconception_repeat_probability(
        stress,
        misconception_probability=0.88,
        misconception_strength=1.0,
        attempt_number=16,
    )

    assert stress.learning_dynamics.purpose == "tutor_stress_test"
    assert stress_early > realistic_early
    assert realistic_late < realistic_early
    assert 0.0 < stress_late < stress_early


@pytest.mark.asyncio
async def test_instruction_records_components_and_contentful_misconception() -> None:
    class EvidencePerception:
        async def observe(self, *, tutor_message, history, state) -> StudentObservation:
            del history, state
            return StudentObservation(
                tutor_message=tutor_message,
                perceived_topic="probability density",
                topic_key="pdf",
                perceived_task="distinguish density from point probability",
                task_key="pdf-point-probability",
                message_type=MessageType.FEEDBACK,
                perceived_feedback=PerceivedFeedback.INCORRECT,
                feedback_confidence=0.9,
                correction_present=True,
                explanation_present=True,
                knowledge_components=["Interval probability is the integral of the density."],
                misconception_statement="The density value is the probability at that point.",
                evidence="The tutor explicitly corrects point probability.",
            )

    reply = await StudentSimulator(
        RecordingGenerator("So I need to integrate over an interval."),
        perception=EvidencePerception(),
    ).respond("A density value is not a point probability; integrate over an interval.")

    assert reply.state.current_task_key == "pdf-point-probability"
    assert next(iter(reply.state.knowledge_components_by_id.values())).statement.startswith("Interval probability")
    misconception = next(iter(reply.state.misconceptions_by_id.values()))
    assert misconception.belief == "The density value is the probability at that point."
    assert misconception.evidence == "The tutor explicitly corrects point probability."


@pytest.mark.asyncio
async def test_offline_perception_evaluation_reports_accuracy_and_topic_stability() -> None:
    class EvaluatedPerception:
        async def observe(self, *, tutor_message, history, state) -> StudentObservation:
            del history
            topic_key = next(iter(state.topic_labels_by_key), "pdf-topic")
            return StudentObservation(
                tutor_message=tutor_message,
                perceived_topic="probability density",
                topic_key=topic_key,
                perceived_task="explain density",
                task_key="pdf-task",
                perceived_feedback=PerceivedFeedback.PARTIAL,
                feedback_confidence=0.8,
                correction_present=True,
                explanation_present=True,
            )

    expected = ExpectedObservation(
        feedback=PerceivedFeedback.PARTIAL,
        correction_present=True,
        explanation_present=True,
        topic_group="pdf",
    )
    report = await evaluate_perception(
        EvaluatedPerception(),
        [
            PerceptionEvaluationCase(id="1", tutor_message="Almost.", expected=expected),
            PerceptionEvaluationCase(id="2", tutor_message="Add the integral condition.", expected=expected),
        ],
    )

    assert report.feedback_accuracy == 1.0
    assert report.topic_stability == 1.0
    assert report.confusion_matrix == {"partial": {"partial": 2}}


@pytest.mark.asyncio
async def test_injected_semantic_perception_drives_the_turn() -> None:
    perception = FixedPerception()
    simulator = StudentSimulator(RecordingGenerator("x is three"), perception=perception)

    reply = await simulator.respond("Solve 2x = 6.")

    assert reply.observation.perceived_topic == "linear equations"
    assert perception.inputs == [("Solve 2x = 6.", ())]


@pytest.mark.asyncio
async def test_external_reply_updates_state_and_visible_history_without_generation() -> None:
    generator = RecordingGenerator()
    simulator = StudentSimulator(generator, perception=FixedPerception())

    reply = await simulator.record_external_reply(
        "You have completed conditional probability.",
        "Can we move on to total probability?",
    )

    assert generator.calls == []
    assert reply.generation_attempts == 0
    assert reply.validation_status is ValidationStatus.SKIPPED
    assert reply.validation_reason == "externally supplied visible student reply"
    assert simulator.state.turns_seen == 1
    assert simulator.history[-2:] == [
        ConversationMessage(role=ConversationRole.TUTOR, content="You have completed conditional probability."),
        ConversationMessage(role=ConversationRole.STUDENT, content="Can we move on to total probability?"),
    ]


@pytest.mark.asyncio
async def test_snapshot_round_trip_includes_private_history() -> None:
    simulator = StudentSimulator(
        RecordingGenerator("My answer."),
        perception=FixedPerception(),
        state=SimulatedStudentState(mastery_by_topic={"topic-a": 0.65}),
        seed="experiment-7",
    )
    await simulator.respond("Can you explain this example?")

    payload = simulator.snapshot().model_dump_json()
    snapshot = StudentSnapshot.model_validate_json(payload)
    restored = StudentSimulator.from_snapshot(
        snapshot,
        generator=RecordingGenerator(),
        perception=FixedPerception(),
    )

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


class RejectingValidator:
    def __init__(self, *, safe_to_emit: bool = True) -> None:
        self.safe_to_emit = safe_to_emit

    async def validate(self, *, message, decision, observation) -> ValidationResult:
        del message, decision, observation
        return ValidationResult(
            accepted=False,
            reason="reply does not match the intended partial behavior",
            repair_hint="Leave one meaningful part unresolved.",
            safe_to_emit=self.safe_to_emit,
            observed_mode=StudentResponseMode.ASK_CLARIFICATION,
        )


@pytest.mark.asyncio
async def test_validator_can_repair_and_retry_without_reprocessing_turn() -> None:
    generator = RecordingGenerator("first response", "second response")
    validator = RejectOnceValidator()
    simulator = StudentSimulator(
        generator,
        perception=FixedPerception(),
        validator=validator,
        validation_policy=ValidationPolicy.ALWAYS,
        max_generation_attempts=2,
    )

    reply = await simulator.respond("Please answer.")

    assert reply.message == "second response"
    assert reply.generation_attempts == 2
    assert reply.state.turns_seen == 1
    assert "Make the uncertainty explicit." in generator.calls[1][0]


def test_risky_validation_policy_targets_only_hard_to_realize_modes() -> None:
    risky = {
        StudentResponseMode.ANSWER_PARTIAL,
        StudentResponseMode.ANSWER_INCORRECT,
        StudentResponseMode.CORRECT_COMPUTATION_WRONG_CONCEPT,
        StudentResponseMode.TOPIC_JUMP,
    }

    for mode in StudentResponseMode:
        assert response_requires_validation(mode, ValidationPolicy.RISKY) is (mode in risky)
        assert response_requires_validation(mode, ValidationPolicy.ALWAYS) is True
        assert response_requires_validation(mode, ValidationPolicy.OFF) is False


def test_bare_negative_feedback_prefers_clarification_over_partial_answer() -> None:
    observation = StudentObservation(
        tutor_message="Wrong.",
        perceived_topic="basic arithmetic",
        topic_key="arithmetic",
        message_type=MessageType.FEEDBACK,
        perceived_feedback=PerceivedFeedback.INCORRECT,
        feedback_confidence=0.95,
        correction_present=False,
        explanation_present=False,
        response_expected=True,
    )

    decision = choose_response_mode(
        load_builtin_persona("tom"),
        observation,
        simulated_state=SimulatedStudentState(confidence=0.1, confusion=0.9),
    )

    assert decision.mode is StudentResponseMode.ASK_CLARIFICATION
    assert "without providing a correction" in decision.reason


@pytest.mark.asyncio
async def test_interactive_validation_failure_keeps_safe_reply_and_commits_state() -> None:
    simulator = StudentSimulator(
        RecordingGenerator("Why would 1 + 1 not equal 2?", "Could you explain why it is not 2?"),
        perception=FixedPerception(),
        validator=RejectingValidator(),
        validation_policy=ValidationPolicy.ALWAYS,
        validation_failure_policy=ValidationFailurePolicy.CONTINUE,
        max_generation_attempts=2,
    )

    reply = await simulator.respond("Wrong.")

    assert reply.message == "Could you explain why it is not 2?"
    assert reply.validation_status is ValidationStatus.DEGRADED
    assert "does not match" in reply.validation_reason
    assert reply.observed_mode is StudentResponseMode.ASK_CLARIFICATION
    assert reply.generation_attempts == 2
    assert simulator.state.turns_seen == 1
    assert simulator.history[-1].content == reply.message


@pytest.mark.asyncio
async def test_validation_failure_never_emits_reply_marked_unsafe() -> None:
    simulator = StudentSimulator(
        RecordingGenerator("My response mode is answer_partial."),
        perception=FixedPerception(),
        validator=RejectingValidator(safe_to_emit=False),
        validation_policy=ValidationPolicy.ALWAYS,
        validation_failure_policy=ValidationFailurePolicy.CONTINUE,
    )

    with pytest.raises(InvalidStudentResponseError, match="does not match"):
        await simulator.respond("Wrong.")

    assert simulator.history == []


class UnavailablePerception:
    async def observe(self, *, tutor_message, history, state) -> StudentObservation:
        del tutor_message, history, state
        raise StudentPerceptionError("semantic perception returned no content")


@pytest.mark.asyncio
async def test_degraded_perception_continues_with_clarification_without_updating_state() -> None:
    initial_state = SimulatedStudentState(
        current_topic="basic arithmetic",
        current_topic_key="arithmetic",
        current_task="add one and one",
        current_task_key="one-plus-one",
        mastery_by_topic={"arithmetic": 0.9},
        attempts_by_task={"one-plus-one": 2},
        confidence=0.7,
        turns_seen=4,
    )
    simulator = StudentSimulator(
        RecordingGenerator("Why would 1 + 1 not equal 2?"),
        perception=UnavailablePerception(),
        state=initial_state,
        perception_failure_policy=PerceptionFailurePolicy.CONTINUE,
    )

    reply = await simulator.respond("No, I mean 1 + 1.")

    assert reply.message == "Why would 1 + 1 not equal 2?"
    assert reply.observation.perception_source == "semantic_degraded"
    assert "no content" in reply.observation.perception_error
    assert reply.decision.mode is StudentResponseMode.ASK_CLARIFICATION
    assert reply.state == initial_state
    assert simulator.state == initial_state
    assert simulator.history[-2:] == [
        ConversationMessage(role=ConversationRole.TUTOR, content="No, I mean 1 + 1."),
        ConversationMessage(role=ConversationRole.STUDENT, content=reply.message),
    ]


@pytest.mark.asyncio
async def test_strict_perception_failure_policy_still_fails_without_committing() -> None:
    simulator = StudentSimulator(RecordingGenerator(), perception=UnavailablePerception())

    with pytest.raises(StudentPerceptionError, match="no content"):
        await simulator.respond("No, I mean 1 + 1.")

    assert simulator.history == []
    assert simulator.state.turns_seen == 0


@pytest.mark.asyncio
async def test_failed_generation_does_not_commit_state_or_history() -> None:
    simulator = StudentSimulator(RecordingGenerator(""), perception=FixedPerception())

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
