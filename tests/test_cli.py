from __future__ import annotations

import json

from argparse import Namespace
from pathlib import Path

import pytest

from student_agent import (
    DEFAULT_PERSONA,
    MisconceptionState,
    PolicyDecision,
    StudentPerceptionError,
    StudentProviderError,
    StudentResponseMode,
    ValidationResult,
)
from student_agent.cli import (
    build_parser,
    doctor_report,
    load_local_environment,
    run_chat,
    select_persona,
    validate_model_configuration,
)
from student_agent.integrations import (
    LiteLLMStudentGenerator,
    LiteLLMStudentPerception,
    LiteLLMStudentResponseValidator,
)
from student_agent.perception import MessageType, StudentObservation
from student_agent.persistence import load_conversation
from student_agent.policy import SimulatedStudentState, evolve_simulated_state


class RecordingGenerator:
    def __init__(self, reply: str = "I am not sure yet.") -> None:
        self.reply = reply
        self.messages: list[str] = []

    async def generate(self, *, instruction: str, tutor_message: str) -> str:
        del instruction
        self.messages.append(tutor_message)
        return self.reply


class StaticSemanticPerception:
    async def observe(self, *, tutor_message, history, state) -> StudentObservation:
        del state
        return StudentObservation(
            tutor_message=tutor_message,
            perceived_topic="probability density",
            topic_key="probability-density",
            message_type=MessageType.QUESTION,
            response_expected=True,
            turn_number=1 + len(history) // 2,
        )


class AcceptingValidator:
    async def validate(self, *, message, decision, observation) -> ValidationResult:
        del message, decision, observation
        return ValidationResult(accepted=True)


class UnavailableSemanticPerception:
    async def observe(self, *, tutor_message, history, state) -> StudentObservation:
        del tutor_message, history, state
        raise StudentPerceptionError("semantic perception exhausted retries")


def chat_args(**overrides) -> Namespace:
    values = {
        "persona": "default",
        "persona_json": "",
        "model": "test/model",
        "reasoning_effort": "low",
        "temperature": None,
        "max_tokens": 1000,
        "timeout": 30.0,
        "perception_model": "",
        "perception_max_tokens": 800,
        "perception_timeout": 20.0,
        "perception_attempts": 3,
        "perception_failure_policy": "continue",
        "validator_model": "",
        "validation_max_tokens": 250,
        "validation_timeout": 20.0,
        "validation_policy": "risky",
        "validation_failure_policy": "continue",
        "generation_attempts": 2,
        "mode": "HiddenStateStudent",
        "seed": "42",
        "message": "Explain a probability density function.",
        "once": True,
        "resume": None,
        "save": None,
        "no_save": True,
    }
    values.update(overrides)
    return Namespace(**values)


def test_cli_parser_supports_chat_and_persona_listing() -> None:
    parser = build_parser()

    chat = parser.parse_args(["chat", "--model", "local/model", "--persona", "luna"])
    personas = parser.parse_args(["personas"])

    assert chat.model == "local/model"
    assert chat.reasoning_effort == ""
    assert chat.persona == "luna"
    assert not hasattr(chat, "perception")
    assert personas.command == "personas"


def test_cli_parser_supports_first_run_doctor() -> None:
    args = build_parser().parse_args(["doctor", "--model", "openai/gpt-4.1-mini"])

    assert args.command == "doctor"
    assert args.model == "openai/gpt-4.1-mini"


def test_cli_reasoning_effort_can_be_overridden(monkeypatch) -> None:
    monkeypatch.setenv("STUDENT_AGENT_REASONING_EFFORT", "medium")
    parser = build_parser()

    assert parser.parse_args(["chat", "--model", "local/model"]).reasoning_effort == "medium"
    assert (
        parser.parse_args(["chat", "--model", "local/model", "--reasoning-effort", "none"]).reasoning_effort == "none"
    )


@pytest.mark.asyncio
async def test_chat_applies_shared_reasoning_effort_to_all_model_calls(monkeypatch) -> None:
    captured: dict[str, str] = {}

    class CapturingGenerator(RecordingGenerator):
        def __init__(self, model, **kwargs):
            del model
            captured["generation"] = kwargs["reasoning_effort"]
            super().__init__()

    class CapturingPerception(StaticSemanticPerception):
        def __init__(self, model, **kwargs):
            del model
            captured["perception"] = kwargs["reasoning_effort"]

    class CapturingValidator(AcceptingValidator):
        def __init__(self, model, **kwargs):
            del model
            captured["validation"] = kwargs["reasoning_effort"]

    monkeypatch.setattr("student_agent.cli.LiteLLMStudentGenerator", CapturingGenerator)
    monkeypatch.setattr("student_agent.cli.LiteLLMStudentPerception", CapturingPerception)
    monkeypatch.setattr("student_agent.cli.LiteLLMStudentResponseValidator", CapturingValidator)

    await run_chat(chat_args(reasoning_effort="none"), write_line=lambda message: None)

    assert captured == {"generation": "none", "perception": "none", "validation": "none"}


def test_local_env_supplies_default_model_without_overwriting_exported_value(monkeypatch, tmp_path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text(
        "OPENAI_API_KEY=test-key-from-file\nSTUDENT_AGENT_MODEL=openai/from-file\n",
        encoding="utf-8",
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setenv("STUDENT_AGENT_MODEL", "openai/exported-value")

    assert load_local_environment() is True
    args = build_parser().parse_args(["chat"])

    assert args.model == "openai/exported-value"
    assert __import__("os").environ["OPENAI_API_KEY"] == "test-key-from-file"


@pytest.mark.parametrize(
    "key",
    ["", "replace-with-your-openai-api-key", "your-real-key", "your-actual-api-key"],
)
def test_openai_preflight_rejects_missing_or_placeholder_key(monkeypatch, tmp_path, key) -> None:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("student_agent.cli.find_spec", lambda package: object())
    monkeypatch.setenv("OPENAI_API_KEY", key)

    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        validate_model_configuration("openai/gpt-4.1-mini")


def test_doctor_never_prints_api_key(monkeypatch) -> None:
    secret = "sk-test-secret-that-must-not-be-printed"
    monkeypatch.setenv("OPENAI_API_KEY", secret)

    lines, ready = doctor_report("openai/gpt-4.1-mini")

    assert ready is True
    assert secret not in "\n".join(lines)
    assert "OPENAI_API_KEY is configured" in "\n".join(lines)


def test_example_environment_file_contains_no_api_key() -> None:
    example = Path(__file__).parents[1] / ".env.example"
    key_line = next(line for line in example.read_text(encoding="utf-8").splitlines() if line.startswith("OPENAI_API_KEY="))

    assert key_line == "OPENAI_API_KEY="


def test_inline_persona_can_be_created_on_the_spot() -> None:
    persona = select_persona("default", '{"id":"new_student","name":"New student"}')

    assert persona.id == "new_student"
    assert persona.name == "New student"


@pytest.mark.asyncio
async def test_one_shot_chat_passes_only_visible_message_to_student() -> None:
    generator = RecordingGenerator("Could you explain what density means?")
    output: list[str] = []

    student = await run_chat(
        chat_args(),
        generator=generator,
        perception=StaticSemanticPerception(),
        validator=AcceptingValidator(),
        write_line=output.append,
    )

    assert generator.messages == ["Explain a probability density function."]
    assert output[-1] == "Student: Could you explain what density means?"
    assert len(student.history) == 2


@pytest.mark.asyncio
async def test_chat_continues_conservatively_when_semantic_perception_is_unavailable() -> None:
    output: list[str] = []

    student = await run_chat(
        chat_args(message="No, I mean ordinary addition."),
        generator=RecordingGenerator("What do you mean by that?"),
        perception=UnavailableSemanticPerception(),
        validator=AcceptingValidator(),
        write_line=output.append,
    )

    assert student.state.turns_seen == 0
    assert student.history[-1].content == "What do you mean by that?"
    assert any("learner state was preserved" in line for line in output)
    assert output[-1] == "Student: What do you mean by that?"


@pytest.mark.asyncio
async def test_one_shot_chat_saves_structured_conversation(tmp_path) -> None:
    destination = tmp_path / "probability.json"
    generator = RecordingGenerator("Density is related to probability over an interval.")

    await run_chat(
        chat_args(save=destination, no_save=False),
        generator=generator,
        perception=StaticSemanticPerception(),
        validator=AcceptingValidator(),
        write_line=lambda message: None,
    )
    saved = load_conversation(destination)

    assert saved.model == "test/model"
    assert saved.snapshot.persona.id == "balanced_student"
    assert [message.content for message in saved.snapshot.history] == [
        "Explain a probability density function.",
        "Density is related to probability over an interval.",
    ]
    assert saved.turns[0].observation.tutor_message == "Explain a probability density function."
    assert saved.turns[0].student_message == "Density is related to probability over an interval."


@pytest.mark.asyncio
async def test_saved_conversation_can_resume_and_update_same_file(tmp_path) -> None:
    destination = tmp_path / "continued.json"
    await run_chat(
        chat_args(save=destination, no_save=False),
        generator=RecordingGenerator("First student reply."),
        perception=StaticSemanticPerception(),
        validator=AcceptingValidator(),
        write_line=lambda message: None,
    )

    resumed = await run_chat(
        chat_args(
            resume=destination,
            save=None,
            no_save=False,
            message="Can you explain that again?",
        ),
        generator=RecordingGenerator("Second student reply."),
        perception=StaticSemanticPerception(),
        validator=AcceptingValidator(),
        write_line=lambda message: None,
    )
    saved = load_conversation(destination)

    assert len(resumed.history) == 4
    assert [turn.student_message for turn in saved.turns] == ["First student reply.", "Second student reply."]
    assert saved.snapshot.state.turns_seen == 2


@pytest.mark.asyncio
async def test_litellm_adapter_extracts_text_and_usage_without_real_dependency() -> None:
    requests: list[dict] = []

    async def completion(**request):
        requests.append(request)
        return {
            "choices": [{"message": {"content": "  My answer.  "}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        }

    generator = LiteLLMStudentGenerator("test/model", completion=completion)
    answer = await generator.generate(instruction="Act as the student.", tutor_message="Visible question")

    assert answer == "My answer."
    assert requests[0]["messages"] == [{"role": "user", "content": "Act as the student."}]
    assert "reasoning_effort" not in requests[0]
    assert generator.usage[0].model == "test/model"
    assert generator.usage[0].total_tokens == 13


@pytest.mark.asyncio
async def test_litellm_semantic_perception_understands_mixed_feedback() -> None:
    requests: list[dict] = []

    async def completion(**request):
        requests.append(request)
        return {
            "choices": [
                {
                    "message": {
                        "content": """{
                            "feedback": "partial",
                            "feedback_confidence": 0.91,
                            "response_expected": true,
                            "message_type": "feedback_and_question",
                            "perceived_topic": "probability laws",
                            "correction_present": true,
                            "explanation_present": true,
                            "evidence": "The tutor confirms the event but corrects the measure distinction."
                        }"""
                    }
                }
            ],
            "usage": {"prompt_tokens": 20, "completion_tokens": 12, "total_tokens": 32},
        }

    perception = LiteLLMStudentPerception("test/model", completion=completion)
    observation = await perception.observe(
        tutor_message=(
            "Yes—the event is exactly {X in B}. The small but important distinction is "
            "that P and the law of X live on different spaces. Which side uses the original space?"
        ),
        history=(),
        state=SimulatedStudentState(current_topic="probability laws"),
    )

    assert observation.perceived_feedback.value == "partial"
    assert observation.feedback_confidence == 0.91
    assert observation.correction_present is True
    assert observation.explanation_present is True
    assert observation.perception_source == "semantic"
    assert requests[0]["response_format"] == {"type": "json_object"}
    assert "temperature" not in requests[0]
    assert perception.usage[0].total_tokens == 32


@pytest.mark.asyncio
async def test_semantic_perception_failure_is_not_replaced_with_rules() -> None:
    async def completion(**request):
        del request
        raise TimeoutError("model timed out")

    perception = LiteLLMStudentPerception("test/model", completion=completion)
    with pytest.raises(StudentPerceptionError, match="failed after 3 attempts"):
        await perception.observe(
            tutor_message="That is almost correct. Can you try again?",
            history=(),
            state=SimulatedStudentState(),
        )

    assert "TimeoutError" in perception.last_error


@pytest.mark.asyncio
async def test_authentication_failure_is_not_retried() -> None:
    calls = 0

    class AuthenticationError(Exception):
        pass

    async def completion(**request):
        nonlocal calls
        del request
        calls += 1
        raise AuthenticationError("invalid provider credential")

    perception = LiteLLMStudentPerception("openai/gpt-4.1-mini", completion=completion)
    with pytest.raises(StudentProviderError, match="AuthenticationError"):
        await perception.observe(
            tutor_message="What is 1 + 1?",
            history=(),
            state=SimulatedStudentState(),
        )

    assert calls == 1


@pytest.mark.asyncio
async def test_semantic_perception_recovers_with_semantic_retry() -> None:
    calls = 0
    requests: list[dict] = []

    async def completion(**request):
        nonlocal calls
        calls += 1
        requests.append(request)
        if calls == 1:
            return {"choices": [{"message": {"content": "not-json"}}]}
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"feedback":"none","feedback_confidence":0.8,'
                            '"response_expected":true,"message_type":"question",'
                            '"perceived_topic":"density functions","existing_topic_key":"",'
                            '"perceived_task":"define a density","existing_task_key":"",'
                            '"correction_present":false,"explanation_present":false,'
                            '"demonstrated_understanding":false,"knowledge_components":[],'
                            '"misconception_statement":"","evidence":"The tutor asks for a definition."}'
                        )
                    }
                }
            ]
        }

    perception = LiteLLMStudentPerception("test/model", completion=completion)
    observation = await perception.observe(
        tutor_message="What is a probability density function?",
        history=(),
        state=SimulatedStudentState(),
    )

    assert calls == 2
    assert [request["max_tokens"] for request in requests] == [800, 1600]
    assert observation.perceived_task == "define a density"
    assert perception.last_error == ""


@pytest.mark.asyncio
async def test_empty_perception_output_records_provider_diagnostics() -> None:
    async def completion(**request):
        del request
        return {
            "model": "test/reasoning-model",
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {"content": None, "refusal": ""},
                }
            ],
            "usage": {"prompt_tokens": 40, "completion_tokens": 800, "total_tokens": 840},
        }

    perception = LiteLLMStudentPerception("test/model", max_attempts=1, completion=completion)
    with pytest.raises(StudentPerceptionError, match="finish_reason=length"):
        await perception.observe(
            tutor_message="No, I mean ordinary addition.",
            history=(),
            state=SimulatedStudentState(),
        )

    diagnostic = perception.usage[0]
    assert diagnostic.operation == "perception"
    assert diagnostic.attempt_number == 1
    assert diagnostic.requested_max_tokens == 800
    assert diagnostic.finish_reason == "length"
    assert diagnostic.content_present is False
    assert diagnostic.total_tokens == 840


@pytest.mark.asyncio
async def test_semantic_perception_reuses_local_topic_and_task_keys() -> None:
    calls = 0

    async def completion(**request):
        nonlocal calls
        calls += 1
        assert "pdf-topic" in request["messages"][1]["content"]
        assert "pdf:point-probability" in request["messages"][1]["content"]
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"feedback":"partial","feedback_confidence":0.9,'
                            '"response_expected":true,"message_type":"feedback_and_question",'
                            '"perceived_topic":"density functions","existing_topic_key":"pdf-topic",'
                            '"perceived_task":"distinguish density from point probability",'
                            '"existing_task_key":"pdf-task","correction_present":true,'
                            '"explanation_present":true,"demonstrated_understanding":false,'
                            '"knowledge_components":["Interval probability uses an integral."],'
                            '"misconception_statement":"Density is point probability.",'
                            '"existing_misconception_id":"pdf:point-probability",'
                            '"evidence":"The tutor corrects the point interpretation."}'
                        )
                    }
                }
            ]
        }

    state = SimulatedStudentState(
        topic_labels_by_key={"pdf-topic": "probability density functions"},
        task_labels_by_key={"pdf-task": "distinguish density from point probability"},
        attempts_by_task={"pdf-task": 1},
        misconceptions_by_id={
            "pdf:point-probability": MisconceptionState(
                topic_key="pdf-topic",
                task_key="pdf-task",
                task_keys=["pdf-task"],
                belief="A density value is a point probability.",
                strength=0.6,
                evidence_count=2,
            )
        },
    )
    observation = await LiteLLMStudentPerception("test/model", completion=completion).observe(
        tutor_message="Almost; remember that interval probability requires integration. Try again.",
        history=(),
        state=state,
    )

    assert observation.topic_key == "pdf-topic"
    assert observation.task_key == "pdf-task"
    assert observation.attempt_number == 2
    assert observation.matched_misconception_id == "pdf:point-probability"
    updated = evolve_simulated_state(state, DEFAULT_PERSONA, observation)
    assert calls == 1
    assert len(updated.misconceptions_by_id) == 1
    assert updated.misconceptions_by_id["pdf:point-probability"].belief == (
        "A density value is a point probability."
    )
    assert updated.misconceptions_by_id["pdf:point-probability"].evidence_count == 3


@pytest.mark.asyncio
async def test_semantic_perception_rejects_invented_misconception_id() -> None:
    async def completion(**request):
        del request
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"feedback":"incorrect","feedback_confidence":0.9,'
                            '"response_expected":true,"message_type":"feedback_and_question",'
                            '"perceived_topic":"density functions","existing_topic_key":"pdf-topic",'
                            '"perceived_task":"distinguish density","existing_task_key":"pdf-task",'
                            '"correction_present":true,"explanation_present":true,'
                            '"demonstrated_understanding":false,"knowledge_components":[],'
                            '"misconception_statement":"Density is point probability.",'
                            '"existing_misconception_id":"invented-id",'
                            '"evidence":"The tutor corrects the interpretation."}'
                        )
                    }
                }
            ]
        }

    state = SimulatedStudentState(
        topic_labels_by_key={"pdf-topic": "density functions"},
        task_labels_by_key={"pdf-task": "distinguish density"},
    )
    observation = await LiteLLMStudentPerception("test/model", completion=completion).observe(
        tutor_message="That interpretation is incorrect. Try again.",
        history=(),
        state=state,
    )

    assert observation.matched_misconception_id == ""


@pytest.mark.asyncio
async def test_semantic_perception_sends_bounded_recent_memory_but_keeps_full_state() -> None:
    requests: list[dict] = []

    async def completion(**request):
        requests.append(request)
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"feedback":"none","feedback_confidence":0.8,'
                            '"response_expected":true,"message_type":"question",'
                            '"perceived_topic":"topic 7","existing_topic_key":"topic-7",'
                            '"perceived_task":"task 7","existing_task_key":"task-7",'
                            '"correction_present":false,"explanation_present":false,'
                            '"demonstrated_understanding":false,"knowledge_components":[],'
                            '"misconception_statement":"","evidence":"A continued question."}'
                        )
                    }
                }
            ]
        }

    state = SimulatedStudentState(
        current_topic="topic 7",
        current_topic_key="topic-7",
        current_task="task 7",
        current_task_key="task-7",
        topic_labels_by_key={f"topic-{index}": f"topic {index}" for index in range(8)},
        topic_aliases_by_key={
            f"topic-{index}": [f"alias {index}-{alias}" for alias in range(5)]
            for index in range(8)
        },
        task_labels_by_key={f"task-{index}": f"task {index}" for index in range(20)},
        task_topic_by_key={
            f"task-{index}": "topic-7" if index < 12 else f"topic-{index % 7}"
            for index in range(20)
        },
        topic_last_seen_turn_by_key={f"topic-{index}": index for index in range(8)},
        task_last_seen_turn_by_key={f"task-{index}": index for index in range(20)},
        misconceptions_by_id={
            f"topic-7:misconception-{index}": MisconceptionState(
                topic_key="topic-7",
                task_key=f"task-{index % 12}",
                belief=f"mistaken belief {index}",
                strength=index / 10,
                evidence_count=index,
            )
            for index in range(10)
        },
    )
    perception = LiteLLMStudentPerception(
        "test/model",
        topic_memory_limit=3,
        current_topic_task_memory_limit=2,
        recent_task_memory_limit=2,
        topic_alias_memory_limit=2,
        misconception_memory_limit=3,
        completion=completion,
    )

    observation = await perception.observe(tutor_message="Continue?", history=(), state=state)

    prompt = requests[0]["messages"][1]["content"]
    memory_json = prompt.split("<student_local_topic_and_task_memory>\n", 1)[1].split(
        "\n</student_local_topic_and_task_memory>", 1
    )[0]
    memory = json.loads(memory_json)
    assert observation.topic_key == "topic-7"
    assert len(memory["topics"]) == 3
    assert "topic-7" in memory["topics"]
    assert len(memory["tasks"]) == 4
    assert "task-7" in memory["tasks"]
    assert sum(topic == "topic-7" for topic in memory["task_topics"].values()) == 2
    assert all(len(aliases) <= 2 for aliases in memory["topic_aliases"].values())
    assert len(memory["misconceptions"]) == 3
    assert len(state.topic_labels_by_key) == 8
    assert len(state.task_labels_by_key) == 20
    assert len(state.misconceptions_by_id) == 10


@pytest.mark.asyncio
async def test_semantic_response_validator_returns_targeted_repair() -> None:
    async def completion(**request):
        assert request["response_format"] == {"type": "json_object"}
        assert "temperature" not in request
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"accepted":false,"safe_to_emit":true,'
                            '"observed_mode":"answer_correct",'
                            '"reason":"The reply is fully correct.",'
                            '"repair_hint":"Omit the normalization condition."}'
                        )
                    }
                }
            ]
        }

    validator = LiteLLMStudentResponseValidator("test/model", completion=completion)
    result = await validator.validate(
        message="A density is nonnegative and integrates to one.",
        decision=PolicyDecision(
            mode=StudentResponseMode.ANSWER_PARTIAL,
            reason="test",
        ),
        observation=StudentObservation(tutor_message="What defines a density?", response_expected=True),
    )

    assert result.accepted is False
    assert result.safe_to_emit is True
    assert result.observed_mode is StudentResponseMode.ANSWER_CORRECT
    assert result.repair_hint == "Omit the normalization condition."


@pytest.mark.asyncio
async def test_semantic_validator_rejects_inconsistent_accepted_mode() -> None:
    calls = 0

    async def completion(**request):
        nonlocal calls
        del request
        calls += 1
        return {
            "choices": [
                {
                    "message": {
                        "content": (
                            '{"accepted":true,"safe_to_emit":true,'
                            '"observed_mode":"answer_correct",'
                            '"reason":"","repair_hint":""}'
                        )
                    }
                }
            ]
        }

    result = await LiteLLMStudentResponseValidator("test/model", completion=completion).validate(
        message="A complete and correct answer.",
        decision=PolicyDecision(mode=StudentResponseMode.ANSWER_PARTIAL, reason="test"),
        observation=StudentObservation(tutor_message="Give a partial answer.", response_expected=True),
    )

    assert calls == 1
    assert result.accepted is False
    assert result.observed_mode is StudentResponseMode.ANSWER_CORRECT
    assert "answer_partial" in result.repair_hint
