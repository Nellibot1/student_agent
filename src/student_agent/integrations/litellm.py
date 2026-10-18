"""Translate between ``student_agent`` and the external LiteLLM library.

LiteLLM is a separate Python library that can send requests to many AI providers through
one common interface. This file contains the small translator classes used by our project:

* ``LiteLLMStudentPerception`` asks a model what the tutor's visible message appears to mean.
* ``LiteLLMStudentGenerator`` asks a model to write one natural student reply.
* ``LiteLLMStudentResponseValidator`` checks whether that reply follows the behavior that
  the simulator selected.

The classes convert our Python objects into LiteLLM request dictionaries and convert the
answers back into our own validated objects. They do not decide how knowledgeable the
student is. That decision remains in the core policy code.

The external LiteLLM library is imported only when one of these adapters is created. This
allows the rest of ``student_agent`` to work with a local model, a test double, or no model
library at all. A different provider integration should implement the same small interfaces
without requiring changes to the engine or policy.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import re

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from ..conversation import ConversationMessage, ConversationRole, render_history
from ..engine import ValidationResult
from ..errors import StudentPerceptionError, StudentProviderError, StudentValidationError
from ..perception import MessageType, PerceivedFeedback, StudentObservation
from ..policy import PolicyDecision, SimulatedStudentState, StudentResponseMode

CompletionCallable = Callable[..., Awaitable[Any]]
DEFAULT_STUDENT_GENERATION_MAX_TOKENS = 1000
NON_RETRYABLE_PROVIDER_ERROR_NAMES = frozenset(
    {
        "AuthenticationError",
        "BadRequestError",
        "NotFoundError",
        "PermissionDeniedError",
        "UnsupportedParamsError",
    }
)


SEMANTIC_PERCEPTION_INSTRUCTION = """You interpret one visible tutor message from the learner's point of view.

Use only the visible conversation supplied below. Do not infer or request hidden grades,
answer keys, course IDs, plans, tools, or tutor state. Determine feedback on the learner's
immediately preceding answer, not whether isolated mathematical statements happen to use
words such as "correct" or "wrong".

Important cases:
- If the tutor confirms one part and then corrects, qualifies, or adds an important missing
  part, classify the overall feedback as partial.
- If the tutor contradicts the learner or remediates a substantive error without literally
  saying "wrong", classify it as incorrect when that meaning is clear.
- Ignore assessment words inside quoted definitions, examples, and newly posed questions.
- Use none when the tutor does not assess the preceding learner answer.
- correction_present means the tutor supplies or clearly points toward a corrected idea.
- explanation_present means the tutor explains why, gives reasoning, or gives an instructive
  example/hint rather than merely announcing an assessment.
- feedback_confidence is confidence in the learner's interpretation, from 0 to 1.
- perceived_topic is a short natural-language label, never a curriculum identifier. Preserve
  the prior perceived topic unless the visible message clearly changes topics.
- existing_topic_key must be one of the supplied local topic keys when the message belongs to
  an existing student-perceived topic; otherwise use an empty string.
- perceived_task is the specific question or learning task within the topic.
- existing_task_key must be one of the supplied local task keys when this continues the same
  task; otherwise use an empty string. Similar subject matter alone is not the same task.
- knowledge_components contains up to four concise propositions visibly taught or assessed.
- misconception_statement is the specific mistaken belief revealed by the visible exchange,
  or an empty string when no particular misconception is supported.
- existing_misconception_id must be an exact ID from the supplied misconception candidates
  when the newly observed belief has the same underlying meaning, even if wording differs.
  Use an empty string when there is no misconception or no semantically equivalent candidate.
  Similar topic alone is not enough to declare a match.
- demonstrated_understanding is true only when the tutor's wording indicates that the learner
  actually demonstrated understanding, not merely because the tutor explained something.

Return exactly one JSON object with these fields and no others:
{
  "feedback": "correct|partial|incorrect|none",
  "feedback_confidence": 0.0,
  "response_expected": true,
  "message_type": "question|feedback|feedback_and_question|explanation|hint|instruction|other",
  "perceived_topic": "short learner-facing topic",
  "existing_topic_key": "local key or empty string",
  "perceived_task": "specific current task",
  "existing_task_key": "local key or empty string",
  "correction_present": false,
  "explanation_present": false,
  "demonstrated_understanding": false,
  "knowledge_components": ["one visible proposition"],
  "misconception_statement": "specific mistaken belief or empty string",
  "existing_misconception_id": "supplied local misconception ID or empty string",
  "evidence": "one short explanation grounded in the visible wording"
}
"""


SEMANTIC_VALIDATION_INSTRUCTION = """Validate whether a generated learner reply follows its intended behavior.

Use only the supplied visible tutor message, student observation, intended reply mode, and
generated reply. Do not act as the tutor and do not require a hidden answer key. Check that:
- the reply addresses the visible message when a response is expected;
- correct, partial, and incorrect modes are behaviorally distinct;
- a partial reply leaves a meaningful gap instead of becoming fully complete;
- an incorrect or wrong-concept reply contains a plausible concrete wrong claim;
- clarification, uncertainty, and topic-jump modes have the requested shape;
- the reply is natural student language and does not reveal simulator instructions;
- when a specific active misconception is supplied for an incorrect mode, the reply is
  consistent with it rather than inventing an unrelated mistake.
- safe_to_emit is false when the reply is empty, reveals simulator instructions or private
  state, or is otherwise unsafe to expose as the student's visible message. A merely imperfect
  match to the intended behavior can still be safe to emit.
- observed_mode classifies what the generated reply actually does, independently of the intended
  mode. Use one of: answer_correct, answer_correct_unsure, correct_with_open_question,
  correct_computation_wrong_concept, answer_partial, answer_incorrect, ask_clarification,
  repeat_understanding_question, topic_jump, or nudge.
- accepted must be false when observed_mode materially differs from intended_mode.

Return exactly one JSON object:
{
  "accepted": true,
  "safe_to_emit": true,
  "observed_mode": "answer_correct",
  "reason": "brief assessment",
  "repair_hint": "specific instruction for regeneration, or empty when accepted"
}
"""


@dataclass(frozen=True)
class GenerationUsage:
    """A simple receipt for one AI-model request.

    Some model providers report how many input and output tokens they used. This class keeps
    that information in one consistent shape, even when providers return slightly different
    response objects. It is used for debugging, experiment reports, and cost tracking; it
    does not influence what the simulated student says.
    """

    model: str
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    total_tokens: int | None = None
    operation: str = ""
    attempt_number: int | None = None
    requested_max_tokens: int | None = None
    finish_reason: str = ""
    refusal: str = ""
    content_present: bool | None = None


class _SemanticObservation(BaseModel):
    """The exact JSON shape expected from the perception model.

    An AI model returns ordinary text, even when we ask it to return JSON. Before that result
    is allowed into the simulator, Pydantic checks every field against this private class.
    Missing fields, unknown fields, invalid category names, and numbers outside their allowed
    range are rejected. A valid value is later converted into the public
    ``StudentObservation`` type used by the rest of the package.

    The leading underscore means this is an internal helper, not part of the public API.
    """

    model_config = ConfigDict(extra="forbid")

    feedback: PerceivedFeedback
    feedback_confidence: float = Field(ge=0.0, le=1.0)
    response_expected: bool
    message_type: MessageType
    perceived_topic: str = Field(min_length=1, max_length=120)
    existing_topic_key: str = ""
    perceived_task: str = Field(default="general task", min_length=1, max_length=160)
    existing_task_key: str = ""
    correction_present: bool
    explanation_present: bool
    demonstrated_understanding: bool = False
    knowledge_components: list[str] = Field(default_factory=list, max_length=4)
    misconception_statement: str = Field(default="", max_length=300)
    existing_misconception_id: str = ""
    evidence: str = Field(default="", max_length=500)


class _SemanticValidation(BaseModel):
    """The exact JSON shape expected from the response-checking model.

    The validator model must say whether the generated student reply matches the intended
    behavior, whether the reply is safe to show, and what behavior it actually observed.
    Pydantic checks that answer here before the engine uses it. This prevents malformed or
    unexpected model output from being treated as a trustworthy validation result.

    The leading underscore means this is an internal helper, not part of the public API.
    """

    model_config = ConfigDict(extra="forbid")

    accepted: bool
    safe_to_emit: bool
    observed_mode: StudentResponseMode
    reason: str = Field(default="", max_length=500)
    repair_hint: str = Field(default="", max_length=500)


class LiteLLMStudentPerception:
    """Ask an AI model how the simulated student interprets the tutor's message.

    Think of this class as the student's reading-and-understanding step. It receives:

    * the latest message that the tutor showed to the student;
    * a short, visible conversation history; and
    * a limited selection of the student's own topic, task, and misconception memory.

    It turns that information into a LiteLLM request and asks for structured JSON. The JSON
    describes things such as whether the tutor asked a question, whether feedback sounded
    correct or incorrect, what topic the student thinks they are discussing, and whether a
    response appears to be expected.

    The result is checked carefully and converted into ``StudentObservation``. This class
    does not receive a hidden answer key or an official tutor grade, so its interpretation can
    be imperfect in the same way a real student's interpretation can be imperfect. It also
    limits how much history and memory is sent so requests do not grow forever.
    """

    def __init__(
        self,
        model: str,
        *,
        reasoning_effort: str = "",
        temperature: float | None = None,
        max_tokens: int = 800,
        retry_max_tokens: int | None = None,
        timeout: float = 20.0,
        history_limit: int = 6,
        topic_memory_limit: int = 6,
        current_topic_task_memory_limit: int = 10,
        recent_task_memory_limit: int = 5,
        topic_alias_memory_limit: int = 3,
        misconception_memory_limit: int = 8,
        max_attempts: int = 3,
        completion: CompletionCallable | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model cannot be empty")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if retry_max_tokens is not None and retry_max_tokens < max_tokens:
            raise ValueError("retry_max_tokens must be at least max_tokens")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if history_limit < 1:
            raise ValueError("history_limit must be at least 1")
        if topic_memory_limit < 1:
            raise ValueError("topic_memory_limit must be at least 1")
        if current_topic_task_memory_limit < 1:
            raise ValueError("current_topic_task_memory_limit must be at least 1")
        if recent_task_memory_limit < 0:
            raise ValueError("recent_task_memory_limit cannot be negative")
        if topic_alias_memory_limit < 1:
            raise ValueError("topic_alias_memory_limit must be at least 1")
        if misconception_memory_limit < 0:
            raise ValueError("misconception_memory_limit cannot be negative")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        # ``completion`` is injectable so tests and applications can supply a fake or a
        # compatible local client without importing LiteLLM.
        self._completion = completion or _load_completion()
        self.model = model.strip()
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.retry_max_tokens = retry_max_tokens or max_tokens * 2
        self.timeout = timeout
        self.history_limit = history_limit
        self.topic_memory_limit = topic_memory_limit
        self.current_topic_task_memory_limit = current_topic_task_memory_limit
        self.recent_task_memory_limit = recent_task_memory_limit
        self.topic_alias_memory_limit = topic_alias_memory_limit
        self.misconception_memory_limit = misconception_memory_limit
        self.max_attempts = max_attempts
        self.usage: list[GenerationUsage] = []
        self.last_error: str = ""

    async def observe(
        self,
        *,
        tutor_message: str,
        history: Sequence[ConversationMessage],
        state: SimulatedStudentState,
    ) -> StudentObservation:
        message = tutor_message.strip()
        if not message:
            raise ValueError("tutor_message cannot be empty")

        # The simulator keeps complete memory, but the model sees a bounded, relevant
        # window. This controls prompt growth without deleting long-term learner state.
        recent_history = history[-self.history_limit :]
        memory = _relevant_perception_memory(
            state,
            topic_limit=self.topic_memory_limit,
            current_topic_task_limit=self.current_topic_task_memory_limit,
            recent_task_limit=self.recent_task_memory_limit,
            alias_limit=self.topic_alias_memory_limit,
            misconception_limit=self.misconception_memory_limit,
        )
        # Tags separate the latest message from history and learner-local memory. They are
        # prompt structure only; no XML parser or external schema is involved.
        user_content = (
            f"<prior_perceived_topic>{state.current_topic or '(none yet)'}</prior_perceived_topic>\n"
            f"<student_local_topic_and_task_memory>\n{json.dumps(memory, ensure_ascii=False)}\n"
            f"</student_local_topic_and_task_memory>\n"
            f"<visible_conversation_history>\n{render_history(recent_history, limit=self.history_limit)}\n"
            f"</visible_conversation_history>\n"
            f"<latest_tutor_message>\n{message}\n</latest_tutor_message>"
        )
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SEMANTIC_PERCEPTION_INSTRUCTION},
                {"role": "user", "content": user_content},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.reasoning_effort:
            request["reasoning_effort"] = self.reasoning_effort

        last_exception: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            # Later attempts get a larger output budget because reasoning models may use
            # part of the budget internally before emitting the required JSON object.
            request["max_tokens"] = min(self.max_tokens * attempt, self.retry_max_tokens)
            try:
                response = await self._completion(**request)
                self.usage.append(
                    _usage_from_response(
                        response,
                        self.model,
                        operation="perception",
                        attempt_number=attempt,
                        requested_max_tokens=request["max_tokens"],
                    )
                )
                content = _completion_text(response)
                # Parse in two steps: recover the JSON object from common fenced responses,
                # then enforce the exact field names and ranges with Pydantic.
                parsed = _SemanticObservation.model_validate(_json_object(content))
                observation = _build_observation(parsed, message=message, history=history, state=state)
                self.last_error = ""
                return observation
            except Exception as exc:  # noqa: BLE001 - provider and schema failures share retry handling
                _raise_non_retryable_provider_error(exc, operation="semantic perception")
                last_exception = exc
                self.last_error = f"{type(exc).__name__}: {exc}"
                if attempt == self.max_attempts:
                    break
                retry_instruction = (
                    "Return only one valid JSON object matching the required schema."
                    if attempt + 1 == self.max_attempts
                    else (
                        "The previous semantic-perception attempt failed validation or transport. "
                        f"Return a fresh object matching the required schema. Error: {self.last_error}"
                    )
                )
                request["messages"] = [
                    {"role": "system", "content": SEMANTIC_PERCEPTION_INSTRUCTION},
                    {"role": "user", "content": user_content},
                    {"role": "user", "content": retry_instruction},
                ]
                await asyncio.sleep(0.1 * attempt)

        raise StudentPerceptionError(
            f"semantic perception failed after {self.max_attempts} attempts: {self.last_error}"
        ) from last_exception


class LiteLLMStudentResponseValidator:
    """Ask an AI model whether a generated reply follows the simulator's plan.

    The Python policy has already selected a behavior such as "give a partial answer" or
    "ask for clarification." The generator may accidentally write something different. This
    class sends the intended behavior and the generated reply to a model and asks it to compare
    them.

    It returns a ``ValidationResult`` saying:

    * whether the reply matches the intended behavior;
    * what behavior the reply actually shows;
    * whether the reply is safe to display; and
    * how the generator could repair it on another attempt.

    This class is not an exam grader and does not decide whether the student's subject-matter
    answer is objectively correct. It checks whether the wording matches the behavior that the
    simulator deliberately chose.
    """

    def __init__(
        self,
        model: str,
        *,
        reasoning_effort: str = "",
        temperature: float | None = None,
        max_tokens: int = 250,
        timeout: float = 20.0,
        max_attempts: int = 2,
        completion: CompletionCallable | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model cannot be empty")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if timeout <= 0:
            raise ValueError("timeout must be positive")
        if max_attempts < 1:
            raise ValueError("max_attempts must be at least 1")
        self._completion = completion or _load_completion()
        self.model = model.strip()
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self.max_attempts = max_attempts
        self.usage: list[GenerationUsage] = []
        self.last_error = ""

    async def validate(
        self,
        *,
        message: str,
        decision: PolicyDecision,
        observation: StudentObservation,
    ) -> ValidationResult:
        payload = {
            "visible_tutor_message": observation.tutor_message,
            "response_expected": observation.response_expected,
            "perceived_topic": observation.perceived_topic,
            "perceived_task": observation.perceived_task,
            "perceived_misconception": decision.target_misconception or observation.misconception_statement,
            "intended_mode": decision.mode.value,
            "intended_behavior": decision.realization_hint(),
            "generated_student_reply": message,
        }
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SEMANTIC_VALIDATION_INSTRUCTION},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
            "response_format": {"type": "json_object"},
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.reasoning_effort:
            request["reasoning_effort"] = self.reasoning_effort

        last_exception: Exception | None = None
        for attempt in range(1, self.max_attempts + 1):
            try:
                response = await self._completion(**request)
                self.usage.append(
                    _usage_from_response(
                        response,
                        self.model,
                        operation="validation",
                        attempt_number=attempt,
                        requested_max_tokens=request["max_tokens"],
                    )
                )
                parsed = _SemanticValidation.model_validate(_json_object(_completion_text(response)))
                # Require both the model's approval and agreement between its observed
                # behavior classification and the Python policy's intended behavior.
                accepted = parsed.accepted and parsed.observed_mode == decision.mode
                reason = parsed.reason.strip()
                repair_hint = parsed.repair_hint.strip()
                if parsed.accepted and not accepted:
                    reason = reason or (
                        f"The reply was classified as {parsed.observed_mode.value}, not {decision.mode.value}."
                    )
                    repair_hint = repair_hint or f"Rewrite the reply so it clearly realizes {decision.mode.value}."
                self.last_error = ""
                return ValidationResult(
                    accepted=accepted,
                    reason=reason,
                    repair_hint=repair_hint,
                    safe_to_emit=parsed.safe_to_emit,
                    observed_mode=parsed.observed_mode,
                )
            except Exception as exc:  # noqa: BLE001 - provider and schema failures share retry handling
                _raise_non_retryable_provider_error(exc, operation="semantic response validation")
                last_exception = exc
                self.last_error = f"{type(exc).__name__}: {exc}"
                if attempt < self.max_attempts:
                    request["messages"] = [
                        {"role": "system", "content": SEMANTIC_VALIDATION_INSTRUCTION},
                        {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                        {
                            "role": "user",
                            "content": (
                                "The previous validation attempt failed. Return a fresh object matching "
                                f"the required schema. Error: {self.last_error}"
                            ),
                        },
                    ]
                    await asyncio.sleep(0.1 * attempt)

        raise StudentValidationError(
            f"semantic response validation failed after {self.max_attempts} attempts: {self.last_error}"
        ) from last_exception


class LiteLLMStudentGenerator:
    """Ask an AI model to write the simulated student's visible reply.

    By the time this class is called, the engine and policy have already decided what kind of
    reply should be produced. The active simulation mode has converted that decision into a
    plain-language instruction. This adapter packages the instruction as a LiteLLM request,
    sends it to the selected model, and returns the model's text.

    The generator is therefore the writer, not the decision-maker. It does not calculate
    mastery, update confidence, choose whether the answer should be correct, or commit any
    student state. The engine performs those jobs and checks that the returned text is usable.
    """

    def __init__(
        self,
        model: str,
        *,
        reasoning_effort: str = "",
        temperature: float | None = None,
        max_tokens: int = DEFAULT_STUDENT_GENERATION_MAX_TOKENS,
        timeout: float = 30.0,
        completion: CompletionCallable | None = None,
    ) -> None:
        if not model.strip():
            raise ValueError("model cannot be empty")
        if max_tokens < 1:
            raise ValueError("max_tokens must be at least 1")
        if timeout <= 0:
            raise ValueError("timeout must be positive")

        if completion is None:
            completion = _load_completion()

        self.model = model.strip()
        self.reasoning_effort = reasoning_effort
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.timeout = timeout
        self._completion = completion
        self.usage: list[GenerationUsage] = []

    async def generate(self, *, instruction: str, tutor_message: str) -> str:
        """Send the package-generated instruction to LiteLLM and return plain text."""
        del tutor_message  # The exact visible message is already embedded in the instruction.
        request: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": instruction}],
            "max_tokens": self.max_tokens,
            "timeout": self.timeout,
        }
        if self.temperature is not None:
            request["temperature"] = self.temperature
        if self.reasoning_effort:
            request["reasoning_effort"] = self.reasoning_effort

        response = await self._completion(**request)
        content = _completion_text(response)
        self.usage.append(
            _usage_from_response(
                response,
                self.model,
                operation="generation",
                attempt_number=1,
                requested_max_tokens=self.max_tokens,
            )
        )
        return content


def _load_completion() -> CompletionCallable:
    """Load LiteLLM's model-calling function only when an adapter needs it.

    ``litellm`` in the import below is the external installed library, not this local file.
    Keeping the import inside this function means that importing the core ``student_agent``
    package does not require LiteLLM. Tests can also inject a fake completion function and
    avoid importing or contacting any external model service.
    """
    try:
        from litellm import acompletion
    except ImportError as exc:  # pragma: no cover - depends on installation extras
        raise RuntimeError(
            "LiteLLM support is not installed. Install it with `pip install 'student-agent[litellm]'`."
        ) from exc
    return acompletion


def _raise_non_retryable_provider_error(exc: Exception, *, operation: str) -> None:
    """Stop immediately when retrying cannot repair provider configuration.

    LiteLLM errors are optional runtime types, so this check uses class names across the
    exception chain rather than importing LiteLLM into the provider-neutral package core.
    Schema errors, timeouts, rate limits, and connection failures remain retryable.
    """
    current: BaseException | None = exc
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        class_names = {item.__name__ for item in type(current).mro()}
        if class_names & NON_RETRYABLE_PROVIDER_ERROR_NAMES:
            raise StudentProviderError(
                f"{operation} failed ({type(current).__name__}): {current}"
            ) from exc
        current = current.__cause__ or current.__context__


def _completion_text(response: Any) -> str:
    """Read text from dict-like or object-like LiteLLM response shapes."""
    choice = _first(_value(response, "choices"))
    message = _value(choice, "message")
    content = _value(message, "content")
    if not isinstance(content, str) or not content.strip():
        finish_reason = _optional_text(_value(choice, "finish_reason")) or "unknown"
        refusal = _optional_text(_value(message, "refusal")) or "none"
        raise ValueError(
            "LiteLLM returned no text in choices[0].message.content "
            f"(finish_reason={finish_reason}, refusal={refusal[:200]})"
        )
    return content.strip()


def _usage_from_response(
    response: Any,
    fallback_model: str,
    *,
    operation: str = "",
    attempt_number: int | None = None,
    requested_max_tokens: int | None = None,
) -> GenerationUsage:
    """Normalize optional provider usage fields without requiring one response class."""
    usage = _value(response, "usage")
    choice = _first(_value(response, "choices"))
    message = _value(choice, "message")
    content = _value(message, "content")
    return GenerationUsage(
        model=_value(response, "model") or fallback_model,
        prompt_tokens=_optional_int(_value(usage, "prompt_tokens")),
        completion_tokens=_optional_int(_value(usage, "completion_tokens")),
        total_tokens=_optional_int(_value(usage, "total_tokens")),
        operation=operation,
        attempt_number=attempt_number,
        requested_max_tokens=requested_max_tokens,
        finish_reason=_optional_text(_value(choice, "finish_reason")),
        refusal=_optional_text(_value(message, "refusal")),
        content_present=isinstance(content, str) and bool(content.strip()),
    )


def _json_object(content: str) -> dict[str, Any]:
    """Extract one JSON object, tolerating a model's optional Markdown code fence."""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.IGNORECASE)
        text = re.sub(r"\s*```$", "", text)
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("semantic perception returned no JSON object")
    payload = json.loads(text[start : end + 1])
    if not isinstance(payload, dict):
        raise TypeError("semantic perception JSON must be an object")
    return payload


def _build_observation(
    parsed: _SemanticObservation,
    *,
    message: str,
    history: Sequence[ConversationMessage],
    state: SimulatedStudentState,
) -> StudentObservation:
    """Turn validated model JSON into the package's canonical observation."""
    topic = parsed.perceived_topic.strip()
    if topic.lower() in {"same", "same topic", "unchanged", "current topic"}:
        topic = state.current_topic or "general discussion"
    # Reuse only keys that the model was actually shown. Otherwise create a stable local
    # key from the learner-facing label; external curriculum IDs never enter this layer.
    topic_key = parsed.existing_topic_key.strip()
    if topic_key not in state.topic_labels_by_key:
        topic_key = _topic_key(topic)

    task = parsed.perceived_task.strip()
    task_key = parsed.existing_task_key.strip()
    if task_key not in state.task_labels_by_key:
        task_key = _task_key(topic_key, task)

    turn_number = 1 + sum(item.role is ConversationRole.TUTOR for item in history)
    previous_attempts = state.attempts_by_task.get(task_key, 0)
    attempt_number = previous_attempts + 1 if parsed.response_expected else previous_attempts
    return StudentObservation(
        tutor_message=message,
        perceived_topic=topic,
        topic_key=topic_key,
        perceived_task=task,
        task_key=task_key,
        message_type=parsed.message_type,
        perceived_feedback=parsed.feedback,
        feedback_confidence=parsed.feedback_confidence,
        correction_present=parsed.correction_present,
        explanation_present=parsed.explanation_present,
        demonstrated_understanding=parsed.demonstrated_understanding,
        knowledge_components=[item.strip() for item in parsed.knowledge_components if item.strip()],
        misconception_statement=parsed.misconception_statement.strip(),
        matched_misconception_id=_validated_misconception_id(parsed, state, topic_key),
        evidence=parsed.evidence.strip(),
        perception_source="semantic",
        response_expected=parsed.response_expected,
        perceived_difficulty=_perceived_difficulty(message),
        mentioned_terms=_content_terms(message),
        turn_number=turn_number,
        attempt_number=attempt_number,
    )


def _relevant_perception_memory(
    state: SimulatedStudentState,
    *,
    topic_limit: int,
    current_topic_task_limit: int,
    recent_task_limit: int,
    alias_limit: int,
    misconception_limit: int,
) -> dict[str, Any]:
    """Select bounded current/recent registry entries without deleting full state."""
    # Selection is current-first and then recent. The full state is left untouched; this
    # function only decides what the next perception request needs to see.
    topic_keys = _ordered_recent_keys(
        state.topic_labels_by_key,
        state.topic_last_seen_turn_by_key,
        current_key=state.current_topic_key,
    )[:topic_limit]

    ordered_task_keys = _ordered_recent_keys(
        state.task_labels_by_key,
        state.task_last_seen_turn_by_key,
        current_key=state.current_task_key,
    )
    all_current_topic_tasks = [
        key
        for key in ordered_task_keys
        if key == state.current_task_key or state.task_topic_by_key.get(key) == state.current_topic_key
    ]
    current_topic_tasks = all_current_topic_tasks[:current_topic_task_limit]
    all_current_topic_task_keys = set(all_current_topic_tasks)
    recent_other_tasks = [key for key in ordered_task_keys if key not in all_current_topic_task_keys][
        :recent_task_limit
    ]
    task_keys = current_topic_tasks + recent_other_tasks
    relevant_topic_keys = set(topic_keys)
    misconception_items = [
        (misconception_id, misconception)
        for misconception_id, misconception in state.misconceptions_by_id.items()
        if misconception.topic_key in relevant_topic_keys
    ]
    misconception_insertion = {
        misconception_id: index for index, misconception_id in enumerate(state.misconceptions_by_id)
    }
    # Prefer active, task-relevant, current-topic, strong, and repeatedly observed beliefs.
    # Insertion order is the final tie-breaker, keeping selection deterministic.
    misconception_items.sort(
        key=lambda item: (
            item[1].status == "active",
            state.current_task_key in ({item[1].task_key} | set(item[1].task_keys)),
            item[1].topic_key == state.current_topic_key,
            item[1].strength,
            item[1].evidence_count,
            misconception_insertion[item[0]],
        ),
        reverse=True,
    )
    misconception_items = misconception_items[:misconception_limit]

    return {
        "current_topic_key": state.current_topic_key,
        "current_task_key": state.current_task_key,
        "topics": {key: state.topic_labels_by_key[key] for key in topic_keys},
        "topic_aliases": {
            key: state.topic_aliases_by_key.get(key, [])[-alias_limit:] for key in topic_keys
        },
        "tasks": {key: state.task_labels_by_key[key] for key in task_keys},
        "task_topics": {key: state.task_topic_by_key.get(key, "") for key in task_keys},
        "misconceptions": {
            misconception_id: {
                "belief": misconception.belief,
                "topic_key": misconception.topic_key,
                "task_keys": misconception.task_keys or ([misconception.task_key] if misconception.task_key else []),
                "strength": round(misconception.strength, 3),
                "status": misconception.status,
            }
            for misconception_id, misconception in misconception_items
        },
    }


def _validated_misconception_id(
    parsed: _SemanticObservation,
    state: SimulatedStudentState,
    topic_key: str,
) -> str:
    """Accept only a supplied, same-topic semantic-memory reference."""
    if not parsed.misconception_statement.strip():
        return ""
    misconception_id = parsed.existing_misconception_id.strip()
    misconception = state.misconceptions_by_id.get(misconception_id)
    if misconception is None or misconception.topic_key != topic_key:
        return ""
    return misconception_id


def _ordered_recent_keys(
    labels: dict[str, str],
    last_seen: dict[str, int],
    *,
    current_key: str,
) -> list[str]:
    """Order known keys current-first, then by recency and insertion order."""
    insertion_order = {key: index for index, key in enumerate(labels)}
    return sorted(
        labels,
        key=lambda key: (
            key == current_key,
            last_seen.get(key, 0),
            insertion_order[key],
        ),
        reverse=True,
    )


def _topic_key(topic: str) -> str:
    """Derive a stable local key from a normalized learner-facing topic label."""
    normalized = " ".join(topic.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:12]


def _task_key(topic_key: str, task: str) -> str:
    """Derive a stable local task key within its learner-perceived topic."""
    normalized = " ".join(task.lower().split())
    return hashlib.sha256(f"{topic_key}:{normalized}".encode()).hexdigest()[:12]


def _content_terms(message: str) -> list[str]:
    """Keep up to eight distinct terms as lightweight, inspectable message telemetry."""
    terms: list[str] = []
    for token in re.findall(r"[A-Za-z][A-Za-z0-9_-]{2,}", message.lower()):
        if token in terms:
            continue
        terms.append(token)
        if len(terms) == 8:
            break
    return terms


def _perceived_difficulty(message: str) -> float:
    """Estimate surface difficulty from length, long words, and mathematical notation.

    This is deliberately a small deterministic heuristic, not a correctness judgment.
    """
    words = message.split()
    long_words = sum(len(word.strip(".,?!:;()[]{}")) >= 10 for word in words)
    notation = sum(character in "=∫∑√^<>[]{}" for character in message)
    return min(1.0, 0.20 + len(words) / 120.0 + long_words * 0.025 + notation * 0.025)


def _value(value: Any, name: str) -> Any:
    if value is None:
        return None
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _first(value: Any) -> Any:
    if not value:
        return None
    return value[0]


def _optional_int(value: Any) -> int | None:
    return int(value) if value is not None else None


def _optional_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, default=str)
