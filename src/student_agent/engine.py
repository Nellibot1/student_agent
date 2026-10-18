"""Orchestrate one simulated student's conversation.

This is the best module to read first when learning the package.  A call to
``StudentSimulator.respond`` moves through the following stages:

1. interpret the visible tutor message (``StudentPerception``),
2. update the student's private learning state,
3. choose the kind of response the student should make, and
4. ask a text generator to write that response in natural language,
5. optionally validate the wording, and
6. commit the state and visible messages only after a usable reply exists.

The engine deliberately knows nothing about a tutor's answer key, lesson plan, database,
or grading state.  It only receives the text that a real student would see.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from .conversation import ConversationMessage, ConversationRole, render_history
from .errors import (
    InvalidStudentResponseError,
    StudentGenerationError,
    StudentPerceptionError,
    StudentValidationError,
)
from .modes import get_simulator_mode
from .perception import StudentObservation, StudentPerception
from .personas import DEFAULT_PERSONA, StudentPersona
from .policy import (
    PolicyDecision,
    SimulatedStudentState,
    SimulationMode,
    StudentResponseMode,
    evolve_simulated_state,
    initial_simulated_state,
)


class StudentTextGenerator(Protocol):
    """Minimal model boundary required by :class:`StudentSimulator`."""

    async def generate(self, *, instruction: str, tutor_message: str) -> str:
        """Generate exactly one student-facing reply."""


@dataclass(frozen=True)
class ValidationResult:
    """Result returned by an optional generated-response validator.

    ``accepted`` means the wording matches the selected behavior. ``safe_to_emit`` is a
    separate, weaker decision used only by the configured degraded-continuation path: a
    reply can miss the intended behavior yet still be safe to show to the tutor.
    """

    accepted: bool
    reason: str = ""
    repair_hint: str = ""
    safe_to_emit: bool = True
    observed_mode: StudentResponseMode | None = None


class StudentResponseValidator(Protocol):
    """Check output against the student's private policy intention."""

    async def validate(
        self,
        *,
        message: str,
        decision: PolicyDecision,
        observation: StudentObservation,
    ) -> ValidationResult:
        """Accept the message or provide guidance for an internal retry."""


class ValidationPolicy(StrEnum):
    """Choose which generated replies spend an extra semantic-validation call."""

    ALWAYS = "always"
    RISKY = "risky"
    OFF = "off"


class PerceptionFailurePolicy(StrEnum):
    """Choose whether exhausted semantic perception aborts or continues safely."""

    FAIL = "fail"
    CONTINUE = "continue"


class ValidationFailurePolicy(StrEnum):
    """Choose whether exhausted response validation aborts or emits a safe candidate."""

    FAIL = "fail"
    CONTINUE = "continue"


class ValidationStatus(StrEnum):
    """Telemetry describing how the committed student reply was validated."""

    SKIPPED = "skipped"
    PASSED = "passed"
    DEGRADED = "degraded"


RISKY_RESPONSE_MODES = frozenset(
    {
        StudentResponseMode.ANSWER_PARTIAL,
        StudentResponseMode.ANSWER_INCORRECT,
        StudentResponseMode.CORRECT_COMPUTATION_WRONG_CONCEPT,
        StudentResponseMode.TOPIC_JUMP,
    }
)


def response_requires_validation(mode: StudentResponseMode, policy: ValidationPolicy) -> bool:
    """Return whether a policy should invoke the semantic response validator."""
    if policy is ValidationPolicy.OFF:
        return False
    if policy is ValidationPolicy.ALWAYS:
        return True
    return mode in RISKY_RESPONSE_MODES


@dataclass(frozen=True)
class PreparedStudentTurn:
    """Provisional turn data that has not yet changed the live learner session.

    A prepared turn may be used more than once when validation asks for a wording retry.
    Every retry keeps the same observation, next state, and policy decision.
    """

    observation: StudentObservation
    state: SimulatedStudentState
    decision: PolicyDecision
    instruction: str


@dataclass(frozen=True)
class StudentReply:
    """One committed visible reply plus inspectable private student telemetry.

    Applications normally send only ``message`` to the tutor. The remaining fields are
    useful for experiment reports and debugging and should remain private from the tutor.
    """

    message: str
    observation: StudentObservation
    state: SimulatedStudentState
    decision: PolicyDecision
    generation_attempts: int = 1
    validation_status: ValidationStatus = ValidationStatus.SKIPPED
    validation_reason: str = ""
    observed_mode: StudentResponseMode | None = None


class StudentSnapshot(BaseModel):
    """Portable, versioned memory for one independent simulated learner.

    Snapshots contain data only. Model clients, network connections, and provider adapters
    must be supplied again when restoring with :meth:`StudentSimulator.from_snapshot`.
    """

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(2, ge=2)
    persona: StudentPersona
    state: SimulatedStudentState
    history: list[ConversationMessage] = Field(default_factory=list)
    mode: SimulationMode
    seed: str = ""


class CallableTextGenerator:
    """Wrap an async function when creating a full generator class would be unnecessary."""

    def __init__(self, generate: Callable[..., Awaitable[str]]) -> None:
        self._generate = generate

    async def generate(self, *, instruction: str, tutor_message: str) -> str:
        return await self._generate(instruction=instruction, tutor_message=tutor_message)


def prepare_student_turn(
    *,
    persona: StudentPersona,
    observation: StudentObservation,
    history: Sequence[ConversationMessage],
    mode: SimulationMode,
    state: SimulatedStudentState,
    repair_hint: str = "",
    decision_override: PolicyDecision | None = None,
) -> PreparedStudentTurn:
    """Build the model instruction for a turn without changing the live session.

    Keeping preparation separate from committing makes retries safe: a rejected model
    response can be regenerated without counting the same tutor message twice.
    """
    mode_handler = get_simulator_mode(mode)
    # Python chooses the behavior first.  The language model only turns that decision
    # into natural student text; it does not decide how knowledgeable the student is.
    decision = decision_override or mode_handler.choose_decision(persona, observation, state)
    if repair_hint:
        decision = decision.model_copy(update={"repair_hint": repair_hint})
    instruction = mode_handler.render_instruction(
        persona,
        observation,
        state,
        decision,
        render_history(history),
    )
    return PreparedStudentTurn(
        observation=observation,
        state=state,
        decision=decision,
        instruction=instruction,
    )


class StudentSimulator:
    """One stateful learner session driven only by visible tutor messages.

    Create a separate instance per simulated learner.  ``generator`` and ``perception``
    are injected adapters, which is what lets the core work with hosted models, local
    models, or test doubles.  Call :meth:`respond` for the usual one-turn API.
    """

    def __init__(
        self,
        generator: StudentTextGenerator,
        *,
        perception: StudentPerception,
        persona: StudentPersona = DEFAULT_PERSONA,
        mode: SimulationMode = SimulationMode.HIDDEN_STATE_STUDENT,
        state: SimulatedStudentState | None = None,
        history: Sequence[ConversationMessage] | None = None,
        seed: str | int = "",
        validator: StudentResponseValidator | None = None,
        validation_policy: ValidationPolicy | str = ValidationPolicy.RISKY,
        perception_failure_policy: PerceptionFailurePolicy | str = PerceptionFailurePolicy.FAIL,
        validation_failure_policy: ValidationFailurePolicy | str = ValidationFailurePolicy.FAIL,
        max_generation_attempts: int = 1,
    ) -> None:
        if max_generation_attempts < 1:
            raise ValueError("max_generation_attempts must be at least 1")
        self.generator = generator
        self.persona = persona
        self.mode = mode
        # Copy caller-owned models so later simulator updates cannot unexpectedly mutate
        # an object that the caller is still holding.
        self.state = state.model_copy(deep=True) if state is not None else initial_simulated_state(persona)
        self.history = [message.model_copy(deep=True) for message in (history or ())]
        self.perception = perception
        self.seed = str(seed)
        self.validator = validator
        self.validation_policy = ValidationPolicy(validation_policy)
        self.perception_failure_policy = PerceptionFailurePolicy(perception_failure_policy)
        self.validation_failure_policy = ValidationFailurePolicy(validation_failure_policy)
        self.max_generation_attempts = max_generation_attempts

    def snapshot(self) -> StudentSnapshot:
        """Return an independent, serializable copy of the learner's committed memory."""
        return StudentSnapshot(
            persona=self.persona.model_copy(deep=True),
            state=self.state.model_copy(deep=True),
            history=[message.model_copy(deep=True) for message in self.history],
            mode=self.mode,
            seed=self.seed,
        )

    @classmethod
    def from_snapshot(
        cls,
        snapshot: StudentSnapshot,
        *,
        generator: StudentTextGenerator,
        perception: StudentPerception,
        validator: StudentResponseValidator | None = None,
        validation_policy: ValidationPolicy | str = ValidationPolicy.RISKY,
        perception_failure_policy: PerceptionFailurePolicy | str = PerceptionFailurePolicy.FAIL,
        validation_failure_policy: ValidationFailurePolicy | str = ValidationFailurePolicy.FAIL,
        max_generation_attempts: int = 1,
    ) -> StudentSimulator:
        """Restore learner data while injecting fresh local runtime adapters."""
        if snapshot.schema_version != 2:
            raise ValueError(f"unsupported snapshot schema version: {snapshot.schema_version}")
        return cls(
            generator,
            persona=snapshot.persona,
            mode=snapshot.mode,
            state=snapshot.state,
            history=snapshot.history,
            perception=perception,
            seed=snapshot.seed,
            validator=validator,
            validation_policy=validation_policy,
            perception_failure_policy=perception_failure_policy,
            validation_failure_policy=validation_failure_policy,
            max_generation_attempts=max_generation_attempts,
        )

    async def prepare(self, tutor_message: str, *, repair_hint: str = "") -> PreparedStudentTurn:
        """Interpret visible text and calculate a provisional turn without committing it.

        Advanced integrations may call this method to inspect the intended behavior before
        generation. The normal public workflow should call :meth:`respond` instead.
        """
        message = tutor_message.strip()
        if not message:
            raise ValueError("tutor_message cannot be empty")

        # Perception is intentionally outside the policy.  It converts unstructured tutor
        # text into the student's *fallible* interpretation, never authoritative metadata.
        try:
            observation = await self.perception.observe(
                tutor_message=message,
                history=tuple(self.history),
                state=self.state.model_copy(deep=True),
            )
        except StudentPerceptionError as exc:
            if self.perception_failure_policy is PerceptionFailurePolicy.FAIL:
                raise
            return self._prepare_degraded_turn(message, exc, repair_hint=repair_hint)
        observation = observation.model_copy(
            update={
                "tutor_message": message,
                "experiment_seed": self.seed,
            }
        )
        # ``next_state`` is still provisional.  It becomes the live state only after a
        # usable student reply has been generated below in ``_commit_reply``.
        next_state = evolve_simulated_state(self.state, self.persona, observation)
        return prepare_student_turn(
            persona=self.persona,
            observation=observation,
            history=tuple(self.history),
            mode=self.mode,
            state=next_state,
            repair_hint=repair_hint,
        )

    def _prepare_degraded_turn(
        self,
        tutor_message: str,
        error: StudentPerceptionError,
        *,
        repair_hint: str = "",
    ) -> PreparedStudentTurn:
        """Continue with clarification while preserving all private learner state."""
        observation = StudentObservation(
            tutor_message=tutor_message,
            perceived_topic=self.state.current_topic or "general discussion",
            topic_key=self.state.current_topic_key or "general",
            perceived_task=self.state.current_task or "understand the tutor's latest message",
            task_key=self.state.current_task_key or "general-task",
            perception_source="semantic_degraded",
            perception_error=str(error),
            response_expected=True,
            evidence="Semantic perception was unavailable; private learner state was preserved.",
            turn_number=1 + sum(item.role is ConversationRole.TUTOR for item in self.history),
            attempt_number=self.state.attempts_by_task.get(self.state.current_task_key, 0),
            experiment_seed=self.seed,
        )
        decision = PolicyDecision(
            mode=StudentResponseMode.ASK_CLARIFICATION,
            reason="semantic perception was unavailable, so ask rather than invent an interpretation",
            repair_hint=(
                repair_hint
                or "Ask a brief, focused clarification about what the tutor means; do not make a new factual claim."
            ),
        )
        return prepare_student_turn(
            persona=self.persona,
            observation=observation,
            history=tuple(self.history),
            mode=self.mode,
            state=self.state.model_copy(deep=True),
            decision_override=decision,
        )

    async def respond(self, tutor_message: str, *, repair_hint: str = "") -> StudentReply:
        """Generate, optionally validate, and commit one student reply.

        If every candidate fails, the method raises and leaves both state and history
        unchanged.  This all-or-nothing behavior is important for resumable experiments.
        """
        # Think of the method as a small transaction: prepare on copies, generate and
        # validate, then commit exactly once. Any raised exception leaves the live learner
        # at the state it had before this call.
        prepared = await self.prepare(tutor_message, repair_hint=repair_hint)
        validation_reason = ""

        for attempt in range(1, self.max_generation_attempts + 1):
            # Each loop produces a fresh wording for the same already-selected behavior.
            try:
                message = (
                    await self.generator.generate(
                        instruction=prepared.instruction,
                        tutor_message=prepared.observation.tutor_message,
                    )
                ).strip()
            except Exception as exc:
                raise StudentGenerationError("student text generation failed") from exc

            if not message:
                validation = ValidationResult(
                    accepted=False,
                    reason="the text generator returned an empty response",
                    repair_hint="Return one non-empty student message.",
                    safe_to_emit=False,
                )
                validation_status = ValidationStatus.SKIPPED
            elif self.validator is None or not response_requires_validation(
                prepared.decision.mode, self.validation_policy
            ):
                # Validation is optional.  The default "risky" policy spends an extra
                # model call only where a subtle behavior mismatch matters most.
                validation = ValidationResult(accepted=True)
                validation_status = ValidationStatus.SKIPPED
            else:
                try:
                    validation = await self.validator.validate(
                        message=message,
                        decision=prepared.decision,
                        observation=prepared.observation,
                    )
                except StudentValidationError as exc:
                    validation = ValidationResult(
                        accepted=False,
                        reason=str(exc),
                        repair_hint="Follow the intended student behavior exactly in one natural reply.",
                        safe_to_emit=_reply_is_locally_safe(message),
                    )
                validation_status = ValidationStatus.PASSED

            if validation.accepted:
                return self._commit_reply(
                    prepared,
                    message,
                    attempt=attempt,
                    validation_status=validation_status,
                    validation_reason=validation.reason,
                    observed_mode=validation.observed_mode,
                )

            validation_reason = validation.reason
            if (
                attempt == self.max_generation_attempts
                and self.validation_failure_policy is ValidationFailurePolicy.CONTINUE
                and message
                and validation.safe_to_emit
                and _reply_is_locally_safe(message)
            ):
                # "Continue" is intended for interactive use. The candidate is kept only
                # when both the semantic validator and a local leakage check consider it
                # safe to expose, and the degraded status remains visible in telemetry.
                return self._commit_reply(
                    prepared,
                    message,
                    attempt=attempt,
                    validation_status=ValidationStatus.DEGRADED,
                    validation_reason=validation.reason,
                    observed_mode=validation.observed_mode,
                )
            if attempt < self.max_generation_attempts:
                # Keep the same observation, state, and policy choice.  Only add the
                # validator's repair advice to the next generation instruction.
                prepared = prepare_student_turn(
                    persona=self.persona,
                    observation=prepared.observation,
                    history=tuple(self.history),
                    mode=self.mode,
                    state=prepared.state,
                    repair_hint=validation.repair_hint or validation.reason,
                    decision_override=prepared.decision,
                )

        raise InvalidStudentResponseError(validation_reason or "generated student response did not pass validation")

    async def record_external_reply(self, tutor_message: str, student_message: str) -> StudentReply:
        """Interpret tutor text and commit a visible student reply supplied by an application.

        This supports experiment protocols that must inject a natural student action, such
        as requesting the next scheduled topic. The private learner state still evolves
        only from semantic perception of the visible tutor message.
        """
        message = student_message.strip()
        if not message:
            raise ValueError("student_message cannot be empty")
        prepared = await self.prepare(tutor_message)
        return self._commit_reply(
            prepared,
            message,
            attempt=0,
            validation_status=ValidationStatus.SKIPPED,
            validation_reason="externally supplied visible student reply",
        )

    def _commit_reply(
        self,
        prepared: PreparedStudentTurn,
        message: str,
        *,
        attempt: int,
        validation_status: ValidationStatus,
        validation_reason: str = "",
        observed_mode: StudentResponseMode | None = None,
    ) -> StudentReply:
        """Commit one accepted or explicitly degraded visible exchange."""
        # This is the only mutation point for a normal turn.  Preparation and validation
        # above operate on copies so a failed attempt cannot leave a half-finished turn.
        self.state = prepared.state.model_copy(deep=True)
        self.history.extend(
            [
                ConversationMessage(
                    role=ConversationRole.TUTOR,
                    content=prepared.observation.tutor_message,
                ),
                ConversationMessage(role=ConversationRole.STUDENT, content=message),
            ]
        )
        return StudentReply(
            message=message,
            observation=prepared.observation,
            state=prepared.state,
            decision=prepared.decision,
            generation_attempts=attempt,
            validation_status=validation_status,
            validation_reason=validation_reason,
            observed_mode=observed_mode,
        )


_UNSAFE_REPLY_MARKERS = (
    "<private_reply_tendency>",
    "<simulated_student_state>",
    "<student_interpretation>",
    "simulation_mode:",
    "my hidden policy",
    "my response mode",
    "these simulator instructions",
)


def _reply_is_locally_safe(message: str) -> bool:
    """Reject obvious simulator-instruction leakage before degraded emission."""
    lowered = message.casefold()
    return bool(message.strip()) and not any(marker in lowered for marker in _UNSAFE_REPLY_MARKERS)
