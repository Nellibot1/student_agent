"""Black-box student engine driven only by visible conversation."""

from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass
from importlib.resources import files
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from .conversation import ConversationMessage, ConversationRole, render_history
from .errors import InvalidStudentResponseError, StudentGenerationError
from .modes import get_simulator_mode
from .perception import RuleBasedPerception, StudentObservation, StudentPerception
from .personas import DEFAULT_PERSONA, StudentPersona
from .policy import (
    PolicyDecision,
    SimulatedStudentState,
    SimulationMode,
    evolve_simulated_state,
    initial_simulated_state,
)


class StudentTextGenerator(Protocol):
    """Minimal model boundary required by :class:`StudentSimulator`."""

    async def generate(self, *, instruction: str, tutor_message: str) -> str:
        """Generate exactly one student-facing reply."""


@dataclass(frozen=True)
class ValidationResult:
    """Result returned by an optional generated-response validator."""

    accepted: bool
    reason: str = ""
    repair_hint: str = ""


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


@dataclass(frozen=True)
class PreparedStudentTurn:
    """Private perception, policy result, and generated instruction for one turn."""

    observation: StudentObservation
    state: SimulatedStudentState
    decision: PolicyDecision
    instruction: str


@dataclass(frozen=True)
class StudentReply:
    """One visible reply plus inspectable private student telemetry."""

    message: str
    observation: StudentObservation
    state: SimulatedStudentState
    decision: PolicyDecision
    generation_attempts: int = 1


class StudentSnapshot(BaseModel):
    """Portable, versioned memory for one independent simulated learner."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(2, ge=2)
    persona: StudentPersona
    state: SimulatedStudentState
    history: list[ConversationMessage] = Field(default_factory=list)
    mode: SimulationMode
    seed: str = ""


class CallableTextGenerator:
    """Adapt an async function to the small generator protocol."""

    def __init__(self, generate: Callable[..., Awaitable[str]]) -> None:
        self._generate = generate

    async def generate(self, *, instruction: str, tutor_message: str) -> str:
        return await self._generate(instruction=instruction, tutor_message=tutor_message)


def default_prompt_template() -> str:
    """Load the package-owned prompt template."""
    return files("student_agent").joinpath("simulator_prompt.md").read_text(encoding="utf-8")


def prepare_student_turn(
    *,
    persona: StudentPersona,
    observation: StudentObservation,
    history: Sequence[ConversationMessage],
    mode: SimulationMode,
    state: SimulatedStudentState,
    prompt_template: str | None = None,
    repair_hint: str = "",
) -> PreparedStudentTurn:
    """Prepare a turn from the student's own perception and private memory."""
    mode_handler = get_simulator_mode(mode)
    decision = mode_handler.choose_decision(persona, observation, state)
    if repair_hint:
        decision = decision.model_copy(update={"repair_hint": repair_hint})
    instruction = mode_handler.render_instruction(
        prompt_template if prompt_template is not None else default_prompt_template(),
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
    """Stateful black-box learner that receives only visible tutor messages."""

    def __init__(
        self,
        generator: StudentTextGenerator,
        *,
        persona: StudentPersona = DEFAULT_PERSONA,
        mode: SimulationMode = SimulationMode.HIDDEN_STATE_STUDENT,
        state: SimulatedStudentState | None = None,
        history: Sequence[ConversationMessage] | None = None,
        perception: StudentPerception | None = None,
        prompt_template: str | None = None,
        seed: str | int = "",
        validator: StudentResponseValidator | None = None,
        max_generation_attempts: int = 1,
    ) -> None:
        if max_generation_attempts < 1:
            raise ValueError("max_generation_attempts must be at least 1")
        self.generator = generator
        self.persona = persona
        self.mode = mode
        self.state = state.model_copy(deep=True) if state is not None else initial_simulated_state(persona)
        self.history = [message.model_copy(deep=True) for message in (history or ())]
        self.perception = perception or RuleBasedPerception()
        self.prompt_template = prompt_template
        self.seed = str(seed)
        self.validator = validator
        self.max_generation_attempts = max_generation_attempts

    def snapshot(self) -> StudentSnapshot:
        """Return serializable private memory; runtime adapters are excluded."""
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
        perception: StudentPerception | None = None,
        prompt_template: str | None = None,
        validator: StudentResponseValidator | None = None,
        max_generation_attempts: int = 1,
    ) -> StudentSimulator:
        """Restore a learner while injecting local runtime adapters."""
        if snapshot.schema_version != 2:
            raise ValueError(f"unsupported snapshot schema version: {snapshot.schema_version}")
        return cls(
            generator,
            persona=snapshot.persona,
            mode=snapshot.mode,
            state=snapshot.state,
            history=snapshot.history,
            perception=perception,
            prompt_template=prompt_template,
            seed=snapshot.seed,
            validator=validator,
            max_generation_attempts=max_generation_attempts,
        )

    async def prepare(self, tutor_message: str, *, repair_hint: str = "") -> PreparedStudentTurn:
        """Interpret visible text and prepare a turn without committing state."""
        message = tutor_message.strip()
        if not message:
            raise ValueError("tutor_message cannot be empty")

        observation = await self.perception.observe(
            tutor_message=message,
            history=tuple(self.history),
            state=self.state.model_copy(deep=True),
        )
        observation = observation.model_copy(
            update={
                "tutor_message": message,
                "experiment_seed": self.seed,
            }
        )
        next_state = evolve_simulated_state(self.state, self.persona, observation)
        return prepare_student_turn(
            persona=self.persona,
            observation=observation,
            history=tuple(self.history),
            mode=self.mode,
            state=next_state,
            prompt_template=self.prompt_template,
            repair_hint=repair_hint,
        )

    async def respond(self, tutor_message: str, *, repair_hint: str = "") -> StudentReply:
        """Respond to visible text without receiving tutor-system metadata."""
        prepared = await self.prepare(tutor_message, repair_hint=repair_hint)
        validation_reason = ""

        for attempt in range(1, self.max_generation_attempts + 1):
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
                )
            elif self.validator is None:
                validation = ValidationResult(accepted=True)
            else:
                validation = await self.validator.validate(
                    message=message,
                    decision=prepared.decision,
                    observation=prepared.observation,
                )

            if validation.accepted:
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
                )

            validation_reason = validation.reason
            if attempt < self.max_generation_attempts:
                prepared = prepare_student_turn(
                    persona=self.persona,
                    observation=prepared.observation,
                    history=tuple(self.history),
                    mode=self.mode,
                    state=prepared.state,
                    prompt_template=self.prompt_template,
                    repair_hint=validation.repair_hint or validation.reason,
                )

        raise InvalidStudentResponseError(validation_reason or "generated student response did not pass validation")
