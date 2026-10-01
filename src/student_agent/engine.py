"""Application-independent student simulation engine.

The engine owns persona/state/policy orchestration and delegates text generation to an
injected implementation. It deliberately has no dependency on a model provider,
application framework, experiment runner, or persistence system.
"""

from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from importlib.resources import files
from typing import Protocol

from pydantic import BaseModel, ConfigDict, Field

from .errors import InvalidStudentResponseError, StudentGenerationError
from .modes import get_simulator_mode
from .personas import DEFAULT_PERSONA, StudentPersona
from .policy import (
    PolicyDecision,
    SimulatedStudentState,
    SimulationContext,
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
    """Optional boundary for checking whether generated text matches the policy."""

    async def validate(
        self,
        *,
        message: str,
        decision: PolicyDecision,
        context: SimulationContext,
    ) -> ValidationResult:
        """Accept the message or provide guidance for one retry."""


@dataclass(frozen=True)
class PreparedStudentTurn:
    """Policy result and instruction prepared before calling a text model."""

    context: SimulationContext
    state: SimulatedStudentState
    decision: PolicyDecision
    instruction: str


@dataclass(frozen=True)
class StudentReply:
    """One generated student reply plus inspectable simulator telemetry."""

    message: str
    context: SimulationContext
    state: SimulatedStudentState
    decision: PolicyDecision
    generation_attempts: int = 1


class StudentSnapshot(BaseModel):
    """Portable, versioned state for saving and restoring one simulated learner."""

    model_config = ConfigDict(extra="forbid")

    schema_version: int = Field(1, ge=1)
    persona: StudentPersona
    state: SimulatedStudentState
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
    context: SimulationContext,
    mode: SimulationMode,
    state: SimulatedStudentState,
    prompt_template: str | None = None,
    repair_hint: str = "",
) -> PreparedStudentTurn:
    """Prepare a turn from explicit inputs without accessing global services."""
    mode_handler = get_simulator_mode(mode)
    decision = mode_handler.choose_decision(persona, context, state)
    if repair_hint:
        decision = decision.model_copy(update={"repair_hint": repair_hint})
    instruction = mode_handler.render_instruction(
        prompt_template if prompt_template is not None else default_prompt_template(),
        persona,
        context,
        state,
        decision,
    )
    return PreparedStudentTurn(
        context=context,
        state=state,
        decision=decision,
        instruction=instruction,
    )


class StudentSimulator:
    """Stateful, reusable simulated student.

    Callers provide a text generator and feed visible tutor messages plus optional
    neutral experiment context. The simulator maintains its own learner state.
    """

    def __init__(
        self,
        generator: StudentTextGenerator,
        *,
        persona: StudentPersona = DEFAULT_PERSONA,
        mode: SimulationMode = SimulationMode.HIDDEN_STATE_STUDENT,
        state: SimulatedStudentState | None = None,
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
        self.prompt_template = prompt_template
        self.seed = str(seed)
        self.validator = validator
        self.max_generation_attempts = max_generation_attempts

    def snapshot(self) -> StudentSnapshot:
        """Return a serializable snapshot; model providers are intentionally excluded."""
        return StudentSnapshot(
            persona=self.persona.model_copy(deep=True),
            state=self.state.model_copy(deep=True),
            mode=self.mode,
            seed=self.seed,
        )

    @classmethod
    def from_snapshot(
        cls,
        snapshot: StudentSnapshot,
        *,
        generator: StudentTextGenerator,
        prompt_template: str | None = None,
        validator: StudentResponseValidator | None = None,
        max_generation_attempts: int = 1,
    ) -> StudentSimulator:
        """Restore a learner while injecting this application's model boundary."""
        if snapshot.schema_version != 1:
            raise ValueError(f"unsupported snapshot schema version: {snapshot.schema_version}")
        return cls(
            generator,
            persona=snapshot.persona,
            mode=snapshot.mode,
            state=snapshot.state,
            prompt_template=prompt_template,
            seed=snapshot.seed,
            validator=validator,
            max_generation_attempts=max_generation_attempts,
        )

    def prepare(
        self,
        tutor_message: str,
        *,
        context: SimulationContext | None = None,
        repair_hint: str = "",
    ) -> PreparedStudentTurn:
        """Advance hidden state and prepare one response instruction."""
        base = context or SimulationContext()
        turn_context = base.model_copy(
            update={
                "tutor_message": tutor_message,
                "experiment_seed": self.seed or base.experiment_seed,
            }
        )
        self.state = evolve_simulated_state(self.state, self.persona, turn_context)
        return prepare_student_turn(
            persona=self.persona,
            context=turn_context,
            mode=self.mode,
            state=self.state,
            prompt_template=self.prompt_template,
            repair_hint=repair_hint,
        )

    async def respond(
        self,
        tutor_message: str,
        *,
        context: SimulationContext | None = None,
        repair_hint: str = "",
    ) -> StudentReply:
        """Generate, optionally validate, and return one student response."""
        prepared = self.prepare(tutor_message, context=context, repair_hint=repair_hint)
        validation_reason = ""

        for attempt in range(1, self.max_generation_attempts + 1):
            try:
                message = (
                    await self.generator.generate(
                        instruction=prepared.instruction,
                        tutor_message=prepared.context.tutor_message,
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
                    context=prepared.context,
                )

            if validation.accepted:
                return StudentReply(
                    message=message,
                    context=prepared.context,
                    state=prepared.state,
                    decision=prepared.decision,
                    generation_attempts=attempt,
                )

            validation_reason = validation.reason
            if attempt < self.max_generation_attempts:
                prepared = prepare_student_turn(
                    persona=self.persona,
                    context=prepared.context,
                    mode=self.mode,
                    state=self.state,
                    prompt_template=self.prompt_template,
                    repair_hint=validation.repair_hint or validation.reason,
                )

        raise InvalidStudentResponseError(
            validation_reason or "generated student response did not pass validation"
        )
