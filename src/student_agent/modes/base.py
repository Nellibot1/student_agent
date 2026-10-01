"""Interface and prompt helpers for student simulator modes."""

from __future__ import annotations

from typing import Protocol

from ..personas import StudentPersona
from ..policy import (
    PolicyDecision,
    SimulatedStudentState,
    SimulationContext,
    SimulationMode,
    choose_response_mode,
)


class StudentSimulatorMode(Protocol):
    mode: SimulationMode

    def choose_decision(
        self,
        persona: StudentPersona,
        context: SimulationContext,
        simulated_state: SimulatedStudentState,
    ) -> PolicyDecision:
        """Choose the target behavior for this student turn."""

    def render_instruction(
        self,
        prompt_template: str,
        persona: StudentPersona,
        context: SimulationContext,
        simulated_state: SimulatedStudentState,
        decision: PolicyDecision,
    ) -> str:
        """Render the prompt-visible instruction for this mode."""


class PolicyBackedMode:
    mode: SimulationMode

    def choose_decision(
        self,
        persona: StudentPersona,
        context: SimulationContext,
        simulated_state: SimulatedStudentState,
    ) -> PolicyDecision:
        return choose_response_mode(persona, context, mode=self.mode, simulated_state=simulated_state)


def fill_simulator_template(
    prompt_template: str,
    *,
    mode: SimulationMode,
    persona_block: str,
    context_block: str,
    decision_block: str,
    simulated_state_block: str,
    response_mode_reference: str,
) -> str:
    return (
        prompt_template.replace("{{persona}}", persona_block)
        .replace("{{context}}", context_block)
        .replace("{{decision}}", decision_block)
        .replace("{{simulated_state}}", simulated_state_block)
        .replace("{{mode}}", mode.value)
        .replace("{{response_mode_reference}}", response_mode_reference)
    )
