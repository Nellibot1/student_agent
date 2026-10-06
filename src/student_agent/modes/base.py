"""Interface and prompt helpers for student simulator modes."""

from __future__ import annotations

from typing import Protocol

from ..perception import StudentObservation
from ..personas import StudentPersona
from ..policy import (
    PolicyDecision,
    SimulatedStudentState,
    SimulationMode,
    choose_response_mode,
)


class StudentSimulatorMode(Protocol):
    mode: SimulationMode

    def choose_decision(
        self,
        persona: StudentPersona,
        observation: StudentObservation,
        simulated_state: SimulatedStudentState,
    ) -> PolicyDecision:
        """Choose the target behavior for this student turn."""

    def render_instruction(
        self,
        prompt_template: str,
        persona: StudentPersona,
        observation: StudentObservation,
        simulated_state: SimulatedStudentState,
        decision: PolicyDecision,
        history_block: str,
    ) -> str:
        """Render the prompt-visible instruction for this mode."""


class PolicyBackedMode:
    mode: SimulationMode

    def choose_decision(
        self,
        persona: StudentPersona,
        observation: StudentObservation,
        simulated_state: SimulatedStudentState,
    ) -> PolicyDecision:
        return choose_response_mode(persona, observation, mode=self.mode, simulated_state=simulated_state)


def fill_simulator_template(
    prompt_template: str,
    *,
    mode: SimulationMode,
    persona_block: str,
    observation_block: str,
    history_block: str,
    tutor_message: str,
    decision_block: str,
    simulated_state_block: str,
    response_mode_reference: str,
) -> str:
    return (
        prompt_template.replace("{{persona}}", persona_block)
        .replace("{{observation}}", observation_block)
        .replace("{{conversation_history}}", history_block)
        .replace("{{tutor_message}}", tutor_message)
        .replace("{{decision}}", decision_block)
        .replace("{{simulated_state}}", simulated_state_block)
        .replace("{{mode}}", mode.value)
        .replace("{{response_mode_reference}}", response_mode_reference)
    )
