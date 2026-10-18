"""Interface between policy decisions and model-generation instructions.

A mode has two jobs: choose the intended response behavior and render enough safe context
for a text generator to express that behavior. It must not turn tutor-private data into
prompt context.
"""

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
    """Contract implemented by each simulator mode registered in ``modes.__init__``."""

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
        persona: StudentPersona,
        observation: StudentObservation,
        simulated_state: SimulatedStudentState,
        decision: PolicyDecision,
        history_block: str,
    ) -> str:
        """Render the prompt-visible instruction for this mode."""


class PolicyBackedMode:
    """Shared decision logic for modes that use the deterministic Python policy."""

    mode: SimulationMode

    def choose_decision(
        self,
        persona: StudentPersona,
        observation: StudentObservation,
        simulated_state: SimulatedStudentState,
    ) -> PolicyDecision:
        """Delegate behavior selection while identifying the active mode."""
        return choose_response_mode(persona, observation, mode=self.mode, simulated_state=simulated_state)
