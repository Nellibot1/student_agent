"""Mode registry for the student simulator."""

from __future__ import annotations

from ..policy import SimulationMode
from .hidden_state import HiddenStateStudentMode
from .state_prompted import StatePromptedStudentMode

_MODES = {
    SimulationMode.STATE_PROMPTED_STUDENT: StatePromptedStudentMode(),
    SimulationMode.HIDDEN_STATE_STUDENT: HiddenStateStudentMode(),
}


def get_simulator_mode(mode: SimulationMode):
    return _MODES[mode]
