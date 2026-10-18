"""Mode registry for the student simulator."""

from __future__ import annotations

from ..policy import SimulationMode
from .hidden_state import HiddenStateStudentMode

_MODES = {
    SimulationMode.HIDDEN_STATE_STUDENT: HiddenStateStudentMode(),
}


def get_simulator_mode(mode: SimulationMode):
    """Return the stateless handler for a validated simulation-mode enum value."""
    return _MODES[mode]
