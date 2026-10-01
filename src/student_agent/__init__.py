"""Provider- and application-independent simulated student public API.

Framework- and model-provider-specific adapters live outside this package.
"""

from .controls import StudentControlProfile, StudentPolicyScalars
from .engine import (
    CallableTextGenerator,
    PreparedStudentTurn,
    StudentReply,
    StudentResponseValidator,
    StudentSimulator,
    StudentSnapshot,
    StudentTextGenerator,
    ValidationResult,
    prepare_student_turn,
)
from .errors import InvalidStudentResponseError, StudentAgentError, StudentGenerationError
from .personas import DEFAULT_PERSONA, StudentPersona, load_persona
from .policy import (
    Grade,
    PolicyDecision,
    SimulatedStudentState,
    SimulationContext,
    SimulationMode,
    StudentResponseMode,
)

__all__ = [
    "DEFAULT_PERSONA",
    "CallableTextGenerator",
    "Grade",
    "InvalidStudentResponseError",
    "PolicyDecision",
    "PreparedStudentTurn",
    "SimulatedStudentState",
    "SimulationContext",
    "SimulationMode",
    "StudentAgentError",
    "StudentControlProfile",
    "StudentGenerationError",
    "StudentPersona",
    "StudentPolicyScalars",
    "StudentReply",
    "StudentResponseMode",
    "StudentResponseValidator",
    "StudentSimulator",
    "StudentSnapshot",
    "StudentTextGenerator",
    "ValidationResult",
    "load_persona",
    "prepare_student_turn",
]
