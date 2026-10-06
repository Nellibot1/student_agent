"""Provider- and application-independent simulated student public API.

Framework- and model-provider-specific adapters live outside this package.
"""

from .controls import StudentControlProfile, StudentPolicyScalars
from .conversation import ConversationMessage, ConversationRole
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
from .perception import (
    MessageType,
    PerceivedFeedback,
    RuleBasedPerception,
    StudentObservation,
    StudentPerception,
)
from .personas import (
    DEFAULT_PERSONA,
    StudentPersona,
    available_builtin_personas,
    load_builtin_persona,
    load_persona,
)
from .policy import (
    PolicyDecision,
    SimulatedStudentState,
    SimulationMode,
    StudentResponseMode,
)

__all__ = [
    "DEFAULT_PERSONA",
    "CallableTextGenerator",
    "ConversationMessage",
    "ConversationRole",
    "InvalidStudentResponseError",
    "MessageType",
    "PerceivedFeedback",
    "PolicyDecision",
    "PreparedStudentTurn",
    "RuleBasedPerception",
    "SimulatedStudentState",
    "SimulationMode",
    "StudentAgentError",
    "StudentControlProfile",
    "StudentGenerationError",
    "StudentObservation",
    "StudentPerception",
    "StudentPersona",
    "StudentPolicyScalars",
    "StudentReply",
    "StudentResponseMode",
    "StudentResponseValidator",
    "StudentSimulator",
    "StudentSnapshot",
    "StudentTextGenerator",
    "ValidationResult",
    "available_builtin_personas",
    "load_builtin_persona",
    "load_persona",
    "prepare_student_turn",
]
