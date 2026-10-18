"""Provider- and application-independent simulated student public API.

Import stable, provider-neutral types from this module. Optional convenience adapters live
under ``student_agent.integrations`` so importing the core package does not require a model
SDK.
"""

from .controls import StudentControlProfile, StudentPolicyScalars
from .conversation import ConversationMessage, ConversationRole
from .engine import (
    CallableTextGenerator,
    PerceptionFailurePolicy,
    PreparedStudentTurn,
    StudentReply,
    StudentResponseValidator,
    StudentSimulator,
    StudentSnapshot,
    StudentTextGenerator,
    ValidationFailurePolicy,
    ValidationPolicy,
    ValidationResult,
    ValidationStatus,
    prepare_student_turn,
    response_requires_validation,
)
from .errors import (
    InvalidStudentResponseError,
    PersonaGenerationError,
    StudentAgentError,
    StudentGenerationError,
    StudentPerceptionError,
    StudentValidationError,
)
from .evaluation import (
    ExpectedObservation,
    PerceptionEvaluationCase,
    PerceptionEvaluationReport,
    evaluate_perception,
    load_evaluation_cases,
    save_evaluation_report,
)
from .perception import (
    MessageType,
    PerceivedFeedback,
    StudentObservation,
    StudentPerception,
)
from .persona_generation import PersonaGenerator, generate_persona_from_description
from .personas import (
    DEFAULT_PERSONA,
    LearningDynamics,
    StudentPersona,
    available_builtin_personas,
    load_builtin_persona,
    load_persona,
    save_persona,
)
from .policy import (
    KnowledgeComponentState,
    MisconceptionState,
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
    "ExpectedObservation",
    "InvalidStudentResponseError",
    "KnowledgeComponentState",
    "LearningDynamics",
    "MessageType",
    "MisconceptionState",
    "PerceivedFeedback",
    "PerceptionEvaluationCase",
    "PerceptionEvaluationReport",
    "PerceptionFailurePolicy",
    "PersonaGenerationError",
    "PersonaGenerator",
    "PolicyDecision",
    "PreparedStudentTurn",
    "SimulatedStudentState",
    "SimulationMode",
    "StudentAgentError",
    "StudentControlProfile",
    "StudentGenerationError",
    "StudentObservation",
    "StudentPerception",
    "StudentPerceptionError",
    "StudentPersona",
    "StudentPolicyScalars",
    "StudentReply",
    "StudentResponseMode",
    "StudentResponseValidator",
    "StudentSimulator",
    "StudentSnapshot",
    "StudentTextGenerator",
    "StudentValidationError",
    "ValidationFailurePolicy",
    "ValidationPolicy",
    "ValidationResult",
    "ValidationStatus",
    "available_builtin_personas",
    "evaluate_perception",
    "generate_persona_from_description",
    "load_builtin_persona",
    "load_evaluation_cases",
    "load_persona",
    "prepare_student_turn",
    "response_requires_validation",
    "save_evaluation_report",
    "save_persona",
]
