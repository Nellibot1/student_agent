"""Public exception hierarchy for errors callers may handle by pipeline stage.

Catch ``StudentAgentError`` for one package-level failure path, or catch a specific subclass
when an application wants different retry or reporting behavior for perception, generation,
validation, or persona creation.
"""


class StudentAgentError(Exception):
    """Base class for package-specific runtime errors."""


class StudentGenerationError(StudentAgentError):
    """Raised when the injected text generator fails."""


class StudentPerceptionError(StudentAgentError):
    """Raised after semantic perception retries are exhausted."""


class StudentValidationError(StudentAgentError):
    """Raised when semantic response validation cannot be completed."""


class PersonaGenerationError(StudentAgentError):
    """Raised after natural-language persona generation retries are exhausted."""


class InvalidStudentResponseError(StudentAgentError):
    """Raised when generated text is empty or fails configured validation."""
