"""Public exception hierarchy for the student agent package."""


class StudentAgentError(Exception):
    """Base class for package-specific runtime errors."""


class StudentGenerationError(StudentAgentError):
    """Raised when the injected text generator fails."""


class InvalidStudentResponseError(StudentAgentError):
    """Raised when generated text is empty or fails configured validation."""
