class AgentOutputError(Exception):
    """Raised when the LLM output cannot be validated into the required schema."""


class KieServiceError(Exception):
    """Raised when Kie API integration fails."""


class KieTimeoutError(KieServiceError):
    """Raised when Kie polling exceeds timeout."""


class KieTaskFailedError(KieServiceError):
    """Raised when Kie marks a task as failed."""
