"""Exception hierarchy for the research environment."""

from __future__ import annotations


class ResearchError(Exception):
    """Base class for every error raised by this package."""


class ConfigError(ResearchError):
    """Raised when configuration is missing or invalid."""


class StorageError(ResearchError):
    """Raised when the persistent store cannot satisfy a request."""


class NotFound(StorageError):
    """Raised when a referenced record does not exist."""


class SourceError(ResearchError):
    """Base class for source/provider failures."""

    def __init__(self, message: str, *, provider: str, retryable: bool = False) -> None:
        super().__init__(message)
        self.provider = provider
        self.retryable = retryable


class SourceUnavailable(SourceError):
    """The provider could not be reached or returned a server-side failure."""

    def __init__(self, message: str, *, provider: str, retryable: bool = True) -> None:
        super().__init__(message, provider=provider, retryable=retryable)


class SourceRejected(SourceError):
    """The provider rejected the request (auth, quota, malformed query)."""


class SourceNotConfigured(SourceError):
    """The provider exists but has no credentials/configuration available."""


class UnsafeRequest(ResearchError):
    """Raised when an outbound request violates the acquisition safety policy."""


class BudgetExceeded(ResearchError):
    """Raised when an operation would exceed an investigation's budget."""

    def __init__(self, resource: str, limit: int | float, used: int | float) -> None:
        super().__init__(
            f"budget exhausted for {resource!r}: used {used} of {limit}"
        )
        self.resource = resource
        self.limit = limit
        self.used = used
