"""Exceptions for video generation."""

from typing import Any


class VideoGenerationError(Exception):
    """Base exception for video generation errors."""

    def __init__(
        self,
        message: str,
        provider: str,
        raw_response: dict[str, Any] | None = None,
        request_id: str | None = None,
        model: str | None = None,
    ):
        self.message = message
        self.provider = provider
        self.raw_response = raw_response
        self.request_id = request_id
        self.model = model
        super().__init__(message)


class ProviderAPIError(VideoGenerationError):
    """Provider API returned an error."""

    pass


class ValidationError(VideoGenerationError):
    """Request validation failed."""

    pass
