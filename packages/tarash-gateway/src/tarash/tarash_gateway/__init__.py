"""Tarash Gateway - Unified interface for AI generation models."""

__version__ = "0.1.0"

from .api import (
    generate_video,
    generate_video_async,
    get_provider_field_mapping,
    register_provider,
    register_provider_field_mapping,
)
from .exceptions import (
    ContentModerationError,
    GenerationFailedError,
    HTTPConnectionError,
    HTTPError,
    TarashException,
    TimeoutError,
    ValidationError,
)
from .models import (
    AspectRatio,
    ImageType,
    Resolution,
    VideoGenerationConfig,
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoGenerationUpdate,
)

__all__ = [
    # API functions
    "generate_video",
    "generate_video_async",
    "register_provider",
    "register_provider_field_mapping",
    "get_provider_field_mapping",
    # Models
    "VideoGenerationConfig",
    "VideoGenerationRequest",
    "VideoGenerationResponse",
    "VideoGenerationUpdate",
    # Types
    "Resolution",
    "AspectRatio",
    "ImageType",
    # Exceptions
    "TarashException",
    "ValidationError",
    "ContentModerationError",
    "HTTPError",
    "HTTPConnectionError",
    "TimeoutError",
    "GenerationFailedError",
]
