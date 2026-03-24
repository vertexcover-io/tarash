"""Tarash Gateway - Unified interface for AI generation models."""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("tarash-gateway")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0"

from .api import (
    generate_image,
    generate_image_async,
    generate_sts,
    generate_sts_async,
    generate_tts,
    generate_tts_async,
    generate_video,
    generate_video_async,
    get_provider_field_mapping,
    health_check,
    health_check_async,
    register_provider,
    register_provider_field_mapping,
)
from .costs import CostEstimate, estimate_cost
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
    AudioGenerationConfig,
    AudioOutputFormat,
    HealthCheckResult,
    ImageGenerationConfig,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageType,
    Resolution,
    STSRequest,
    STSResponse,
    STSUpdate,
    TTSRequest,
    TTSResponse,
    TTSUpdate,
    VideoGenerationConfig,
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoGenerationUpdate,
    format_to_content_type,
)

__all__ = [
    # API functions
    "generate_video",
    "generate_video_async",
    "generate_image",
    "generate_image_async",
    "generate_tts",
    "generate_tts_async",
    "generate_sts",
    "generate_sts_async",
    "register_provider",
    "register_provider_field_mapping",
    "get_provider_field_mapping",
    "health_check",
    "health_check_async",
    # Models
    "VideoGenerationConfig",
    "VideoGenerationRequest",
    "VideoGenerationResponse",
    "VideoGenerationUpdate",
    "ImageGenerationConfig",
    "ImageGenerationRequest",
    "ImageGenerationResponse",
    "AudioGenerationConfig",
    "TTSRequest",
    "TTSResponse",
    "TTSUpdate",
    "STSRequest",
    "STSResponse",
    "STSUpdate",
    "HealthCheckResult",
    # Types
    "Resolution",
    "AspectRatio",
    "ImageType",
    "AudioOutputFormat",
    "format_to_content_type",
    # Cost estimation
    "CostEstimate",
    "estimate_cost",
    # Exceptions
    "TarashException",
    "ValidationError",
    "ContentModerationError",
    "HTTPError",
    "HTTPConnectionError",
    "TimeoutError",
    "GenerationFailedError",
]
