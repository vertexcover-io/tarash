"""Video generation module."""

from tarash.tarash_gateway.video.api import (
    generate_video,
    generate_video_async,
    get_provider_field_mapping,
    register_provider,
    register_provider_field_mapping,
)
from tarash.tarash_gateway.video.exceptions import (
    ContentModerationError,
    GenerationFailedError,
    HTTPError,
    TarashException,
    ValidationError,
)
from tarash.tarash_gateway.video.models import (
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
    "VideoType",
    # Exceptions
    "TarashException",
    "ValidationError",
    "ContentModerationError",
    "HTTPError",
    "GenerationFailedError",
]
