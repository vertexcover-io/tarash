"""Video generation module."""

from tarash.tarash_gateway.video.api import generate_video, generate_video_async
from tarash.tarash_gateway.video.exceptions import (
    ProviderAPIError,
    ValidationError,
    VideoGenerationError,
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
    "VideoGenerationError",
    "ProviderAPIError",
    "ValidationError",
]
