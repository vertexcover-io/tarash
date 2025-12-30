"""Video generation module."""

from tarash.tarash_gateway.video.api import generate_video, generate_video_stream
from tarash.tarash_gateway.video.exceptions import (
    ProviderAPIError,
    ValidationError,
    VideoGenerationError,
)
from tarash.tarash_gateway.video.models import (
    AspectRatio,
    Resolution,
    VideoGenerationConfig,
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoGenerationUpdate,
)

__all__ = [
    # API functions
    "generate_video",
    "generate_video_stream",
    # Models
    "VideoGenerationConfig",
    "VideoGenerationRequest",
    "VideoGenerationResponse",
    "VideoGenerationUpdate",
    # Types
    "Resolution",
    "AspectRatio",
    # Exceptions
    "VideoGenerationError",
    "ProviderAPIError",
    "ValidationError",
]
