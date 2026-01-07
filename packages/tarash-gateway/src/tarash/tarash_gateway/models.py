"""Core data models for video and image generation."""

from dataclasses import dataclass
from datetime import datetime
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    ClassVar,
    Literal,
    Protocol,
    TypeAlias,
    TypeVar,
    TypedDict,
    cast,
)
from collections.abc import Awaitable

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

if TYPE_CHECKING:
    from tarash.tarash_gateway.mock import MockConfig

# ==================== Type Aliases ====================
AnyDict: TypeAlias = dict[str, Any]  # pyright: ignore[reportExplicitAny]

Resolution = Literal["360p", "480p", "720p", "1080p", "4k"]
AspectRatio = Literal["16:9", "9:16", "1:1", "4:3", "21:9"]
Base64 = str
StatusType = Literal["queued", "processing", "completed", "failed"]


class MediaContent(TypedDict):
    """Media content as bytes with content type."""

    content: bytes
    content_type: str


MediaType = Base64 | HttpUrl | MediaContent


class ImageType(TypedDict):
    image: MediaType
    type: Literal["reference", "first_frame", "last_frame", "asset", "style"]


def _empty_image_list() -> list[ImageType]:
    """Factory function for empty image list with proper type."""
    return []


# Progress callback types (video)
SyncProgressCallback = Callable[["VideoGenerationUpdate"], None]
AsyncProgressCallback = Callable[["VideoGenerationUpdate"], Awaitable[None]]
ProgressCallback = SyncProgressCallback | AsyncProgressCallback

# Progress callback types (image)
SyncImageProgressCallback = Callable[["ImageGenerationUpdate"], None]
AsyncImageProgressCallback = Callable[["ImageGenerationUpdate"], Awaitable[None]]
ImageProgressCallback = SyncImageProgressCallback | AsyncImageProgressCallback

# ==================== Execution Metadata ====================


@dataclass
class AttemptMetadata:
    """Metadata for a single attempt in the fallback chain."""

    provider: str
    model: str
    attempt_number: int
    started_at: datetime
    ended_at: datetime | None
    status: Literal["success", "failed", "skipped"]
    error_type: str | None
    error_message: str | None
    is_retryable: bool | None
    request_id: str | None

    @property
    def elapsed_seconds(self) -> float | None:
        """Compute elapsed time from timestamps."""
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()


@dataclass
class ExecutionMetadata:
    """Metadata for the complete execution including all fallback attempts."""

    total_attempts: int
    successful_attempt: int | None
    attempts: list[AttemptMetadata]
    fallback_triggered: bool
    configs_in_chain: int

    @property
    def total_elapsed_seconds(self) -> float:
        """Compute total elapsed time from first start to last end."""
        if not self.attempts:
            return 0.0

        first_start = self.attempts[0].started_at
        last_end = max(
            (
                attempt.ended_at
                for attempt in self.attempts
                if attempt.ended_at is not None
            ),
            default=first_start,
        )

        return (last_end - first_start).total_seconds()


# ==================== Configuration ====================


class VideoGenerationConfig(BaseModel):
    """Configuration for video generation provider."""

    model: str  # e.g., "fal-ai/veo3.1", "openai/sora-2"
    provider: str  # e.g., "fal", "openai", "vertex", "replicate"
    api_key: str
    base_url: str | None = None
    api_version: str | None = None  # For Azure OpenAI (e.g., "2024-05-01-preview")
    timeout: int = 600  # 10 minutes default
    max_poll_attempts: int = 120
    poll_interval: int = 5  # seconds
    mock: "MockConfig | None" = None  # Mock configuration
    fallback_configs: list["VideoGenerationConfig"] | None = None  # Fallback chain

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)


# ==================== Request ====================


class VideoGenerationRequest(BaseModel):
    """Video generation request with common parameters."""

    prompt: str
    duration_seconds: int | None = None
    resolution: Resolution | None = None
    aspect_ratio: AspectRatio | None = None
    generate_audio: bool | None = None
    image_list: list[ImageType] = Field(default_factory=_empty_image_list)
    video: MediaType | None = None
    seed: int | None = None
    number_of_videos: int = 1
    negative_prompt: str | None = None
    enhance_prompt: bool | None = None

    # Model-specific parameters
    extra_params: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def capture_extra_fields(cls, data: dict[str, object]) -> dict[str, object]:
        extra_params: dict[str, object] = cast(
            dict[str, object], data.pop("extra_params", {})
        )

        # Get all field names defined in the model
        known_fields = set(cls.model_fields.keys())

        # Extract extra fields
        extra = {k: v for k, v in data.items() if k not in known_fields}

        # Remove extra fields from data (so Pydantic doesn't complain)
        for k in extra.keys():
            _ = data.pop(k)

        extra_params.update(extra)

        # Store in extra_params
        data["extra_params"] = extra_params

        return data


# ==================== Response ====================


class VideoGenerationResponse(BaseModel):
    """Normalized video generation response."""

    request_id: str  # Our unique ID for this request

    video: MediaType
    content_type: str | None = None
    audio_url: str | None = None
    duration: float | None = None  # seconds
    resolution: str | None = None
    aspect_ratio: str | None = None
    status: Literal["completed", "failed"]
    is_mock: bool = False  # Indicates if this is a mock response

    # Debugging & provider-specific data
    raw_response: dict[str, object]
    provider_metadata: dict[str, object] = Field(default_factory=dict)
    execution_metadata: ExecutionMetadata | None = None  # Fallback execution tracking

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)


class VideoGenerationUpdate(BaseModel):
    """Progress update during video generation."""

    request_id: str  # Same ID across all updates for this request

    status: StatusType
    progress_percent: int | None = Field(None, ge=0, le=100)
    update: dict[str, object]
    result: VideoGenerationResponse | None = None
    error: str | None = None


# ==================== Image Generation Models ====================


class ImageGenerationConfig(BaseModel):
    """Configuration for image generation provider."""

    model: str  # e.g., "fal-ai/flux-pro", "dall-e-3"
    provider: str  # e.g., "fal", "openai"
    api_key: str
    base_url: str | None = None
    api_version: str | None = None  # For Azure OpenAI
    timeout: int = 120  # 2 minutes default (images are faster)
    max_poll_attempts: int = 60
    poll_interval: int = 2  # seconds
    mock: "MockConfig | None" = None
    fallback_configs: list["ImageGenerationConfig"] | None = None

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)


class ImageGenerationRequest(BaseModel):
    """Image generation request with common parameters."""

    prompt: str
    negative_prompt: str | None = None
    size: str | None = None  # "1024x1024", "1024x1792", "1792x1024", etc.
    quality: str | None = None  # "standard", "hd"
    style: str | None = None  # "natural", "vivid"
    n: int | None = None  # Number of images to generate
    image_list: list[ImageType] = Field(
        default_factory=_empty_image_list
    )  # For img2img
    mask_image: MediaType | None = None  # For inpainting
    seed: int | None = None
    aspect_ratio: AspectRatio | None = None  # Alternative to size

    # Model-specific parameters
    extra_params: dict[str, object] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def capture_extra_fields(cls, data: dict[str, object]) -> dict[str, object]:
        extra_params: dict[str, object] = cast(
            dict[str, object], data.pop("extra_params", {})
        )

        # Get all field names defined in the model
        known_fields = set(cls.model_fields.keys())

        # Extract extra fields
        extra = {k: v for k, v in data.items() if k not in known_fields}

        # Remove extra fields from data (so Pydantic doesn't complain)
        for k in extra.keys():
            _ = data.pop(k)

        extra_params.update(extra)

        # Store in extra_params
        data["extra_params"] = extra_params

        return data


class ImageGenerationResponse(BaseModel):
    """Normalized image generation response."""

    request_id: str  # Our unique ID for this request

    images: list[str]  # List of URLs or base64 data
    content_type: str | None = "image/png"
    status: Literal["completed", "failed"]
    is_mock: bool = False
    revised_prompt: str | None = None  # Some providers may revise the prompt

    # Debugging & provider-specific data
    raw_response: dict[str, object]
    provider_metadata: dict[str, object] = Field(default_factory=dict)
    execution_metadata: ExecutionMetadata | None = None

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)


class ImageGenerationUpdate(BaseModel):
    """Progress update during image generation."""

    request_id: str

    status: StatusType
    progress_percent: int | None = Field(None, ge=0, le=100)
    update: dict[str, object]
    result: ImageGenerationResponse | None = None
    error: str | None = None


# ==================== Model-Specific Parameters ====================


class BaseVideoParams(TypedDict, total=False):
    """Base video parameters - extensible dict for provider-specific params."""

    pass


# Kling Camera Control
class KlingCameraConfig(BaseModel):
    """Camera movement configuration for Kling 'simple' type.

    Choose ONE parameter to be non-zero, all others must be zero.
    All values range from -10 to 10.
    """

    horizontal: float | None = Field(
        None,
        ge=-10.0,
        le=10.0,
        description="Camera horizontal movement (x-axis). Negative=left, positive=right",
    )
    vertical: float | None = Field(
        None,
        ge=-10.0,
        le=10.0,
        description="Camera vertical movement (y-axis). Negative=down, positive=up",
    )
    pan: float | None = Field(
        None,
        ge=-10.0,
        le=10.0,
        description="Camera rotation in vertical plane (x-axis rotation). Negative=down, positive=up",
    )
    tilt: float | None = Field(
        None,
        ge=-10.0,
        le=10.0,
        description="Camera rotation in horizontal plane (y-axis rotation). Negative=left, positive=right",
    )
    roll: float | None = Field(
        None,
        ge=-10.0,
        le=10.0,
        description="Camera roll (z-axis rotation). Negative=counterclockwise, positive=clockwise",
    )
    zoom: float | None = Field(
        None,
        ge=-10.0,
        le=10.0,
        description="Camera focal length change. Negative=zoom out (wider), positive=zoom in (narrower)",
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


class KlingCameraControl(BaseModel):
    """Camera control for Kling video generation.

    Predefined camera movement types:
    - simple: Custom movement using config (one parameter from config must be set)
    - down_back: Camera descends and moves backward (pan down + zoom out). Config must be None.
    - forward_up: Camera moves forward and tilts up (zoom in + pan up). Config must be None.
    - right_turn_forward: Rotate right and move forward. Config must be None.
    - left_turn_forward: Rotate left and move forward. Config must be None.
    """

    type: Literal[
        "simple", "down_back", "forward_up", "right_turn_forward", "left_turn_forward"
    ]
    config: KlingCameraConfig | None = Field(
        None, description="Required for 'simple' type, must be None for other types"
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(extra="forbid")


# Kling
class KlingVideoParams(BaseVideoParams):
    """Kling-specific parameters."""

    mode: Literal["std", "pro"]
    sound: bool
    negative_prompt: str | None
    cfg_scale: float | None
    camera_control: KlingCameraControl | None


# ==================== Provider Handler Protocol ====================

ClientT = TypeVar("ClientT", covariant=True)
RequestT = TypeVar("RequestT", covariant=True)
ProviderResponseT = TypeVar("ProviderResponseT", contravariant=True)


class ProviderHandler(Protocol):
    """Protocol for provider handler implementations.

    Providers implement video and/or image generation methods.
    Not all providers support both - those that don't will raise NotImplementedError.
    """

    # ==================== Video Generation ====================

    async def generate_video_async(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """
        Generate video asynchronously with progress callback.

        Args:
            config: Provider configuration
            request: Video generation request
            on_progress: Optional callback for progress updates

        Returns:
            Final VideoGenerationResponse when complete

        Raises:
            NotImplementedError: If provider doesn't support video generation
        """
        ...

    def generate_video(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """
        Generate video synchronously (blocking) with progress callback.

        Args:
            config: Provider configuration
            request: Video generation request
            on_progress: Optional callback for progress updates

        Returns:
            Final VideoGenerationResponse

        Raises:
            NotImplementedError: If provider doesn't support video generation
        """
        ...

    # ==================== Image Generation ====================

    async def generate_image_async(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        on_progress: ImageProgressCallback | None = None,
    ) -> ImageGenerationResponse:
        """
        Generate image asynchronously with progress callback.

        Args:
            config: Provider configuration
            request: Image generation request
            on_progress: Optional callback for progress updates

        Returns:
            Final ImageGenerationResponse when complete

        Raises:
            NotImplementedError: If provider doesn't support image generation
        """
        ...

    def generate_image(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        on_progress: ImageProgressCallback | None = None,
    ) -> ImageGenerationResponse:
        """
        Generate image synchronously (blocking) with progress callback.

        Args:
            config: Provider configuration
            request: Image generation request
            on_progress: Optional callback for progress updates

        Returns:
            Final ImageGenerationResponse

        Raises:
            NotImplementedError: If provider doesn't support image generation
        """
        ...
