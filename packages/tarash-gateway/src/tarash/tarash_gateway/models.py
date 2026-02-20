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
    """An image with a semantic role used in video/image generation requests."""

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
    """Metadata for a single provider attempt within the fallback chain.

    Captured automatically by the orchestrator for each provider tried.
    Accessible via ``VideoGenerationResponse.execution_metadata.attempts``.
    """

    provider: str
    """Provider identifier (e.g. ``"fal"``, ``"runway"``)."""
    model: str
    """Model name used for this attempt."""
    attempt_number: int
    """1-based index of this attempt in the fallback chain."""
    started_at: datetime
    """UTC timestamp when this attempt began."""
    ended_at: datetime | None
    """UTC timestamp when this attempt completed, or ``None`` if still running."""
    status: Literal["success", "failed", "skipped"]
    """Outcome of this attempt."""
    error_type: str | None
    """Exception class name if the attempt failed, otherwise ``None``."""
    error_message: str | None
    """Human-readable error message if the attempt failed, otherwise ``None``."""
    is_retryable: bool | None
    """Whether the error was classified as retryable (triggers next fallback)."""
    request_id: str | None
    """Provider-assigned request ID if available before failure."""

    @property
    def elapsed_seconds(self) -> float | None:
        """Compute elapsed time in seconds for this attempt."""
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()


@dataclass
class ExecutionMetadata:
    """Metadata for the complete execution across all fallback attempts.

    Attached to every ``VideoGenerationResponse`` and ``ImageGenerationResponse``
    so callers can inspect timing, retry behaviour, and which provider ultimately
    succeeded.
    """

    total_attempts: int
    """Total number of provider attempts made (including failed ones)."""
    successful_attempt: int | None
    """1-based index of the attempt that succeeded, or ``None`` on total failure."""
    attempts: list[AttemptMetadata]
    """Ordered list of per-attempt metadata, one entry per provider tried."""
    fallback_triggered: bool
    """``True`` if at least one fallback was triggered due to a retryable error."""
    configs_in_chain: int
    """Total number of configs in the fallback chain (primary + fallbacks)."""

    @property
    def total_elapsed_seconds(self) -> float:
        """Total wall-clock time in seconds from first attempt start to last end."""
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
    """Configuration for a video generation request.

    Passed to ``generate_video()`` or ``generate_video_async()`` to select the
    provider, model, credentials, and timeout behaviour. Immutable — create a
    copy via ``model_copy(update={...})`` to change fields.

    Example:
        ```python
        config = VideoGenerationConfig(
            provider="fal",
            model="fal-ai/veo3",
            api_key="FAL_KEY",
            timeout=600,
            fallback_configs=[
                VideoGenerationConfig(provider="runway", api_key="RUNWAY_KEY"),
            ],
        )
        ```
    """

    model: str = Field(
        description="Model identifier, e.g. 'fal-ai/veo3', 'openai/sora-2'."
    )
    provider: str = Field(
        description="Provider identifier, e.g. 'fal', 'openai', 'runway'."
    )
    api_key: str | None = Field(
        default=None,
        description="API key for authenticating with the provider. Optional for Google Vertex AI.",
    )
    base_url: str | None = Field(
        default=None, description="Override the provider's base API URL."
    )
    api_version: str | None = Field(
        default=None,
        description="API version string. Required for Azure OpenAI (e.g. '2024-05-01-preview').",
    )
    timeout: int = Field(
        default=600, description="Maximum seconds to wait for generation to complete."
    )
    max_poll_attempts: int = Field(
        default=120, description="Maximum number of status-poll iterations."
    )
    poll_interval: int = Field(
        default=5, description="Seconds to wait between status polls."
    )
    mock: "MockConfig | None" = Field(
        default=None, description="If set, enables mock generation for testing."
    )
    fallback_configs: list["VideoGenerationConfig"] | None = Field(
        default=None,
        description="Ordered list of fallback configs to try on retryable errors.",
    )
    provider_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional provider-specific configuration (e.g. GCP project for Vertex AI).",
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)


# ==================== Request ====================


class VideoGenerationRequest(BaseModel):
    """Parameters for a video generation request.

    All standard fields map to a common interface shared across providers.
    Provider-specific parameters that have no standard equivalent can be passed
    as keyword arguments — they are automatically captured into ``extra_params``
    by the ``capture_extra_fields`` validator.

    Example:
        ```python
        # Standard usage
        request = VideoGenerationRequest(
            prompt="A cat playing piano",
            duration_seconds=4,
            aspect_ratio="16:9",
        )

        # Kling-specific param captured into extra_params automatically
        request = VideoGenerationRequest(
            prompt="A cat playing piano",
            cfg_scale=0.5,
        )
        ```
    """

    prompt: str = Field(description="Text description of the video to generate.")
    duration_seconds: int | None = Field(
        default=None, description="Requested video duration in seconds."
    )
    resolution: Resolution | None = Field(
        default=None, description="Requested resolution, e.g. '1080p', '720p'."
    )
    aspect_ratio: AspectRatio | None = Field(
        default=None, description="Requested aspect ratio, e.g. '16:9', '9:16'."
    )
    generate_audio: bool | None = Field(
        default=None, description="Request audio generation alongside the video."
    )
    image_list: list[ImageType] = Field(
        default_factory=_empty_image_list,
        description="Input images with semantic roles (first_frame, last_frame, reference, etc.).",
    )
    video: MediaType | None = Field(
        default=None, description="Input video for extend or remix workflows."
    )
    seed: int | None = Field(
        default=None, description="Seed for reproducible generation."
    )
    number_of_videos: int = Field(
        default=1, description="Number of video variants to generate."
    )
    negative_prompt: str | None = Field(
        default=None, description="Elements to avoid in the output."
    )
    enhance_prompt: bool | None = Field(
        default=None, description="Allow the provider to enhance the prompt."
    )

    # Model-specific parameters
    extra_params: dict[str, object] = Field(
        default_factory=dict,
        description="Provider- or model-specific parameters with no standard equivalent.",
    )

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
    """Normalized response returned by every video generation call.

    Immutable. All provider-specific data is preserved in ``raw_response``
    and ``provider_metadata`` for debugging.
    """

    request_id: str = Field(description="Tarash-assigned unique ID for this request.")
    video: MediaType = Field(
        description="Generated video as a URL, base64 string, or bytes."
    )
    content_type: str | None = Field(
        default=None, description="MIME type of the video, e.g. 'video/mp4'."
    )
    audio_url: str | None = Field(
        default=None, description="URL to the generated audio track, if any."
    )
    duration: float | None = Field(
        default=None, description="Actual video duration in seconds."
    )
    resolution: str | None = Field(
        default=None, description="Actual resolution of the generated video."
    )
    aspect_ratio: str | None = Field(
        default=None, description="Actual aspect ratio of the generated video."
    )
    status: Literal["completed", "failed"] = Field(
        description="Final generation status."
    )
    is_mock: bool = Field(
        default=False,
        description="True if the response was produced by the mock provider.",
    )
    raw_response: dict[str, object] = Field(
        description="Unmodified provider response, preserved for debugging."
    )
    provider_metadata: dict[str, object] = Field(
        default_factory=dict,
        description="Additional provider-specific fields not covered by the standard interface.",
    )
    execution_metadata: ExecutionMetadata | None = Field(
        default=None,
        description="Timing and fallback attempt details captured by the orchestrator.",
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)


class VideoGenerationUpdate(BaseModel):
    """A progress event emitted during video generation polling.

    Passed to the ``on_progress`` callback on each polling cycle. When
    ``status`` is ``"completed"``, ``result`` will be populated with the
    final response.
    """

    request_id: str = Field(
        description="Same ID as the originating request, for correlation."
    )
    status: StatusType = Field(description="Current generation status.")
    progress_percent: int | None = Field(
        None, ge=0, le=100, description="Estimated completion percentage (0–100)."
    )
    update: dict[str, object] = Field(
        description="Raw event payload from the provider polling cycle."
    )
    result: VideoGenerationResponse | None = Field(
        default=None, description="Final response, set only when status is 'completed'."
    )
    error: str | None = Field(
        default=None, description="Error message if status is 'failed'."
    )


# ==================== Image Generation Models ====================


class ImageGenerationConfig(BaseModel):
    """Configuration for an image generation request.

    Passed to ``generate_image()`` or ``generate_image_async()``. Immutable —
    use ``model_copy(update={...})`` to derive a modified copy.
    """

    model: str = Field(
        description="Model identifier, e.g. 'dall-e-3', 'fal-ai/flux-pro'."
    )
    provider: str = Field(
        description="Provider identifier, e.g. 'openai', 'fal', 'stability'."
    )
    api_key: str | None = Field(
        default=None,
        description="API key for authenticating with the provider.",
    )
    base_url: str | None = Field(
        default=None, description="Override the provider's base API URL."
    )
    api_version: str | None = Field(
        default=None,
        description="API version string. Required for Azure OpenAI.",
    )
    timeout: int = Field(
        default=120,
        description="Maximum seconds to wait for generation (default 2 min).",
    )
    max_poll_attempts: int = Field(
        default=60, description="Maximum number of status-poll iterations."
    )
    poll_interval: int = Field(
        default=2, description="Seconds to wait between status polls."
    )
    mock: "MockConfig | None" = Field(
        default=None, description="If set, enables mock generation for testing."
    )
    fallback_configs: list["ImageGenerationConfig"] | None = Field(
        default=None,
        description="Ordered list of fallback configs to try on retryable errors.",
    )
    provider_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional provider-specific configuration.",
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)


class ImageGenerationRequest(BaseModel):
    """Parameters for an image generation request.

    Unknown keyword arguments are automatically captured into ``extra_params``
    by the ``capture_extra_fields`` validator, allowing provider-specific
    parameters to be passed without modifying the standard interface.
    """

    prompt: str = Field(description="Text description of the image to generate.")
    negative_prompt: str | None = Field(
        default=None, description="Elements to avoid in the output."
    )
    size: str | None = Field(
        default=None, description="Output size as 'WxH', e.g. '1024x1024', '1792x1024'."
    )
    quality: str | None = Field(
        default=None, description="Quality level, e.g. 'standard' or 'hd'."
    )
    style: str | None = Field(
        default=None, description="Style mode, e.g. 'vivid' or 'natural' (OpenAI)."
    )
    n: int | None = Field(
        default=None, description="Number of images to generate in one request."
    )
    image_list: list[ImageType] = Field(
        default_factory=_empty_image_list,
        description="Input images for img2img or inpainting workflows.",
    )
    mask_image: MediaType | None = Field(
        default=None, description="Mask image for inpainting (white = edit area)."
    )
    seed: int | None = Field(
        default=None, description="Seed for reproducible generation."
    )
    aspect_ratio: AspectRatio | None = Field(
        default=None, description="Aspect ratio as an alternative to explicit size."
    )

    # Model-specific parameters
    extra_params: dict[str, object] = Field(
        default_factory=dict,
        description="Provider- or model-specific parameters with no standard equivalent.",
    )

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
    """Normalized response returned by every image generation call.

    Immutable. All provider-specific data is preserved in ``raw_response``
    and ``provider_metadata`` for debugging.
    """

    request_id: str = Field(description="Tarash-assigned unique ID for this request.")
    images: list[str] = Field(
        description="Generated images as a list of URLs or base64-encoded strings."
    )
    content_type: str | None = Field(
        default="image/png", description="MIME type of the generated images."
    )
    status: Literal["completed", "failed"] = Field(
        description="Final generation status."
    )
    is_mock: bool = Field(
        default=False,
        description="True if the response was produced by the mock provider.",
    )
    revised_prompt: str | None = Field(
        default=None,
        description="Prompt as revised by the provider (e.g. OpenAI may modify for safety).",
    )
    raw_response: dict[str, object] = Field(
        description="Unmodified provider response, preserved for debugging."
    )
    provider_metadata: dict[str, object] = Field(
        default_factory=dict,
        description="Additional provider-specific fields not covered by the standard interface.",
    )
    execution_metadata: ExecutionMetadata | None = Field(
        default=None,
        description="Timing and fallback attempt details captured by the orchestrator.",
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)


class ImageGenerationUpdate(BaseModel):
    """A progress event emitted during image generation polling.

    Passed to the ``on_progress`` callback on each polling cycle. When
    ``status`` is ``"completed"``, ``result`` will be populated with the
    final response.
    """

    request_id: str = Field(
        description="Same ID as the originating request, for correlation."
    )
    status: StatusType = Field(description="Current generation status.")
    progress_percent: int | None = Field(
        None, ge=0, le=100, description="Estimated completion percentage (0–100)."
    )
    update: dict[str, object] = Field(
        description="Raw event payload from the provider polling cycle."
    )
    result: ImageGenerationResponse | None = Field(
        default=None, description="Final response, set only when status is 'completed'."
    )
    error: str | None = Field(
        default=None, description="Error message if status is 'failed'."
    )


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
    """Interface that all provider implementations must satisfy.

    Providers handle both video and image generation where supported.
    Methods for unsupported modalities should raise ``NotImplementedError``.
    Register a custom implementation at runtime with ``register_provider()``.
    """

    # ==================== Video Generation ====================

    async def generate_video_async(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """Generate a video asynchronously.

        Args:
            config: Provider configuration with API key, model, and timeouts.
            request: Video generation parameters.
            on_progress: Optional callback invoked on each polling cycle.

        Returns:
            ``VideoGenerationResponse`` with video URL and metadata.

        Raises:
            NotImplementedError: If this provider does not support video generation.
            TarashException: On any provider-level error.
        """
        ...

    def generate_video(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """Generate a video synchronously (blocking).

        Args:
            config: Provider configuration with API key, model, and timeouts.
            request: Video generation parameters.
            on_progress: Optional callback invoked on each polling cycle.

        Returns:
            ``VideoGenerationResponse`` with video URL and metadata.

        Raises:
            NotImplementedError: If this provider does not support video generation.
            TarashException: On any provider-level error.
        """
        ...

    # ==================== Image Generation ====================

    async def generate_image_async(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        on_progress: ImageProgressCallback | None = None,
    ) -> ImageGenerationResponse:
        """Generate an image asynchronously.

        Args:
            config: Provider configuration with API key, model, and timeouts.
            request: Image generation parameters.
            on_progress: Optional callback invoked during generation.

        Returns:
            ``ImageGenerationResponse`` with generated images and metadata.

        Raises:
            NotImplementedError: If this provider does not support image generation.
            TarashException: On any provider-level error.
        """
        ...

    def generate_image(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        on_progress: ImageProgressCallback | None = None,
    ) -> ImageGenerationResponse:
        """Generate an image synchronously (blocking).

        Args:
            config: Provider configuration with API key, model, and timeouts.
            request: Image generation parameters.
            on_progress: Optional callback invoked during generation.

        Returns:
            ``ImageGenerationResponse`` with generated images and metadata.

        Raises:
            NotImplementedError: If this provider does not support image generation.
            TarashException: On any provider-level error.
        """
        ...
