"""Core data models for video generation."""

from typing import Any, Awaitable, Callable, Literal, Protocol, Union

from pydantic import BaseModel, Field

# ==================== Type Aliases ====================

Resolution = Literal["360p", "480p", "720p", "1080p", "4k"]
AspectRatio = Literal["16:9", "9:16", "1:1", "4:3", "21:9"]

# Progress callback can be sync or async
ProgressCallback = Union[
    Callable[["VideoGenerationUpdate"], None],
    Callable[["VideoGenerationUpdate"], Awaitable[None]],
]

# ==================== Configuration ====================


class VideoGenerationConfig(BaseModel):
    """Configuration for video generation provider."""

    model: str  # e.g., "fal-ai/veo3.1", "openai/sora-2"
    provider: str  # e.g., "fal", "openai", "vertex", "replicate"
    api_key: str
    base_url: str | None = None
    timeout: int = 600  # 10 minutes default
    max_poll_attempts: int = 120
    poll_interval: int = 5  # seconds

    model_config = {"frozen": True}


# ==================== Request ====================


class VideoGenerationRequest(BaseModel):
    """Video generation request with common parameters."""

    prompt: str

    # Common optional parameters (not all providers support all)
    duration: int | str | None = None  # Provider-specific format
    resolution: Resolution | str | None = None
    aspect_ratio: AspectRatio | str | None = None

    # Image/video inputs for I2V, V2V
    image_urls: list[str] = Field(default_factory=list)
    video_url: str | None = None

    # Model-specific parameters
    model_params: dict[str, Any] = Field(default_factory=dict)


# ==================== Response ====================


class VideoGenerationResponse(BaseModel):
    """Normalized video generation response."""

    request_id: str  # Our unique ID for this request

    video_url: str
    content_type: str | None = None
    audio_url: str | None = None
    duration: float | None = None  # seconds
    resolution: str | None = None
    aspect_ratio: str | None = None
    status: Literal["completed", "failed"]

    # Debugging & provider-specific data
    raw_response: dict[str, Any]
    provider_metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = {"frozen": True}


class VideoGenerationUpdate(BaseModel):
    """Progress update during video generation."""

    request_id: str  # Same ID across all updates for this request

    status: Literal["queued", "processing", "completed", "failed"]
    progress_percent: int | None = Field(None, ge=0, le=100)
    update: dict[str, Any]
    result: VideoGenerationResponse | None = None
    error: str | None = None


# ==================== Model-Specific Parameters ====================


class BaseVideoParams(BaseModel):
    """Base class for model-specific parameters."""

    model_config = {"extra": "forbid"}


# Veo (via Fal)
class VeoVideoParams(BaseVideoParams):
    """Veo-specific parameters (beyond common)."""

    negative_prompt: str | None = None
    seed: int | None = None
    generate_audio: bool = True
    auto_fix: bool = True  # Fal-specific
    enhance_prompt: bool = True  # Fal-specific


# Sora
class SoraVideoParams(BaseVideoParams):
    """Sora-specific parameters."""

    remix_video_id: str | None = None


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

    model_config = {"extra": "forbid"}


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

    model_config = {"extra": "forbid"}


# Kling
class KlingVideoParams(BaseVideoParams):
    """Kling-specific parameters."""

    mode: Literal["std", "pro"] = "std"
    sound: bool = False
    negative_prompt: str | None = None
    cfg_scale: float | None = Field(None, ge=0.0, le=1.0)
    camera_control: KlingCameraControl | None = None


# Registry of model parameter schemas
MODEL_PARAMS_SCHEMAS: dict[str, type[BaseVideoParams]] = {
    # Fal Veo models
    "fal-ai/veo3": VeoVideoParams,
    "fal-ai/veo3.1": VeoVideoParams,
    "fal-ai/veo3.1/fast": VeoVideoParams,
    # OpenAI Sora
    "openai/sora-2": SoraVideoParams,
    "openai/sora-2-pro": SoraVideoParams,
    # Kling models
    "kling/2.6": KlingVideoParams,
    "kling/2.6-pro": KlingVideoParams,
    "kling/3.1": KlingVideoParams,
}


# ==================== Provider Handler Protocol ====================


class ProviderHandler(Protocol):
    """Protocol for provider handler implementations."""

    def _get_client(
        self, config: VideoGenerationConfig, client_type: Literal["sync", "async"]
    ) -> Any:
        """
        Get or create provider client.

        Args:
            config: Provider configuration
            client_type: Type of client to return ("sync" or "async")

        Returns:
            Provider-specific client instance (sync or async)
        """
        ...

    def _validate_params(
        self, config: VideoGenerationConfig, request: VideoGenerationRequest
    ) -> dict[str, Any]:
        """
        Validate and transform model_params.

        Args:
            config: Provider configuration
            request: Video generation request

        Returns:
            Validated model parameters dict
        """
        ...

    def _convert_request(
        self, config: VideoGenerationConfig, request: VideoGenerationRequest
    ) -> dict[str, Any]:
        """
        Convert VideoGenerationRequest to provider-specific format.

        Args:
            config: Provider configuration
            request: Video generation request

        Returns:
            Provider-specific request payload
        """
        ...

    def _convert_response(
        self,
        config: VideoGenerationConfig,
        request_id: str,
        provider_response: Any,
    ) -> VideoGenerationResponse:
        """
        Convert provider response to VideoGenerationResponse.

        Args:
            config: Provider configuration
            request_id: Our request ID
            provider_response: Raw provider response

        Returns:
            Normalized VideoGenerationResponse
        """
        ...

    async def generate_video_async(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """
        Generate video asynchronously with progress callback (sync or async).

        Args:
            config: Provider configuration
            request: Video generation request
            on_progress: Optional callback (sync or async) for progress updates

        Returns:
            Final VideoGenerationResponse when complete
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
            on_progress: Optional callback (sync or async) for progress updates

        Returns:
            Final VideoGenerationResponse
        """
        ...
