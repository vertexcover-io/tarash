"""Core data models for video generation."""

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Awaitable, Callable, Literal, Protocol, TypedDict, Union

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, model_validator

# ==================== Type Aliases ====================

Resolution = Literal["360p", "480p", "720p", "1080p", "4k"]
AspectRatio = Literal["16:9", "9:16", "1:1", "4:3", "21:9"]
Base64 = str


class MediaContent(TypedDict):
    """Media content as bytes with content type."""

    content: bytes
    content_type: str


MediaType = Union[Base64, HttpUrl, MediaContent]


class ImageType(TypedDict):
    image: MediaType
    type: Literal["reference", "first_frame", "last_frame", "asset", "style"]


# Progress callback can be sync or async
ProgressCallback = Union[
    Callable[["VideoGenerationUpdate"], None],
    Callable[["VideoGenerationUpdate"], Awaitable[None]],
]

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

    model_config = {"frozen": True}


# ==================== Request ====================


class VideoGenerationRequest(BaseModel):
    """Video generation request with common parameters."""

    prompt: str
    duration_seconds: int | None = None
    resolution: Resolution | None = None
    aspect_ratio: AspectRatio | None = None
    generate_audio: bool | None = None
    image_list: list[ImageType] = Field(default_factory=list)
    video: MediaType | None = None
    seed: int | None = None
    number_of_videos: int = 1
    negative_prompt: str | None = None
    enhance_prompt: bool | None = None

    # Model-specific parameters
    extra_params: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def capture_extra_fields(cls, data: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(data, dict):
            return data

        extra_params = data.pop("extra_params", {})

        # Get all field names defined in the model
        known_fields = set(cls.model_fields.keys())

        # Extract extra fields
        extra = {k: v for k, v in data.items() if k not in known_fields}

        # Remove extra fields from data (so Pydantic doesn't complain)
        for k in extra.keys():
            data.pop(k)

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
    raw_response: dict[str, Any]
    provider_metadata: dict[str, Any] = Field(default_factory=dict)
    execution_metadata: ExecutionMetadata | None = None  # Fallback execution tracking

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


class BaseVideoParams(TypedDict, total=False):
    __pydantic_config__ = ConfigDict(extra="allow")


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


# ==================== Mock Configuration ====================


@dataclass(frozen=True)
class MockVideoSpec:
    """Specification for a mock video in the library."""

    aspect_ratio: AspectRatio
    resolution: Resolution
    duration: float  # seconds
    url: str


# Mock video library - hardcoded sample videos
MOCK_VIDEO_LIBRARY: list[MockVideoSpec] = [
    # 9:16 Portrait
    MockVideoSpec(
        "9:16",
        "720p",
        4.0,
        "https://storage.googleapis.com/tarash-mock-videos/portrait-720p-4s.mp4",
    ),
    MockVideoSpec(
        "9:16",
        "1080p",
        4.0,
        "https://storage.googleapis.com/tarash-mock-videos/portrait-1080p-4s.mp4",
    ),
    MockVideoSpec(
        "9:16",
        "720p",
        8.0,
        "https://storage.googleapis.com/tarash-mock-videos/portrait-720p-8s.mp4",
    ),
    # 16:9 Landscape
    MockVideoSpec(
        "16:9",
        "720p",
        4.0,
        "https://storage.googleapis.com/tarash-mock-videos/landscape-720p-4s.mp4",
    ),
    MockVideoSpec(
        "16:9",
        "1080p",
        4.0,
        "https://storage.googleapis.com/tarash-mock-videos/landscape-1080p-4s.mp4",
    ),
    MockVideoSpec(
        "16:9",
        "4k",
        8.0,
        "https://storage.googleapis.com/tarash-mock-videos/landscape-4k-8s.mp4",
    ),
    # 1:1 Square
    MockVideoSpec(
        "1:1",
        "720p",
        4.0,
        "https://storage.googleapis.com/tarash-mock-videos/square-720p-4s.mp4",
    ),
    MockVideoSpec(
        "1:1",
        "1080p",
        4.0,
        "https://storage.googleapis.com/tarash-mock-videos/square-1080p-4s.mp4",
    ),
    # 4:3 Classic
    MockVideoSpec(
        "4:3",
        "480p",
        4.0,
        "https://storage.googleapis.com/tarash-mock-videos/classic-480p-4s.mp4",
    ),
    # 21:9 Ultrawide
    MockVideoSpec(
        "21:9",
        "1080p",
        4.0,
        "https://storage.googleapis.com/tarash-mock-videos/ultrawide-1080p-4s.mp4",
    ),
]


class MockPollingConfig(BaseModel):
    """Configuration for simulating polling updates."""

    enabled: bool = True
    status_sequence: list[Literal["queued", "processing", "completed", "failed"]] = (
        Field(default_factory=lambda: ["queued", "processing", "completed"])
    )
    delay_between_updates: float = 0.5
    progress_percentages: list[int] | None = None
    custom_updates: list[dict[str, Any]] | None = None

    @model_validator(mode="after")
    def validate_polling_config(self) -> "MockPollingConfig":
        if self.progress_percentages and len(self.progress_percentages) != len(
            self.status_sequence
        ):
            raise ValueError(
                "progress_percentages length must match status_sequence length"
            )
        if self.custom_updates and len(self.custom_updates) != len(
            self.status_sequence
        ):
            raise ValueError("custom_updates length must match status_sequence length")
        return self

    model_config = {"frozen": True}


class MockResponse(BaseModel):
    """A single mock response (success or error) with weight."""

    weight: float = 1.0
    mock_response: VideoGenerationResponse | None = None
    output_video: MediaType | None = None
    output_video_type: Literal["url", "content"] = "url"
    error: Exception | None = None

    @model_validator(mode="after")
    def validate_response(self) -> "MockResponse":
        has_success = self.mock_response is not None or self.output_video is not None
        has_error = self.error is not None

        if has_success and has_error:
            raise ValueError(
                "Cannot specify both success and error in the same MockResponse"
            )
        if self.mock_response and self.output_video:
            raise ValueError("Cannot specify both mock_response and output_video")
        if self.weight <= 0:
            raise ValueError("weight must be positive")
        return self

    model_config = {"frozen": True, "arbitrary_types_allowed": True}


class MockConfig(BaseModel):
    """Configuration for mocking video generation responses."""

    enabled: bool
    responses: list[MockResponse] | None = None
    polling: MockPollingConfig | None = None

    @model_validator(mode="after")
    def validate_config(self) -> "MockConfig":
        if self.enabled and not self.responses:
            self.responses = [MockResponse(weight=1.0)]
        if self.responses:
            total_weight = sum(r.weight for r in self.responses)
            if total_weight <= 0:
                raise ValueError("Total weight must be positive")
        return self

    model_config = {"frozen": True, "arbitrary_types_allowed": True}


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
        request: VideoGenerationRequest,
        request_id: str,
        provider_response: Any,
    ) -> VideoGenerationResponse:
        """
        Convert provider response to VideoGenerationResponse.

        Args:
            config: Provider configuration
            request: Original video generation request
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
