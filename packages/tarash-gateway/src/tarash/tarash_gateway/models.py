"""Core data models for video, image, and audio generation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
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


class AudioOutputFormat(BaseModel):
    """Structured audio output format."""

    format: str = Field(
        description="Audio format/codec, e.g. 'mp3', 'wav', 'pcm', 'flac', 'opus'."
    )
    sample_rate: int | None = Field(
        default=None, description="Sample rate in Hz, e.g. 44100."
    )
    bitrate: int | None = Field(default=None, description="Bitrate in kbps, e.g. 128.")


def format_to_content_type(format: str) -> str:
    """Map audio format to MIME content type."""
    return {
        "mp3": "audio/mpeg",
        "wav": "audio/wav",
        "pcm": "audio/pcm",
        "flac": "audio/flac",
        "opus": "audio/opus",
        "ulaw": "audio/basic",
        "mulaw": "audio/basic",
        "alaw": "audio/basic",
        "aac": "audio/aac",
        "linear16": "audio/pcm",
    }.get(format, "audio/mpeg")


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

# Progress callback types (TTS)
SyncTTSProgressCallback = Callable[["TTSUpdate"], None]
AsyncTTSProgressCallback = Callable[["TTSUpdate"], Awaitable[None]]
TTSProgressCallback = SyncTTSProgressCallback | AsyncTTSProgressCallback

# Progress callback types (STS)
SyncSTSProgressCallback = Callable[["STSUpdate"], None]
AsyncSTSProgressCallback = Callable[["STSUpdate"], Awaitable[None]]
STSProgressCallback = SyncSTSProgressCallback | AsyncSTSProgressCallback

# ==================== Cost ====================


def _safe_int(val: Any) -> int:
    """Safely extract an integer from a value, returning 0 for non-numeric types."""
    if isinstance(val, (int, float)):
        return int(val)
    return 0


@dataclass(frozen=True)
class TokenUsage:
    """Token breakdown for multi-rate token-based pricing."""

    text_input_tokens: int = 0
    image_input_tokens: int = 0
    cached_tokens: int = 0
    image_output_tokens: int = 0
    text_output_tokens: int = 0


@dataclass(frozen=True)
class CostComponent:
    """A single cost component within a compound generation."""

    amount_usd: Decimal | None
    """Estimated cost in USD for this component, or ``None`` if unknown."""
    raw_amount: float
    """Raw quantity used for cost calculation."""
    raw_unit: str
    """Unit of the raw quantity (e.g. ``"tokens"``, ``"images"``)."""


@dataclass(frozen=True)
class GenerationCost:
    """Cost information for a single generation request.

    Attached to provider responses and attempt metadata to track
    per-request cost estimates.
    """

    amount_usd: Decimal | None
    """Estimated cost in USD, or ``None`` if unknown."""
    raw_amount: float
    """Raw quantity used for cost calculation (e.g. seconds, characters)."""
    raw_unit: str
    """Unit of the raw quantity (e.g. ``"seconds"``, ``"characters"``, ``"images"``)."""
    token_usage: TokenUsage | None = None
    """Token breakdown when cost is computed from per-token rates."""
    breakdown: tuple[CostComponent, ...] = ()
    """Per-component cost breakdown. Empty for single-modality responses."""

    @classmethod
    def from_pricing_table(
        cls,
        provider: str,
        model: str,
        quantity: float,
    ) -> GenerationCost | None:
        """Look up a (provider, model) pair in the pricing table and compute cost.

        Args:
            provider: Provider identifier (e.g. ``"fal"``).
            model: Model name (e.g. ``"fal-ai/veo3"``).
            quantity: The quantity to multiply by the per-unit price.

        Returns:
            A ``GenerationCost`` with computed ``amount_usd``, or ``None`` if the
            pair is not found in the table.
        """
        from tarash.tarash_gateway.pricing import PRICING_TABLE

        entry = PRICING_TABLE.get((provider, model))
        if entry is None:
            return None
        return cls(
            amount_usd=Decimal(str(entry.usd_per_unit)) * Decimal(str(quantity)),
            raw_amount=quantity,
            raw_unit=entry.unit,
        )

    @classmethod
    def from_token_usage(
        cls,
        model: str,
        usage: Any,
    ) -> GenerationCost | None:
        """Compute cost from OpenAI image API usage token breakdown.

        Uses separate per-token rates for text input, image input, cached input,
        and output tokens. Returns ``None`` if the model has no known rates or
        usage data is not available.

        Args:
            model: Model name (e.g., ``"gpt-image-1"``, ``"gpt-image-1.5"``).
            usage: OpenAI usage object with ``input_tokens``, ``output_tokens``,
                   ``input_tokens_details``, and ``output_tokens_details``.

        Returns:
            A ``GenerationCost`` with exact token-based cost, or ``None``.
        """
        from tarash.tarash_gateway.pricing import OPENAI_IMAGE_TOKEN_RATES

        rates = OPENAI_IMAGE_TOKEN_RATES.get(model)
        if rates is None or usage is None:
            return None

        # Validate that usage has real numeric data (not a MagicMock)
        total_tokens = _safe_int(getattr(usage, "total_tokens", 0))
        if total_tokens == 0:
            return None

        total_cost = Decimal("0")

        # --- Input token breakdown ---
        input_details = getattr(usage, "input_tokens_details", None)
        has_input_details = input_details is not None and isinstance(
            getattr(input_details, "text_tokens", None), (int, float)
        )

        text_input = 0
        image_input = 0
        cached_tokens = 0

        if has_input_details:
            text_input = _safe_int(getattr(input_details, "text_tokens", 0))
            image_input = _safe_int(getattr(input_details, "image_tokens", 0))
            cached_tokens = _safe_int(getattr(input_details, "cached_tokens", 0))

            # Cached tokens reduce cost — distribute proportionally
            uncached_text = max(0, text_input - cached_tokens)
            cached_text = min(text_input, cached_tokens)
            remaining_cached = max(0, cached_tokens - cached_text)
            uncached_image = max(0, image_input - remaining_cached)
            cached_image = min(image_input, remaining_cached)

            total_cost += Decimal(str(uncached_text)) * Decimal(
                str(rates["text_input"])
            )
            total_cost += Decimal(str(cached_text)) * Decimal(
                str(rates.get("cached_text_input", rates["text_input"]))
            )
            total_cost += Decimal(str(uncached_image)) * Decimal(
                str(rates["image_input"])
            )
            total_cost += Decimal(str(cached_image)) * Decimal(
                str(rates.get("cached_image_input", rates["image_input"]))
            )
        else:
            # No detailed breakdown — use image_input rate for all input tokens
            input_tokens = _safe_int(getattr(usage, "input_tokens", 0))
            total_cost += Decimal(str(input_tokens)) * Decimal(
                str(rates["image_input"])
            )

        # --- Output tokens ---
        output_details = getattr(usage, "output_tokens_details", None)
        has_output_details = output_details is not None and isinstance(
            getattr(output_details, "image_tokens", None), (int, float)
        )

        image_output = 0
        text_output = 0

        if has_output_details:
            image_output = _safe_int(getattr(output_details, "image_tokens", 0))
            text_output = _safe_int(getattr(output_details, "text_tokens", 0))
            total_cost += Decimal(str(image_output)) * Decimal(
                str(rates["image_output"])
            )
            total_cost += Decimal(str(text_output)) * Decimal(
                str(rates.get("text_output", rates["image_output"]))
            )
        else:
            output_tokens = _safe_int(getattr(usage, "output_tokens", 0))
            image_output = output_tokens
            total_cost += Decimal(str(output_tokens)) * Decimal(
                str(rates["image_output"])
            )

        return cls(
            amount_usd=total_cost,
            raw_amount=float(total_tokens),
            raw_unit="tokens",
            token_usage=TokenUsage(
                text_input_tokens=text_input,
                image_input_tokens=image_input,
                cached_tokens=cached_tokens,
                image_output_tokens=image_output,
                text_output_tokens=text_output,
            ),
        )

    @classmethod
    def from_credits(
        cls,
        quantity: float,
        credits_per_unit: float,
        credit_to_usd: float,
        raw_unit: str = "credits",
    ) -> GenerationCost:
        """Compute cost from a credits-based billing model.

        Args:
            quantity: Number of units consumed (e.g. seconds, images).
            credits_per_unit: Credits charged per unit.
            credit_to_usd: USD value of one credit.
            raw_unit: Unit label for ``raw_unit`` field.

        Returns:
            A ``GenerationCost`` with the computed USD amount.
        """
        amount = (
            Decimal(str(quantity))
            * Decimal(str(credits_per_unit))
            * Decimal(str(credit_to_usd))
        )
        return cls(
            amount_usd=amount,
            raw_amount=quantity,
            raw_unit=raw_unit,
        )


# ==================== Execution Metadata ====================


@dataclass
class AttemptMetadata:
    """Metadata for a single provider attempt within the fallback chain.

    Captured automatically by the orchestrator for each provider tried.
    Accessible via [VideoGenerationResponse][] ``execution_metadata.attempts``.
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
    cost: GenerationCost | None = None
    """Cost information for this attempt, populated from the provider response."""

    @property
    def elapsed_seconds(self) -> float | None:
        """Compute elapsed time in seconds for this attempt."""
        if self.ended_at is None:
            return None
        return (self.ended_at - self.started_at).total_seconds()


@dataclass
class ExecutionMetadata:
    """Metadata for the complete execution across all fallback attempts.

    Attached to every [VideoGenerationResponse][] and [ImageGenerationResponse][]
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
    total_cost_usd: Decimal | None = None
    """Sum of all attempt costs in USD, or ``None`` if any attempt lacks cost data."""

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
    """Configuration for a video generation request."""

    model: str = Field(
        description="Model identifier, e.g. 'fal-ai/veo3', 'openai/sora-2'."
    )
    provider: str = Field(
        description="Provider identifier: 'fal', 'openai', 'azure-openai', 'google', 'runway', 'replicate', 'stability'."
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
    """Parameters for a video generation request."""

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
    """Normalized response returned by every video generation call."""

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
    cost: GenerationCost | None = Field(
        default=None,
        description="Cost information for this generation request.",
    )
    raw_response: dict[str, object] = Field(
        description="Unmodified provider response, preserved for debugging."
    )
    execution_metadata: ExecutionMetadata | None = Field(
        default=None,
        description="Timing and fallback attempt details captured by the orchestrator.",
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)


class VideoGenerationUpdate(BaseModel):
    """A progress event emitted during video generation polling."""

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
    """Configuration for an image generation request."""

    model: str = Field(
        description="Model identifier, e.g. 'dall-e-3', 'fal-ai/flux/dev'."
    )
    provider: str = Field(
        description="Provider identifier: 'fal', 'openai', 'azure-openai', 'google', 'runway', 'replicate', 'stability'."
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
    """Parameters for an image generation request."""

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
    """Normalized response returned by every image generation call."""

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
    cost: GenerationCost | None = Field(
        default=None,
        description="Cost information for this generation request.",
    )
    raw_response: dict[str, object] = Field(
        description="Unmodified provider response, preserved for debugging."
    )
    execution_metadata: ExecutionMetadata | None = Field(
        default=None,
        description="Timing and fallback attempt details captured by the orchestrator.",
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)


class ImageGenerationUpdate(BaseModel):
    """A progress event emitted during image generation polling."""

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


# ==================== Audio Generation Models ====================


class AudioGenerationConfig(BaseModel):
    """Configuration for a TTS or STS audio generation request."""

    model: str = Field(
        description="Model identifier, e.g. 'eleven_multilingual_v2', 'sonic-3', 'fal-ai/minimax/speech-2.8-hd'."
    )
    provider: str = Field(
        description="Provider identifier, e.g. 'elevenlabs', 'cartesia', 'fal'."
    )
    api_key: str | None = Field(
        default=None,
        description="API key for authenticating with the provider.",
    )
    timeout: int = Field(
        default=240,
        description="Maximum seconds to wait for generation to complete.",
    )
    mock: "MockConfig | None" = Field(
        default=None, description="If set, enables mock generation for testing."
    )
    fallback_configs: list["AudioGenerationConfig"] | None = Field(
        default=None,
        description="Ordered list of fallback configs to try on retryable errors.",
    )
    provider_config: dict[str, Any] = Field(
        default_factory=dict,
        description="Additional provider-specific configuration.",
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)


class TTSRequest(BaseModel):
    """Parameters for a text-to-speech request."""

    text: str = Field(description="Text to convert to speech.")
    voice_id: str | None = Field(
        default=None, description="Voice identifier (provider-specific)."
    )
    output_format: AudioOutputFormat = Field(
        default_factory=lambda: AudioOutputFormat(
            format="mp3", sample_rate=44100, bitrate=128
        ),
        description="Audio output format with codec, sample_rate (Hz), and bitrate (kbps).",
    )
    language_code: str | None = Field(
        default=None, description="Language hint for the provider."
    )
    voice_settings: dict[str, Any] | None = Field(
        default=None,
        description="Provider-specific voice settings (e.g. stability, speed, emotion, pitch).",
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
        known_fields = set(cls.model_fields.keys())
        extra = {k: v for k, v in data.items() if k not in known_fields}
        for k in extra.keys():
            _ = data.pop(k)
        extra_params.update(extra)
        data["extra_params"] = extra_params
        return data


class STSRequest(BaseModel):
    """Parameters for a speech-to-speech request."""

    audio: MediaType = Field(description="Input audio (bytes, URL, or MediaContent).")
    voice_id: str = Field(description="Voice identifier (provider-specific).")
    output_format: AudioOutputFormat = Field(
        default_factory=lambda: AudioOutputFormat(
            format="mp3", sample_rate=44100, bitrate=128
        ),
        description="Audio output format with codec, sample_rate (Hz), and bitrate (kbps).",
    )
    voice_settings: dict[str, Any] | None = Field(
        default=None,
        description="Provider-specific voice settings (e.g. stability, speed, emotion, pitch).",
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
        known_fields = set(cls.model_fields.keys())
        extra = {k: v for k, v in data.items() if k not in known_fields}
        for k in extra.keys():
            _ = data.pop(k)
        extra_params.update(extra)
        data["extra_params"] = extra_params
        return data


class TTSResponse(BaseModel):
    """Normalized response returned by every TTS call."""

    request_id: str = Field(description="Tarash-assigned unique ID for this request.")
    audio: str = Field(description="Base64-encoded audio bytes.")
    content_type: str | None = Field(
        default=None, description="MIME type of the audio, e.g. 'audio/mpeg'."
    )
    duration: float | None = Field(
        default=None, description="Audio duration in seconds, if available."
    )
    status: Literal["completed", "failed"] = Field(
        description="Final generation status."
    )
    is_mock: bool = Field(
        default=False,
        description="True if the response was produced by the mock provider.",
    )
    cost: GenerationCost | None = Field(
        default=None,
        description="Cost information for this generation request.",
    )
    raw_response: dict[str, object] = Field(
        description="Unmodified provider response, preserved for debugging."
    )
    execution_metadata: ExecutionMetadata | None = Field(
        default=None,
        description="Timing and fallback attempt details captured by the orchestrator.",
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)


class STSResponse(BaseModel):
    """Normalized response returned by every STS call."""

    request_id: str = Field(description="Tarash-assigned unique ID for this request.")
    audio: str = Field(description="Base64-encoded audio bytes.")
    content_type: str | None = Field(
        default=None, description="MIME type of the audio, e.g. 'audio/mpeg'."
    )
    duration: float | None = Field(
        default=None, description="Audio duration in seconds, if available."
    )
    status: Literal["completed", "failed"] = Field(
        description="Final generation status."
    )
    is_mock: bool = Field(
        default=False,
        description="True if the response was produced by the mock provider.",
    )
    cost: GenerationCost | None = Field(
        default=None,
        description="Cost information for this generation request.",
    )
    raw_response: dict[str, object] = Field(
        description="Unmodified provider response, preserved for debugging."
    )
    execution_metadata: ExecutionMetadata | None = Field(
        default=None,
        description="Timing and fallback attempt details captured by the orchestrator.",
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)


class TTSUpdate(BaseModel):
    """A progress event emitted during TTS generation."""

    request_id: str = Field(
        description="Same ID as the originating request, for correlation."
    )
    status: StatusType = Field(description="Current generation status.")
    progress_percent: int | None = Field(
        None, ge=0, le=100, description="Estimated completion percentage (0–100)."
    )
    update: dict[str, object] = Field(
        description="Raw event payload from the provider."
    )
    result: TTSResponse | None = Field(
        default=None, description="Final response, set only when status is 'completed'."
    )
    error: str | None = Field(
        default=None, description="Error message if status is 'failed'."
    )


class STSUpdate(BaseModel):
    """A progress event emitted during STS generation."""

    request_id: str = Field(
        description="Same ID as the originating request, for correlation."
    )
    status: StatusType = Field(description="Current generation status.")
    progress_percent: int | None = Field(
        None, ge=0, le=100, description="Estimated completion percentage (0–100)."
    )
    update: dict[str, object] = Field(
        description="Raw event payload from the provider."
    )
    result: STSResponse | None = Field(
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

    # ==================== TTS Generation ====================

    async def generate_tts_async(
        self,
        config: AudioGenerationConfig,
        request: TTSRequest,
        on_progress: TTSProgressCallback | None = None,
    ) -> TTSResponse:
        """Generate speech from text asynchronously.

        Args:
            config: Provider configuration with API key, model, and timeouts.
            request: TTS parameters (text, voice_id, output_format, etc.).
            on_progress: Optional callback invoked during generation.

        Returns:
            ``TTSResponse`` with base64-encoded audio and metadata.

        Raises:
            NotImplementedError: If this provider does not support TTS.
            TarashException: On any provider-level error.
        """
        ...

    def generate_tts(
        self,
        config: AudioGenerationConfig,
        request: TTSRequest,
        on_progress: TTSProgressCallback | None = None,
    ) -> TTSResponse:
        """Generate speech from text synchronously (blocking).

        Args:
            config: Provider configuration with API key, model, and timeouts.
            request: TTS parameters (text, voice_id, output_format, etc.).
            on_progress: Optional callback invoked during generation.

        Returns:
            ``TTSResponse`` with base64-encoded audio and metadata.

        Raises:
            NotImplementedError: If this provider does not support TTS.
            TarashException: On any provider-level error.
        """
        ...

    # ==================== STS Generation ====================

    async def generate_sts_async(
        self,
        config: AudioGenerationConfig,
        request: STSRequest,
        on_progress: STSProgressCallback | None = None,
    ) -> STSResponse:
        """Convert speech to speech asynchronously.

        Args:
            config: Provider configuration with API key, model, and timeouts.
            request: STS parameters (audio, voice_id, output_format, etc.).
            on_progress: Optional callback invoked during generation.

        Returns:
            ``STSResponse`` with base64-encoded audio and metadata.

        Raises:
            NotImplementedError: If this provider does not support STS.
            TarashException: On any provider-level error.
        """
        ...

    def generate_sts(
        self,
        config: AudioGenerationConfig,
        request: STSRequest,
        on_progress: STSProgressCallback | None = None,
    ) -> STSResponse:
        """Convert speech to speech synchronously (blocking).

        Args:
            config: Provider configuration with API key, model, and timeouts.
            request: STS parameters (audio, voice_id, output_format, etc.).
            on_progress: Optional callback invoked during generation.

        Returns:
            ``STSResponse`` with base64-encoded audio and metadata.

        Raises:
            NotImplementedError: If this provider does not support STS.
            TarashException: On any provider-level error.
        """
        ...


# ==================== Resolve Forward References ====================
# MockConfig is used as a string annotation ("MockConfig | None") in the config
# models above to avoid a circular import (mock.py imports from models.py).
# By this point all classes are defined, so mock.py can safely import them.
from tarash.tarash_gateway.mock import MockConfig  # noqa: E402, F811

VideoGenerationConfig.model_rebuild()
ImageGenerationConfig.model_rebuild()
AudioGenerationConfig.model_rebuild()
