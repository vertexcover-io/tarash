"""OpenAI provider handler for Sora video generation."""

from __future__ import annotations

import asyncio
import io
import time
import traceback
import uuid
from typing import TYPE_CHECKING, Any, Literal, overload

from typing_extensions import NotRequired, Required, TypedDict

from tarash.tarash_gateway.logging import ProviderLogger, log_error
from tarash.tarash_gateway.exceptions import (
    GenerationFailedError,
    HTTPConnectionError,
    HTTPError,
    TarashException,
    TimeoutError,
    ValidationError,
    handle_video_generation_errors,
)
from tarash.tarash_gateway.models import (
    ImageGenerationConfig,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ImageProgressCallback,
    MediaContent,
    ProgressCallback,
    StatusType,
    VideoGenerationConfig,
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoGenerationUpdate,
)
from tarash.tarash_gateway.utils import (
    download_media_from_url,
    get_filename_from_url,
    validate_duration,
    validate_model_params,
)
from tarash.tarash_gateway.providers.field_mappers import (
    FieldMapper,
    apply_field_mappers,
    get_field_mappers_from_registry,
    passthrough_field_mapper,
    size_field_mapper,
    quality_field_mapper,
    style_field_mapper,
    n_images_field_mapper,
)

# Import OpenAI types for type checking
if TYPE_CHECKING:
    from openai import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        AsyncOpenAI,
        AsyncAzureOpenAI,
        AzureOpenAI,
        BadRequestError,
        OpenAI,
        UnprocessableEntityError,
    )
    from openai.types import Video

# Runtime imports with error handling
try:
    from openai import AsyncOpenAI, AsyncAzureOpenAI, AzureOpenAI, OpenAI
    from openai import (
        APIStatusError,
        APIConnectionError,
        APITimeoutError,
        BadRequestError,
        UnprocessableEntityError,
    )
    from openai.types import Video

    has_openai = True
except ImportError:
    has_openai = False

# Logger name constant
_LOGGER_NAME = "tarash.tarash_gateway.providers.openai"


# ==================== Image Field Mappers ====================


# Generic image field mappers (fallback)
GENERIC_IMAGE_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "size": size_field_mapper(),
    "quality": quality_field_mapper(),
    "n": n_images_field_mapper(),
}

# GPT-Image-1.5 field mappers
GPT_IMAGE_15_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "size": size_field_mapper(
        allowed_values=["1024x1024", "1024x1792", "1792x1024", "auto"],
        provider="openai",
        model="gpt-image-1.5",
    ),
    "quality": quality_field_mapper(allowed_values=["low", "medium", "high", "auto"]),
    "n": n_images_field_mapper(min_value=1, max_value=4),
    "output_format": passthrough_field_mapper("output_format"),
    "background": passthrough_field_mapper("background"),
}

# DALL-E 3 field mappers
DALLE3_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "size": size_field_mapper(
        allowed_values=["1024x1024", "1024x1792", "1792x1024"],
        provider="openai",
        model="dall-e-3",
    ),
    "quality": quality_field_mapper(allowed_values=["standard", "hd"]),
    "style": style_field_mapper(allowed_values=["natural", "vivid"]),
    "n": n_images_field_mapper(min_value=1, max_value=1),
}

# DALL-E 2 field mappers
DALLE2_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "size": size_field_mapper(
        allowed_values=["256x256", "512x512", "1024x1024"],
        provider="openai",
        model="dall-e-2",
    ),
    "n": n_images_field_mapper(min_value=1, max_value=10),
}

# Image model registry
OPENAI_IMAGE_MODEL_REGISTRY: dict[str, dict[str, FieldMapper]] = {
    "gpt-image-1.5": GPT_IMAGE_15_FIELD_MAPPERS,
    "dall-e-3": DALLE3_FIELD_MAPPERS,
    "dall-e-2": DALLE2_FIELD_MAPPERS,
}


def get_openai_image_field_mappers(model_name: str) -> dict[str, FieldMapper]:
    """Get field mappers for OpenAI image model.

    Args:
        model_name: Model name (e.g., "gpt-image-1.5", "dall-e-3", "dall-e-2")

    Returns:
        Dict mapping API field names to FieldMapper objects
    """
    return get_field_mappers_from_registry(
        model_name,
        OPENAI_IMAGE_MODEL_REGISTRY,
        GENERIC_IMAGE_FIELD_MAPPERS,
    )


# ==================== Video Models ====================


# Supported video sizes for OpenAI Sora
# Maps aspect ratios to size strings in "WIDTHxHEIGHT" format
ASPECT_RATIO_TO_SIZE: dict[str, str] = {
    "16:9": "1280x720",
    "9:16": "720x1280",
    "1:1": "1024x1024",
    "16:10": "1792x1024",  # Landscape widescreen
    "10:16": "1024x1792",  # Portrait widescreen
}


class OpenAIVideoParams(TypedDict, total=False):
    """OpenAI Sora-specific parameters."""

    # Additional parameters can be added here as OpenAI expands the API
    pass


class OpenAIVideoResponse(TypedDict):
    """Response data for OpenAI video conversion.

    Contains the Video object and the HttpxBinaryResponseContent response object.
    """

    video: Required["Video"]  # OpenAI Video object (required)
    content: NotRequired[bytes | None]  # Optional - may be None if download failed
    content_type: NotRequired[str | None]  # Optional - may be None if download failed


def parse_openai_video_status(video: Video) -> VideoGenerationUpdate:
    """Parse OpenAI video object to VideoGenerationUpdate.

    Args:
        video: OpenAI Video object from API response

    Returns:
        VideoGenerationUpdate with normalized status
    """
    status = getattr(video, "status", "processing")
    progress = getattr(video, "progress", None)

    # Normalize status to our standard values
    status_map: dict[str, StatusType] = {
        "queued": "queued",
        "pending": "queued",
        "in_progress": "processing",
        "processing": "processing",
        "completed": "completed",
        "failed": "failed",
    }
    normalized_status = status_map.get(status, "processing")

    return VideoGenerationUpdate(
        request_id=video.id,
        status=normalized_status,
        progress_percent=progress,
        update={},
    )


class OpenAIProviderHandler:
    """Handler for OpenAI Sora video generation."""

    def __init__(self):
        """Initialize handler (stateless, no config stored)."""
        if not has_openai:
            raise ImportError(
                "openai is required for OpenAI provider. Install with: pip install tarash-gateway[openai]"
            )

    @overload
    def _get_client(
        self, config: VideoGenerationConfig, client_type: Literal["async"]
    ) -> "AsyncOpenAI | AsyncAzureOpenAI": ...

    @overload
    def _get_client(
        self, config: VideoGenerationConfig, client_type: Literal["sync"]
    ) -> "OpenAI | AzureOpenAI": ...

    def _get_client(
        self, config: VideoGenerationConfig, client_type: str
    ) -> "AsyncOpenAI | AsyncAzureOpenAI | OpenAI | AzureOpenAI":
        """
        Get or create OpenAI client for the given config.

        Args:
            config: Provider configuration
            client_type: Type of client to return ("sync" or "async")

        Returns:
            OpenAI (sync) or AsyncOpenAI (async) client instance
        """
        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)
        if client_type == "async":
            logger.debug(
                "Creating new async OpenAI client",
                {"base_url": config.base_url or "default"},
            )
            return AsyncOpenAI(
                api_key=config.api_key,
                timeout=float(config.timeout) if config.timeout else None,
                base_url=config.base_url or None,
            )
        else:  # sync
            logger.debug(
                "Creating new sync OpenAI client",
                {"base_url": config.base_url or "default"},
            )
            return OpenAI(
                api_key=config.api_key,
                timeout=float(config.timeout) if config.timeout else None,
                base_url=config.base_url or None,
            )

    def _validate_params(
        self, config: VideoGenerationConfig, request: VideoGenerationRequest
    ) -> dict[str, Any]:
        """
        Validate model_params using OpenAI-specific schema.

        Args:
            config: Provider configuration
            request: Video generation request

        Returns:
            Validated model parameters dict
        """
        if not request.extra_params:
            return {}

        return validate_model_params(
            schema=OpenAIVideoParams,
            data=request.extra_params,
            provider=config.provider,
            model=config.model,
        )

    def _convert_request(
        self, config: VideoGenerationConfig, request: VideoGenerationRequest
    ) -> dict[str, Any]:
        """
        Convert VideoGenerationRequest to OpenAI API format.

        Args:
            config: Provider configuration
            request: Video generation request

        Returns:
            Dict with parameters for openai.videos.create()

        Raises:
            ValidationError: If more than 1 image is provided
        """
        # Start with validated model_params
        openai_params: dict[str, Any] = self._validate_params(config, request)

        # Required parameters
        openai_params["model"] = config.model
        openai_params["prompt"] = request.prompt

        # Duration: OpenAI uses 'seconds' parameter
        # Sora 2: 4, 8, 12 seconds
        # Sora 2 Pro: 10, 15, 25 seconds
        if request.duration_seconds is not None:
            if "pro" in config.model.lower():
                allowed_durations = [10, 15, 25]
            else:
                allowed_durations = [4, 8, 12]
            validated_duration = validate_duration(
                request.duration_seconds,
                allowed_durations,
                config.provider,
                config.model,
            )
            openai_params["seconds"] = validated_duration

        # Size/resolution: OpenAI uses 'size' in format "WIDTHxHEIGHT"
        # Convert aspect_ratio to size if provided
        if request.aspect_ratio is not None:
            size = ASPECT_RATIO_TO_SIZE.get(request.aspect_ratio)
            if not size:
                supported_ratios = ", ".join(ASPECT_RATIO_TO_SIZE.keys())
                raise ValidationError(
                    f"Invalid aspect ratio: {request.aspect_ratio}. "
                    f"Supported ratios: {supported_ratios}",
                    provider=config.provider,
                )
            openai_params["size"] = size

        # Image-to-video: OpenAI only supports 1 input_reference image
        if request.image_list:
            if len(request.image_list) > 1:
                raise ValidationError(
                    f"OpenAI Sora only supports 1 reference image, got {len(request.image_list)}",
                    provider=config.provider,
                )

            # Convert first image to appropriate format for OpenAI FileTypes
            img = request.image_list[0]
            media = img["image"]

            # Check if it's a MediaContent dict (TypedDict) with bytes
            if (
                isinstance(media, dict)
                and "content" in media
                and "content_type" in media
            ):
                # We have bytes already - wrap in BytesIO for file-like object
                content_bytes = media["content"]
                content_type = media["content_type"]
                # Extract extension from content type (e.g., "image/jpeg" -> "jpeg")
                ext = content_type.split("/")[-1] if "/" in content_type else "bin"
                filename = f"reference.{ext}"
                openai_params["input_reference"] = io.BytesIO(content_bytes)
                openai_params["input_reference"].name = filename
            else:
                # String URL - download the image first
                url = str(media)
                content_bytes, content_type = download_media_from_url(
                    url, config.provider
                )
                filename = get_filename_from_url(url)
                openai_params["input_reference"] = (
                    filename,
                    io.BytesIO(content_bytes),
                    content_type,
                )

        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)
        logger.info(
            "Mapped request to provider format",
            {"converted_request": openai_params},
            redact=True,
        )

        return openai_params

    def _convert_response(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        request_id: str,
        provider_response: OpenAIVideoResponse,
    ) -> VideoGenerationResponse:
        """
        Convert OpenAI video response to VideoGenerationResponse.

        Args:
            config: Provider configuration
            request: Original video generation request
            request_id: Our request ID
            response_data: TypedDict containing video object and optional content

        Returns:
            Normalized VideoGenerationResponse
        """
        video = provider_response["video"]
        video_content = provider_response.get("content")
        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)
        logger = logger.with_request_id(request_id)

        if video.status == "failed":
            error = getattr(video, "error", None)
            error_msg = (
                getattr(error, "message", "Video generation failed")
                if error
                else "Video generation failed"
            )
            logger.error(
                "OpenAI video generation failed",
                {"video_id": video.id, "error_message": error_msg},
            )
            raise GenerationFailedError(
                error_msg,
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response={
                    "video": video.model_dump()
                    if hasattr(video, "model_dump")
                    else str(video)
                },
            )
        elif video_content is None or len(video_content) == 0:
            logger.error(
                "No video content found in OpenAI response",
                {"video_id": video.id},
            )
            raise GenerationFailedError(
                "No video content found in response",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response=video.model_dump(),
            )

        # Type narrowing: video_content is guaranteed to be bytes at this point
        # due to the None check above
        assert video_content is not None, "video_content should not be None"

        content_type_value = provider_response.get("content_type") or "video/mp4"
        video_media: MediaContent = {
            "content": video_content,
            "content_type": content_type_value,
        }
        return VideoGenerationResponse(
            request_id=request_id,
            video=video_media,
            content_type=content_type_value,
            duration=float(getattr(video, "seconds", 0))
            if hasattr(video, "seconds")
            else None,
            resolution=getattr(video, "size", None),
            status="completed",
            raw_response=video.model_dump(),
            provider_metadata={},
        )

    def _handle_error(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        request_id: str | None,
        ex: Exception,
    ) -> TarashException:
        """Handle errors from OpenAI API."""
        if isinstance(ex, TarashException):
            return ex

        # API timeout errors
        if has_openai and isinstance(ex, APITimeoutError):
            return TimeoutError(
                f"Request timed out: {str(ex)}",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response={"error": str(ex)},
                timeout_seconds=config.timeout,
            )

        # API connection errors
        if has_openai and isinstance(ex, APIConnectionError):
            return HTTPConnectionError(
                f"Connection error: {str(ex)}",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response={"error": str(ex)},
            )

        # API status errors (4xx, 5xx)
        if has_openai and isinstance(ex, APIStatusError):
            # Extract message from body or use default
            error_message: str
            body = ex.body
            if isinstance(body, dict):
                # OpenAI SDK body type is complex; safely extract message
                body_message = body.get("message")  # type: ignore[union-attr]
                if isinstance(body_message, str):
                    error_message = body_message
                else:
                    error_message = ex.message
            else:
                error_message = ex.message

            raw_response: dict[str, object] = {
                "status_code": ex.status_code,
                "message": error_message,
                "type": ex.type,
                "param": ex.param,
                "code": ex.code,
            }

            # Validation errors (400, 422)
            if isinstance(ex, (BadRequestError, UnprocessableEntityError)):
                return ValidationError(
                    error_message,
                    provider=config.provider,
                    model=config.model,
                    request_id=request_id,
                    raw_response=raw_response,
                )

            # All other HTTP errors (401, 403, 429, 500, etc.)
            return HTTPError(
                error_message,
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response=raw_response,
                status_code=ex.status_code,
            )

        # Unknown errors
        log_error(
            f"OpenAI unknown error: {str(ex)}",
            context={
                "provider": config.provider,
                "model": config.model,
                "request_id": request_id,
                "error_type": type(ex).__name__,
            },
            logger_name=_LOGGER_NAME,
            exc_info=True,
        )
        return GenerationFailedError(
            f"Error while generating video: {str(ex)}",
            provider=config.provider,
            model=config.model,
            request_id=request_id,
            raw_response={
                "error": str(ex),
                "error_type": type(ex).__name__,
                "traceback": traceback.format_exc(),
            },
        )

    def _log_poll_status(
        self,
        logger: ProviderLogger,
        video: Video,
        poll_attempts: int,
        max_poll_attempts: int,
        last_logged_progress: int,
    ) -> int:
        """
        Log polling status for video generation.

        Returns the updated last_logged_progress value.
        """
        current_status = video.status
        progress = video.progress
        extra: dict[str, object] = {
            "status": current_status,
            "progress_percent": progress,
            "poll_attempt": poll_attempts + 1,
            "max_attempts": max_poll_attempts,
        }

        if current_status == "completed" or current_status == "failed":
            logger.info("Progress status update", extra)
        elif progress != last_logged_progress:
            logger.info("Progress status update", extra)
        elif poll_attempts % 10 == 0:
            logger.debug("Progress status update", extra)

        return last_logged_progress

    def _check_for_timeout(
        self,
        logger: ProviderLogger,
        video: Video,
        config: VideoGenerationConfig,
        poll_attempts: int,
    ) -> None:
        """
        Check if video generation timed out and raise error if so.

        Args:
            logger: ProviderLogger with request_id bound
            video: Current video object
            config: Provider configuration
            poll_attempts: Number of poll attempts made

        Raises:
            TimeoutError: If video generation timed out
        """
        if video.status not in ("completed", "failed"):
            timeout_seconds = config.max_poll_attempts * config.poll_interval
            logger.error(
                "OpenAI video generation timed out",
                {
                    "video_id": video.id,
                    "poll_attempts": poll_attempts,
                    "max_attempts": config.max_poll_attempts,
                    "timeout_seconds": timeout_seconds,
                },
            )
            raise TimeoutError(
                f"Video generation timed out after {config.max_poll_attempts} attempts ({timeout_seconds}s)",
                provider=config.provider,
                model=config.model,
                request_id=logger.request_id,
                raw_response={"status": "timeout", "poll_attempts": poll_attempts},
                timeout_seconds=timeout_seconds,
            )

    async def _download_video_async(
        self,
        client: "AsyncOpenAI",
        video: Video,
        logger: ProviderLogger,
    ) -> tuple[bytes | None, str | None]:
        """
        Download video content asynchronously if completed.

        Args:
            client: Async OpenAI client
            video: Video object
            logger: ProviderLogger with request_id bound

        Returns:
            Tuple of (content bytes, content_type) or (None, None) if failed
        """
        if video.status == "completed":
            logger.info(
                "OpenAI video generation completed, downloading content",
                {"video_id": video.id},
            )
            video_response = await client.videos.download_content(video.id)
            content = await video_response.response.aread()
            content_type = "video/mp4"
            logger.debug(
                "OpenAI video content downloaded",
                {"video_id": video.id, "content_size_bytes": len(content)},
            )
            return content, content_type
        else:
            return None, None

    def _download_video_sync(
        self,
        client: "OpenAI",
        video: Video,
        logger: ProviderLogger,
    ) -> tuple[bytes | None, str | None]:
        """
        Download video content synchronously if completed.

        Args:
            client: Sync OpenAI client
            video: Video object
            logger: ProviderLogger with request_id bound

        Returns:
            Tuple of (content bytes, content_type) or (None, None) if failed
        """
        if video.status == "completed":
            logger.info(
                "OpenAI video generation completed, downloading content",
                {"video_id": video.id},
            )
            video_response = client.videos.download_content(video.id)
            content = video_response.read()
            content_type = "video/mp4"
            logger.debug(
                "OpenAI video content downloaded",
                {"video_id": video.id, "content_size_bytes": len(content)},
            )
            return content, content_type
        else:
            return None, None

    def _create_final_response(
        self,
        video: Video,
        content: bytes | None,
        content_type: str | None,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        logger: ProviderLogger,
    ) -> VideoGenerationResponse:
        """
        Create final VideoGenerationResponse with logging.

        Args:
            video: Video object
            content: Downloaded video content or None
            content_type: Content type or None
            config: Provider configuration
            request: Original request
            logger: ProviderLogger with request_id bound

        Returns:
            Final VideoGenerationResponse
        """
        logger.debug(
            "Request complete",
            {
                "response": {
                    "video_status": video.status,
                    "content_size": len(content) if content else 0,
                },
            },
            redact=True,
        )

        response_data: OpenAIVideoResponse = {
            "video": video,
            "content": content,
            "content_type": content_type,
        }
        request_id = logger.request_id or ""
        response = self._convert_response(config, request, request_id, response_data)

        logger.info(
            "Final generated response",
            {"response": response},
            redact=True,
        )

        return response

    async def _poll_and_download_async(
        self,
        client: "AsyncOpenAI",
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        video: Video,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """
        Poll for video completion and download content (async).

        Args:
            client: Async OpenAI client
            config: Provider configuration
            request: Video generation request
            video: Initial video object from create call
            on_progress: Optional async callback for progress updates

        Returns:
            Final VideoGenerationResponse

        Raises:
            GenerationFailedError: If polling times out or generation fails
        """
        request_id = video.id
        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)
        logger = logger.with_request_id(request_id)

        # Poll for completion
        poll_attempts = 0
        last_logged_progress = -1
        while poll_attempts < config.max_poll_attempts:
            if video.status == "completed" or video.status == "failed":
                break

            # Log status updates
            last_logged_progress = self._log_poll_status(
                logger,
                video,
                poll_attempts,
                config.max_poll_attempts,
                last_logged_progress,
            )

            start = time.time()
            if on_progress:
                update = parse_openai_video_status(video)
                result = on_progress(update)
                if asyncio.iscoroutine(result):
                    await result

            end_time = time.time()
            if end_time - start < config.poll_interval:
                await asyncio.sleep(config.poll_interval - (end_time - start))

            # Get updated video status
            video = await client.videos.retrieve(video.id)
            poll_attempts += 1

        # Check if timed out
        self._check_for_timeout(logger, video, config, poll_attempts)

        # Download content if completed
        content, content_type = await self._download_video_async(client, video, logger)

        # Create and return final response
        return self._create_final_response(
            video, content, content_type, config, request, logger
        )

    def _poll_and_download_sync(
        self,
        client: "OpenAI",
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        video: Video,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """
        Poll for video completion and download content (sync).

        Args:
            client: Sync OpenAI client
            config: Provider configuration
            request: Video generation request
            video: Initial video object from create call
            on_progress: Optional callback for progress updates

        Returns:
            Final VideoGenerationResponse

        Raises:
            GenerationFailedError: If polling times out or generation fails
        """
        request_id = video.id
        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)
        logger = logger.with_request_id(request_id)

        # Poll for completion
        poll_attempts = 0
        last_logged_progress = -1
        while poll_attempts < config.max_poll_attempts:
            # Parse and report progress
            if on_progress:
                update = parse_openai_video_status(video)
                on_progress(update)

            # Check if done
            if video.status == "completed" or video.status == "failed":
                break

            # Log status updates
            last_logged_progress = self._log_poll_status(
                logger,
                video,
                poll_attempts,
                config.max_poll_attempts,
                last_logged_progress,
            )

            # Wait before next poll
            time.sleep(config.poll_interval)

            # Get updated video status
            video = client.videos.retrieve(video.id)
            poll_attempts += 1

        # Check if timed out
        self._check_for_timeout(logger, video, config, poll_attempts)

        # Download content if completed
        content, content_type = self._download_video_sync(client, video, logger)

        # Create and return final response
        return self._create_final_response(
            video, content, content_type, config, request, logger
        )

    @handle_video_generation_errors
    async def generate_video_async(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """
        Generate video asynchronously via OpenAI Sora.

        Supports both video generation and video remix:
        - Generation: Uses client.videos.create()
        - Remix: Uses client.videos.remix() when video_id is in extra_params

        Args:
            config: Provider configuration
            request: Video generation request (use extra_params={"video_id": "..."} for remix)
            on_progress: Optional async callback for progress updates

        Returns:
            Final VideoGenerationResponse when complete
        """
        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)
        client: AsyncOpenAI = self._get_client(config, "async")
        openai_params = self._convert_request(config, request)

        # Check if this is a remix request (video_id in extra_params)
        video_id = (
            request.extra_params.get("video_id") if request.extra_params else None
        )
        is_remix = video_id is not None

        logger.debug(
            "Starting API call",
            {"is_remix": is_remix, "video_id": video_id if is_remix else None},
        )

        # Create video job or remix existing video
        request_id: str | None = None
        try:
            if is_remix:
                # Use remix endpoint: POST /videos/{video_id}/remix
                # Remove video_id from params as it goes in the URL
                remix_params = {k: v for k, v in openai_params.items() if k != "model"}
                video = await client.videos.remix(
                    video_id=str(video_id), **remix_params
                )
            else:
                # Use standard create endpoint
                video = await client.videos.create(**openai_params)

            request_id = video.id
            logger = logger.with_request_id(request_id)

            logger.debug(
                "Request submitted",
                {"is_remix": is_remix},
            )
            return await self._poll_and_download_async(
                client, config, request, video, on_progress
            )
        except Exception as ex:
            raise self._handle_error(config, request, request_id, ex)

    @handle_video_generation_errors
    def generate_video(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """
        Generate video synchronously (blocking).

        Supports both video generation and video remix:
        - Generation: Uses client.videos.create()
        - Remix: Uses client.videos.remix() when video_id is in extra_params

        Args:
            config: Provider configuration
            request: Video generation request (use extra_params={"video_id": "..."} for remix)
            on_progress: Optional callback for progress updates

        Returns:
            Final VideoGenerationResponse
        """
        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)
        client: OpenAI = self._get_client(config, "sync")
        openai_params = self._convert_request(config, request)

        # Check if this is a remix request (video_id in extra_params)
        video_id = (
            request.extra_params.get("video_id") if request.extra_params else None
        )
        is_remix = video_id is not None

        logger.debug(
            "Starting API call",
            {"is_remix": is_remix, "video_id": video_id if is_remix else None},
        )
        request_id: str | None = None
        try:
            # Create video job or remix existing video
            if is_remix:
                # Use remix endpoint: POST /videos/{video_id}/remix
                # Remove video_id from params as it goes in the URL
                remix_params = {k: v for k, v in openai_params.items() if k != "model"}
                video = client.videos.remix(video_id=str(video_id), **remix_params)
            else:
                # Use standard create endpoint
                video = client.videos.create(**openai_params)

            request_id = video.id
            logger = logger.with_request_id(request_id)

            logger.debug(
                "Request submitted",
                {"is_remix": is_remix},
            )

            return self._poll_and_download_sync(
                client, config, request, video, on_progress
            )
        except Exception as ex:
            raise self._handle_error(config, request, request_id, ex)

    # ==================== Image Generation ====================

    def _convert_image_request(
        self, config: ImageGenerationConfig, request: ImageGenerationRequest
    ) -> dict[str, Any]:
        """Convert ImageGenerationRequest to OpenAI API format.

        Args:
            config: Provider configuration
            request: Image generation request

        Returns:
            Dict with parameters for openai.images.generate()
        """
        field_mappers = get_openai_image_field_mappers(config.model)
        openai_params = apply_field_mappers(field_mappers, request)

        openai_params["model"] = config.model

        if request.extra_params:
            for key, value in request.extra_params.items():
                if key not in openai_params and value is not None:
                    openai_params[key] = value

        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)
        logger.info(
            "Mapped image request to provider format",
            {"converted_request": openai_params},
            redact=True,
        )

        return openai_params

    def _convert_image_response(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        request_id: str,
        provider_response: Any,
    ) -> ImageGenerationResponse:
        """Convert OpenAI image response to ImageGenerationResponse.

        Args:
            config: Provider configuration
            request: Original image generation request
            request_id: Our request ID
            provider_response: OpenAI API response

        Returns:
            Normalized ImageGenerationResponse
        """
        images = [img.url for img in provider_response.data]

        revised_prompt = None
        if provider_response.data and hasattr(
            provider_response.data[0], "revised_prompt"
        ):
            revised_prompt = provider_response.data[0].revised_prompt

        return ImageGenerationResponse(
            request_id=request_id,
            images=images,
            content_type="image/png",
            status="completed",
            revised_prompt=revised_prompt,
            raw_response=provider_response.model_dump()
            if hasattr(provider_response, "model_dump")
            else {},
            provider_metadata={},
        )

    async def generate_image_async(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        on_progress: ImageProgressCallback | None = None,
    ) -> ImageGenerationResponse:
        """Generate image asynchronously via OpenAI.

        Args:
            config: Provider configuration
            request: Image generation request
            on_progress: Optional async callback for progress updates

        Returns:
            ImageGenerationResponse when complete
        """
        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)
        request_id = f"openai-image-{uuid.uuid4()}"
        logger = logger.with_request_id(request_id)

        client: AsyncOpenAI = self._get_client(config, "async")
        openai_params = self._convert_image_request(config, request)

        logger.debug("Starting image generation API call")

        try:
            response = await client.images.generate(**openai_params)

            logger.info(
                "Image generation completed", {"num_images": len(response.data)}
            )

            return self._convert_image_response(config, request, request_id, response)
        except Exception as ex:
            raise self._handle_error(config, request, request_id, ex)

    def generate_image(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        on_progress: ImageProgressCallback | None = None,
    ) -> ImageGenerationResponse:
        """Generate image synchronously (blocking).

        Args:
            config: Provider configuration
            request: Image generation request
            on_progress: Optional callback for progress updates

        Returns:
            ImageGenerationResponse
        """
        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)
        request_id = f"openai-image-{uuid.uuid4()}"
        logger = logger.with_request_id(request_id)

        client: OpenAI = self._get_client(config, "sync")
        openai_params = self._convert_image_request(config, request)

        logger.debug("Starting image generation API call")

        try:
            response = client.images.generate(**openai_params)

            logger.info(
                "Image generation completed", {"num_images": len(response.data)}
            )

            return self._convert_image_response(config, request, request_id, response)
        except Exception as ex:
            raise self._handle_error(config, request, request_id, ex)
