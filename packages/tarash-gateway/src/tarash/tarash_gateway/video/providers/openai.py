"""OpenAI provider handler for Sora video generation."""

import asyncio
import time
import traceback
from typing import TYPE_CHECKING, Any

from typing_extensions import TypedDict

from tarash.tarash_gateway.video.exceptions import (
    GenerationFailedError,
    TarashException,
    ValidationError,
    handle_video_generation_errors,
)
from tarash.tarash_gateway.video.models import (
    ProgressCallback,
    VideoGenerationConfig,
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoGenerationUpdate,
)
from tarash.tarash_gateway.video.utils import (
    download_media_from_url,
    get_filename_from_url,
    validate_duration,
    validate_model_params,
)

try:
    from openai import AsyncOpenAI, OpenAI
except ImportError:
    pass

if TYPE_CHECKING:
    from openai import AsyncOpenAI, OpenAI


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


def parse_openai_video_status(video: Any) -> VideoGenerationUpdate:
    """Parse OpenAI video object to VideoGenerationUpdate.

    Args:
        video: OpenAI Video object from API response

    Returns:
        VideoGenerationUpdate with normalized status
    """
    status = getattr(video, "status", "processing")
    progress = getattr(video, "progress", None)

    # Normalize status to our standard values
    status_map = {
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
        update={
            "model": getattr(video, "model", None),
            "size": getattr(video, "size", None),
            "seconds": getattr(video, "seconds", None),
        },
    )


class OpenAIProviderHandler:
    """Handler for OpenAI Sora video generation."""

    def __init__(self):
        """Initialize handler (stateless, no config stored)."""
        if OpenAI is None:
            raise ImportError(
                "openai is required for OpenAI provider. "
                "Install with: pip install tarash-gateway[openai]"
            )

        self._sync_client_cache: dict[str, Any] = {}
        self._async_client_cache: dict[str, Any] = {}

    def _get_client(
        self, config: VideoGenerationConfig, client_type: str
    ) -> "AsyncOpenAI | OpenAI":
        """
        Get or create OpenAI client for the given config.

        Args:
            config: Provider configuration
            client_type: Type of client to return ("sync" or "async")

        Returns:
            OpenAI (sync) or AsyncOpenAI (async) client instance
        """
        # Use API key + base_url as cache key
        cache_key = f"{config.api_key}:{config.base_url or 'default'}:{client_type}"

        if client_type == "async":
            if cache_key not in self._async_client_cache:
                client_kwargs: dict[str, Any] = {
                    "api_key": config.api_key,
                    "timeout": config.timeout,
                }
                if config.base_url:
                    client_kwargs["base_url"] = config.base_url

                self._async_client_cache[cache_key] = AsyncOpenAI(**client_kwargs)
            return self._async_client_cache[cache_key]
        else:  # sync
            if cache_key not in self._sync_client_cache:
                client_kwargs: dict[str, Any] = {
                    "api_key": config.api_key,
                    "timeout": config.timeout,
                }
                if config.base_url:
                    client_kwargs["base_url"] = config.base_url

                self._sync_client_cache[cache_key] = OpenAI(**client_kwargs)
            return self._sync_client_cache[cache_key]

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
                # We have bytes already - pass as (filename, bytes, content_type) tuple
                content_bytes = media["content"]
                content_type = media["content_type"]
                # Extract extension from content type (e.g., "image/jpeg" -> "jpeg")
                ext = content_type.split("/")[-1] if "/" in content_type else "bin"
                filename = f"reference.{ext}"
                openai_params["input_reference"] = (
                    filename,
                    content_bytes,
                    content_type,
                )
            else:
                # String URL - download the image first
                url = str(media)
                content_bytes, content_type = download_media_from_url(
                    url, config.provider
                )
                filename = get_filename_from_url(url)
                openai_params["input_reference"] = (
                    filename,
                    content_bytes,
                    content_type,
                )

        return openai_params

    def _convert_response(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        request_id: str,
        video: Any,
        video_content: Any | None = None,
    ) -> VideoGenerationResponse:
        """
        Convert OpenAI video response to VideoGenerationResponse.

        Args:
            config: Provider configuration
            request: Original video generation request
            request_id: Our request ID
            video: OpenAI Video object
            video_content: Optional video content response

        Returns:
            Normalized VideoGenerationResponse
        """
        if video.status == "failed":
            error = getattr(video, "error", None)
            error_msg = (
                getattr(error, "message", "Video generation failed")
                if error
                else "Video generation failed"
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

        if video.status != "completed":
            raise GenerationFailedError(
                f"Video is not completed, status: {video.status}",
                provider=config.provider,
                raw_response={
                    "video": video.model_dump()
                    if hasattr(video, "model_dump")
                    else str(video)
                },
            )

        # Get video URL - it may be on the video object or from content download
        video_url = getattr(video, "url", None)
        if not video_url and video_content:
            # video_content might be a binary response or have a URL
            if hasattr(video_content, "url"):
                video_url = video_content.url
            elif hasattr(video_content, "content"):
                # Return as MediaContent
                return VideoGenerationResponse(
                    request_id=request_id,
                    video={
                        "content": video_content.content,
                        "content_type": "video/mp4",
                    },
                    content_type="video/mp4",
                    duration=float(getattr(video, "seconds", 0))
                    if hasattr(video, "seconds")
                    else None,
                    status="completed",
                    raw_response=video.model_dump()
                    if hasattr(video, "model_dump")
                    else {"id": video.id},
                    provider_metadata={},
                )

        if not video_url:
            raise GenerationFailedError(
                "No video URL found in response",
                provider=config.provider,
                raw_response={
                    "video": video.model_dump()
                    if hasattr(video, "model_dump")
                    else str(video)
                },
            )

        return VideoGenerationResponse(
            request_id=request_id,
            video=video_url,
            content_type="video/mp4",
            duration=float(getattr(video, "seconds", 0))
            if hasattr(video, "seconds")
            else None,
            resolution=getattr(video, "size", None),
            status="completed",
            raw_response=video.model_dump()
            if hasattr(video, "model_dump")
            else {"id": video.id},
            provider_metadata={},
        )

    def _handle_error(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        request_id: str,
        ex: Exception,
    ) -> TarashException:
        """Handle errors from OpenAI API."""
        if isinstance(ex, TarashException):
            return ex

        # Handle OpenAI specific errors
        from openai import BadRequestError

        if BadRequestError is not None and isinstance(ex, BadRequestError):
            error_msg = str(ex)
            return ValidationError(
                f"Invalid request parameters: {error_msg}",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response={
                    "error": error_msg,
                    "status_code": 400,
                    "traceback": traceback.format_exc(),
                },
            )

        error_type = type(ex).__name__
        error_msg = str(ex)

        return TarashException(
            f"Error while generating video: {error_msg}",
            provider=config.provider,
            model=config.model,
            request_id=request_id,
            raw_response={
                "error": error_msg,
                "error_type": error_type,
                "traceback": traceback.format_exc(),
            },
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

        Args:
            config: Provider configuration
            request: Video generation request
            on_progress: Optional async callback for progress updates

        Returns:
            Final VideoGenerationResponse when complete
        """
        client: AsyncOpenAI = self._get_client(config, "async")
        openai_params = self._convert_request(config, request)

        # Create video job
        video = await client.videos.create(**openai_params)
        request_id = video.id

        try:
            # Poll for completion
            poll_attempts = 0
            while poll_attempts < config.max_poll_attempts:
                if video.status == "completed" or video.status == "failed":
                    break

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

            if video.status not in ("completed", "failed"):
                raise GenerationFailedError(
                    f"Video generation timed out after {config.max_poll_attempts} attempts",
                    provider=config.provider,
                    model=config.model,
                    request_id=request_id,
                    raw_response={"status": "timeout"},
                )

            # Convert final response
            response = self._convert_response(config, request, request_id, video)
            return response

        except Exception as ex:
            raise self._handle_error(config, request, request_id, ex) from ex

    @handle_video_generation_errors
    def generate_video(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """
        Generate video synchronously (blocking).

        Args:
            config: Provider configuration
            request: Video generation request
            on_progress: Optional callback for progress updates

        Returns:
            Final VideoGenerationResponse
        """
        client = self._get_client(config, "sync")
        openai_params = self._convert_request(config, request)

        # Create video job
        video = client.videos.create(**openai_params)
        request_id = video.id

        try:
            # Poll for completion
            poll_attempts = 0
            while poll_attempts < config.max_poll_attempts:
                # Parse and report progress
                if on_progress:
                    update = parse_openai_video_status(video)
                    on_progress(update)

                # Check if done
                if video.status == "completed" or video.status == "failed":
                    break

                # Wait before next poll
                time.sleep(config.poll_interval)

                # Get updated video status
                video = client.videos.retrieve(video.id)
                poll_attempts += 1

            if video.status not in ("completed", "failed"):
                raise GenerationFailedError(
                    f"Video generation timed out after {config.max_poll_attempts} attempts",
                    provider=config.provider,
                    model=config.model,
                    request_id=request_id,
                    raw_response={"status": "timeout"},
                )

            # Convert final response
            response = self._convert_response(config, request, request_id, video)
            return response

        except Exception as ex:
            raise self._handle_error(config, request, request_id, ex) from ex
