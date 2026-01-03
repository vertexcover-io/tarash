"""Runway ML provider handler for video generation."""

import asyncio
import io
import time
from typing import Any, Literal

from runwayml.lib.polling import AsyncAwaitableTaskRetrieveResponse
from typing_extensions import TypedDict

from tarash.tarash_gateway.logging import log_debug, log_error, log_info
from tarash.tarash_gateway.video.exceptions import (
    GenerationFailedError,
    HTTPConnectionError,
    HTTPError,
    TarashException,
    TimeoutError,
    ValidationError,
    handle_video_generation_errors,
)
from tarash.tarash_gateway.video.models import (
    MediaType,
    ProgressCallback,
    VideoGenerationConfig,
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoGenerationUpdate,
)
from tarash.tarash_gateway.video.utils import validate_duration, validate_model_params

try:
    from runwayml import AsyncRunwayML, RunwayML
    from runwayml.lib.polling import AsyncNewTaskCreatedResponse, NewTaskCreatedResponse

    # Import specific exception types from runwayml SDK
    from runwayml import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        BadRequestError,
        UnprocessableEntityError,
    )
except ImportError:
    pass

# Logger name constant
_LOGGER_NAME = "tarash.tarash_gateway.video.providers.runway"

# Aspect ratio to size mappings
TEXT_TO_VIDEO_RATIOS: dict[str, str] = {
    "16:9": "1280:720",
    "9:16": "720:1280",
    "16:9-wide": "1080:1920",
    "9:16-wide": "1920:1080",
}

IMAGE_VIDEO_RATIOS: dict[str, str] = {
    "16:9": "1280:720",
    "9:16": "720:1280",
    "4:3": "1104:832",
    "3:4": "832:1104",
    "1:1": "960:960",
    "21:9": "1584:672",
    "480p": "848:480",
    "vga": "640:480",
}

# Terminal task statuses
_TERMINAL_STATUSES = ("SUCCEEDED", "FAILED", "CANCELLED")


class RunwayVideoParams(TypedDict, total=False):
    """Runway-specific parameters."""

    content_moderation: dict[str, str]


def _get_endpoint_from_model(
    model: str, has_image: bool, has_video: bool
) -> Literal["text_to_video", "image_to_video", "video_to_video"]:
    """Determine which Runway endpoint to use based on model and input type."""
    model_lower = model.lower()

    # Video-to-video models
    if "aleph" in model_lower:
        if not has_video:
            raise ValidationError(
                f"Model {model} requires video input", provider="runway"
            )
        return "video_to_video"

    # Image-to-video only models
    if "turbo" in model_lower and "veo" not in model_lower:
        if not has_image:
            raise ValidationError(
                f"Model {model} requires image input", provider="runway"
            )
        return "image_to_video"

    # VEO models support both text and image-to-video
    if "veo" in model_lower:
        if has_video:
            raise ValidationError(
                f"Model {model} does not support video input", provider="runway"
            )
        return "image_to_video" if has_image else "text_to_video"

    # Default behavior for unknown models
    if has_video:
        raise ValidationError(
            f"Model {model} does not support video input", provider="runway"
        )
    return "image_to_video" if has_image else "text_to_video"


def _convert_media_to_file(media: MediaType, default_name: str) -> io.BytesIO | str:
    """Convert MediaType to file object or URL string."""
    if isinstance(media, str):
        return media

    # MediaContent dict with bytes
    content_bytes = media["content"]
    content_type = media["content_type"]
    ext = content_type.split("/")[-1] if "/" in content_type else "bin"
    file_obj = io.BytesIO(content_bytes)
    file_obj.name = f"{default_name}.{ext}"
    return file_obj


def _convert_aspect_ratio(
    aspect_ratio: str | None, ratio_map: dict[str, str], endpoint: str, provider: str
) -> str:
    """Convert aspect ratio to Runway format with validation."""
    if not aspect_ratio:
        return "1280:720"  # Default

    ratio = ratio_map.get(aspect_ratio)
    if not ratio:
        supported = ", ".join(ratio_map.keys())
        raise ValidationError(
            f"Invalid aspect ratio for {endpoint}: {aspect_ratio}. Supported: {supported}",
            provider=provider,
        )
    return ratio


def _extract_video_url(output: Any) -> str | None:
    """Extract video URL from task output.

    Args:
        output: Task output (type not exported by SDK, can be list/str/dict)

    Returns:
        Video URL string or None if not found
    """
    if isinstance(output, list):
        return output[0] if output else None
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        return output.get("url")
    return None


def parse_runway_task_status(
    task: AsyncAwaitableTaskRetrieveResponse,
) -> VideoGenerationUpdate:
    """Parse Runway task object to VideoGenerationUpdate.

    Args:
        task: Runway task object (Task type not exported by SDK)
    """
    status_map = {
        "PENDING": "queued",
        "THROTTLED": "queued",
        "RUNNING": "processing",
        "SUCCEEDED": "completed",
        "FAILED": "failed",
        "CANCELLED": "failed",
    }

    return VideoGenerationUpdate(
        request_id=task.id,
        status=status_map.get(task.status, "processing"),
        progress_percent=None,
        update={"raw_status": task.status},
    )


class RunwayProviderHandler:
    """Handler for Runway ML video generation."""

    def __init__(self):
        """Initialize handler (stateless, no config stored)."""
        if RunwayML is None:
            raise ImportError(
                "runwayml is required for Runway provider. "
                "Install with: pip install tarash-gateway[runway]"
            )
        self._sync_client_cache: dict[str, RunwayML] = {}
        self._async_client_cache: dict[str, AsyncRunwayML] = {}

    def _get_client(
        self, config: VideoGenerationConfig, client_type: str
    ) -> RunwayML | AsyncRunwayML:
        """Get or create Runway client for the given config."""
        cache_key = f"{config.api_key}:{client_type}"
        cache = (
            self._async_client_cache
            if client_type == "async"
            else self._sync_client_cache
        )

        if cache_key not in cache:
            log_debug(
                f"Creating new {client_type} Runway client",
                context={"provider": config.provider, "model": config.model},
                logger_name=_LOGGER_NAME,
            )
            ClientClass = AsyncRunwayML if client_type == "async" else RunwayML
            cache[cache_key] = ClientClass(api_key=config.api_key)

        return cache[cache_key]

    def _validate_params(
        self, config: VideoGenerationConfig, request: VideoGenerationRequest
    ) -> dict[str, Any]:
        """Validate model_params using Runway-specific schema."""
        return (
            validate_model_params(
                schema=RunwayVideoParams,
                data=request.extra_params,
                provider=config.provider,
                model=config.model,
            )
            if request.extra_params
            else {}
        )

    def _convert_request(
        self, config: VideoGenerationConfig, request: VideoGenerationRequest
    ) -> tuple[str, dict[str, Any]]:
        """Convert VideoGenerationRequest to Runway API format."""
        params: dict[str, Any] = self._validate_params(config, request)

        # Determine endpoint
        endpoint = _get_endpoint_from_model(
            config.model, bool(request.image_list), bool(request.video)
        )

        # Common parameters
        params["model"] = config.model
        if request.seed is not None:
            params["seed"] = int(request.seed)
        if "content_moderation" in request.extra_params:
            params["content_moderation"] = request.extra_params["content_moderation"]

        # Endpoint-specific parameters
        if endpoint == "text_to_video":
            params["prompt_text"] = request.prompt
            params["ratio"] = _convert_aspect_ratio(
                request.aspect_ratio,
                TEXT_TO_VIDEO_RATIOS,
                "text-to-video",
                config.provider,
            )
            if request.duration_seconds is not None:
                params["duration"] = validate_duration(
                    request.duration_seconds, [4, 6, 8], config.provider, config.model
                )
            if request.generate_audio is not None:
                params["audio"] = request.generate_audio

        elif endpoint == "image_to_video":
            # Extract reference image (validated by endpoint routing)
            reference_images = [
                img
                for img in request.image_list
                if img["type"] in ("reference", "first_frame")
            ]
            if not reference_images:
                raise ValidationError(
                    "No reference image found", provider=config.provider
                )

            params["prompt_image"] = _convert_media_to_file(
                reference_images[0]["image"], "prompt_image"
            )
            params["ratio"] = _convert_aspect_ratio(
                request.aspect_ratio,
                IMAGE_VIDEO_RATIOS,
                "image-to-video",
                config.provider,
            )
            if request.prompt:
                params["prompt_text"] = request.prompt
            if request.duration_seconds is not None:
                if not 2 <= request.duration_seconds <= 10:
                    raise ValidationError(
                        f"Duration must be 2-10 seconds, got {request.duration_seconds}",
                        provider=config.provider,
                    )
                params["duration"] = int(request.duration_seconds)

        elif endpoint == "video_to_video":
            params["video_uri"] = _convert_media_to_file(request.video, "input_video")
            params["prompt_text"] = request.prompt
            params["ratio"] = _convert_aspect_ratio(
                request.aspect_ratio,
                IMAGE_VIDEO_RATIOS,
                "video-to-video",
                config.provider,
            )

            # Add image references if provided
            if request.image_list:
                reference_imgs = [
                    img for img in request.image_list if img["type"] == "reference"
                ]
                if reference_imgs:
                    params["references"] = [
                        {"type": "image", "uri": str(img["image"])}
                        for img in reference_imgs
                    ]

        log_info(
            "Mapped request to provider format",
            context={
                "provider": config.provider,
                "model": config.model,
                "endpoint": endpoint,
                "converted_request": params,
            },
            logger_name=_LOGGER_NAME,
            redact=True,
        )

        return endpoint, params

    def _convert_response(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        request_id: str,
        task: Any,  # Task type not exported by SDK
    ) -> VideoGenerationResponse:
        """Convert Runway task response to VideoGenerationResponse.

        Args:
            config: Provider configuration
            request: Original request
            request_id: Request ID
            task: Runway task object (type not exported by SDK)
        """
        if task.status == "FAILED":
            error_msg = str(getattr(task, "error", "Video generation failed"))
            log_error(
                "Runway video generation failed",
                context={
                    "provider": config.provider,
                    "model": config.model,
                    "request_id": request_id,
                    "error": error_msg,
                },
                logger_name=_LOGGER_NAME,
            )
            raise GenerationFailedError(
                error_msg,
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response={
                    "task_id": task.id,
                    "status": task.status,
                    "error": error_msg,
                },
            )

        output = getattr(task, "output", None)
        video_url = _extract_video_url(output) if output else None

        if not video_url:
            raise GenerationFailedError(
                "No video URL found in task output",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response={"task_id": task.id, "output": str(output)},
            )

        return VideoGenerationResponse(
            request_id=request_id,
            video=video_url,
            content_type="video/mp4",
            status="completed",
            raw_response={
                "task_id": task.id,
                "status": task.status,
                "output": str(output),
            },
            provider_metadata={},
        )

    def _handle_error(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        request_id: str,
        ex: Exception,
    ) -> TarashException:
        """Handle errors from Runway API."""
        if isinstance(ex, TarashException):
            return ex

        # API timeout errors
        if isinstance(ex, APITimeoutError):
            return TimeoutError(
                f"Request timed out: {ex.message}",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response={"error": ex.message},
                timeout_seconds=config.timeout,
            )

        # API connection errors
        if isinstance(ex, APIConnectionError):
            return HTTPConnectionError(
                f"Connection error: {ex.message}",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response={"error": ex.message},
            )

        # API status errors (4xx, 5xx)
        if isinstance(ex, APIStatusError):
            raw_response = {
                "status_code": ex.status_code,
                "message": ex.message,
                "body": ex.body,
            }

            # Validation errors (400, 422)
            if isinstance(ex, (BadRequestError, UnprocessableEntityError)):
                return ValidationError(
                    ex.message,
                    provider=config.provider,
                    model=config.model,
                    request_id=request_id,
                    raw_response=raw_response,
                )

            # All other HTTP errors (401, 403, 429, 500, etc.)
            return HTTPError(
                ex.message,
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response=raw_response,
                status_code=ex.status_code,
            )

        # Unknown errors
        log_error(
            f"Runway unknown error: {str(ex)}",
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
            raw_response={"error": str(ex), "error_type": type(ex).__name__},
        )

    def _call_endpoint(
        self, client: RunwayML | AsyncRunwayML, endpoint: str, params: dict[str, Any]
    ) -> NewTaskCreatedResponse | AsyncNewTaskCreatedResponse:
        """Call the appropriate Runway endpoint."""
        endpoint_map = {
            "text_to_video": lambda: client.text_to_video.create(**params),
            "image_to_video": lambda: client.image_to_video.create(**params),
            "video_to_video": lambda: client.video_to_video.create(**params),
        }
        return endpoint_map[endpoint]()

    async def _poll_until_complete(
        self,
        client: RunwayML | AsyncRunwayML,
        task_id: str,
        config: VideoGenerationConfig,
        on_progress: ProgressCallback | None,
        is_async: bool,
    ) -> Any:  # Returns Task (type not exported by SDK)
        """Poll task until completion (unified for sync/async).

        Args:
            client: Runway client (sync or async)
            task: Initial task object (type not exported by SDK)
            config: Provider configuration
            on_progress: Optional progress callback
            is_async: Whether to use async operations

        Returns:
            Completed task object (type not exported by SDK)
        """
        request_id = task_id
        poll_attempts = 0

        while poll_attempts < config.max_poll_attempts:
            # Wait and retrieve updated status
            if is_async:
                await asyncio.sleep(config.poll_interval)
                task = await client.tasks.retrieve(task_id)
            else:
                time.sleep(config.poll_interval)
                task = client.tasks.retrieve(task_id)

            poll_attempts += 1

            # Report progress
            if on_progress:
                update = parse_runway_task_status(task)
                result = on_progress(update)
                if is_async and asyncio.iscoroutine(result):
                    await result

            # Check if terminal
            if task.status in _TERMINAL_STATUSES:
                break

            # Log progress
            log_info(
                "Progress status update",
                context={
                    "provider": config.provider,
                    "model": config.model,
                    "request_id": request_id,
                    "status": task.status,
                    "poll_attempt": poll_attempts + 1,
                },
                logger_name=_LOGGER_NAME,
            )

        # Check timeout
        if task.status not in _TERMINAL_STATUSES:
            timeout_seconds = config.max_poll_attempts * config.poll_interval
            log_error(
                "Runway video generation timed out",
                context={
                    "provider": config.provider,
                    "model": config.model,
                    "request_id": request_id,
                    "poll_attempts": poll_attempts,
                    "max_attempts": config.max_poll_attempts,
                    "timeout_seconds": timeout_seconds,
                },
                logger_name=_LOGGER_NAME,
            )
            raise TimeoutError(
                f"Video generation timed out after {config.max_poll_attempts} attempts ({timeout_seconds}s)",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response={"status": "timeout", "poll_attempts": poll_attempts},
                timeout_seconds=timeout_seconds,
            )

        return task

    @handle_video_generation_errors
    async def generate_video_async(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """Generate video asynchronously via Runway ML."""
        client = self._get_client(config, "async")
        endpoint, params = self._convert_request(config, request)

        log_debug(
            "Starting API call",
            context={
                "provider": config.provider,
                "model": config.model,
                "endpoint": endpoint,
            },
            logger_name=_LOGGER_NAME,
        )

        try:
            task = await self._call_endpoint(client, endpoint, params)
            request_id = task.id

            log_debug(
                "Request submitted",
                context={
                    "provider": config.provider,
                    "model": config.model,
                    "request_id": request_id,
                    "endpoint": endpoint,
                },
                logger_name=_LOGGER_NAME,
            )

            task = await self._poll_until_complete(
                client, task, config, on_progress, is_async=True
            )

            log_debug(
                "Request complete",
                context={
                    "provider": config.provider,
                    "model": config.model,
                    "request_id": request_id,
                    "task_status": task.status,
                },
                logger_name=_LOGGER_NAME,
                redact=True,
            )

            response = self._convert_response(config, request, request_id, task)

            log_info(
                "Final generated response",
                context={
                    "provider": config.provider,
                    "model": config.model,
                    "request_id": request_id,
                    "response": response,
                },
                logger_name=_LOGGER_NAME,
                redact=True,
            )

            return response

        except Exception as ex:
            raise self._handle_error(config, request, getattr(task, "id", None), ex)

    @handle_video_generation_errors
    def generate_video(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """Generate video synchronously (blocking)."""
        client = self._get_client(config, "sync")
        endpoint, params = self._convert_request(config, request)

        log_debug(
            "Starting API call",
            context={
                "provider": config.provider,
                "model": config.model,
                "endpoint": endpoint,
            },
            logger_name=_LOGGER_NAME,
        )

        try:
            task = self._call_endpoint(client, endpoint, params)
            request_id = task.id

            log_debug(
                "Request submitted",
                context={
                    "provider": config.provider,
                    "model": config.model,
                    "request_id": request_id,
                    "endpoint": endpoint,
                },
                logger_name=_LOGGER_NAME,
            )

            # Use asyncio.run for the unified polling (it handles sync correctly)
            task = asyncio.run(
                self._poll_until_complete(
                    client, task.id, config, on_progress, is_async=False
                )
            )

            log_debug(
                "Request complete",
                context={
                    "provider": config.provider,
                    "model": config.model,
                    "request_id": request_id,
                    "task_status": task.status,
                },
                logger_name=_LOGGER_NAME,
                redact=True,
            )

            response = self._convert_response(config, request, request_id, task)

            log_info(
                "Final generated response",
                context={
                    "provider": config.provider,
                    "model": config.model,
                    "request_id": request_id,
                    "response": response,
                },
                logger_name=_LOGGER_NAME,
                redact=True,
            )

            return response

        except Exception as ex:
            raise self._handle_error(config, request, getattr(task, "id", None), ex)
