"""Runway ML provider handler for video generation."""

import asyncio
import io
import time
from typing import TYPE_CHECKING, Any, Literal, cast, overload

from typing_extensions import TypedDict

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
    AnyDict,
    MediaType,
    ProgressCallback,
    VideoGenerationConfig,
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoGenerationUpdate,
)
from tarash.tarash_gateway.utils import validate_duration, validate_model_params

try:
    from runwayml import AsyncRunwayML, RunwayML
    from runwayml.lib.polling import (
        AsyncAwaitableTaskRetrieveResponse,
        AsyncNewTaskCreatedResponse,
        NewTaskCreatedResponse,
    )  # pyright: ignore[reportMissingTypeStubs]

    # Import specific exception types from runwayml SDK
    from runwayml import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        BadRequestError,
        UnprocessableEntityError,
    )

    has_runwayml = True
except ImportError:
    has_runwayml = False

if TYPE_CHECKING:
    from runwayml import AsyncRunwayML, RunwayML
    from runwayml.lib.polling import (
        AsyncAwaitableTaskRetrieveResponse,
        AsyncNewTaskCreatedResponse,
        NewTaskCreatedResponse,
    )  # pyright: ignore[reportMissingTypeStubs]
    from runwayml import (
        APIConnectionError,
        APIStatusError,
        APITimeoutError,
        BadRequestError,
        UnprocessableEntityError,
    )
else:
    # Runtime fallbacks for when runwayml is not installed
    if not has_runwayml:
        RunwayML = object  # type: ignore[assignment, misc]
        AsyncRunwayML = object  # type: ignore[assignment, misc]
        AsyncAwaitableTaskRetrieveResponse = object  # type: ignore[assignment, misc]
        AsyncNewTaskCreatedResponse = object  # type: ignore[assignment, misc]
        NewTaskCreatedResponse = object  # type: ignore[assignment, misc]
        APIConnectionError = Exception  # type: ignore[assignment, misc]
        APIStatusError = Exception  # type: ignore[assignment, misc]
        APITimeoutError = Exception  # type: ignore[assignment, misc]
        BadRequestError = Exception  # type: ignore[assignment, misc]
        UnprocessableEntityError = Exception  # type: ignore[assignment, misc]

# Logger name constant
_LOGGER_NAME = "tarash.tarash_gateway.providers.runway"

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

    # MediaContent dict with bytes (type guard for dict access)
    if isinstance(media, dict):
        content_bytes = media["content"]
        content_type = media["content_type"]
        ext = content_type.split("/")[-1] if "/" in content_type else "bin"
        file_obj = io.BytesIO(content_bytes)
        file_obj.name = f"{default_name}.{ext}"
        return file_obj

    # Fallback for HttpUrl type (though it should be caught by isinstance(media, str))
    return str(media)


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


def _extract_video_url(output: object) -> str | None:
    """Extract video URL from task output.

    Args:
        output: Task output (type not exported by SDK, can be list/str/dict)

    Returns:
        Video URL string or None if not found
    """
    if isinstance(output, list):
        return cast(str, output[0]) if output else None
    if isinstance(output, str):
        return output
    if isinstance(output, dict):
        url = output.get("url")  # pyright: ignore[reportUnknownMemberType,reportUnknownVariableType]
        return cast(str, url) if url is not None else None
    return None


def parse_runway_task_status(
    task: AsyncAwaitableTaskRetrieveResponse,
) -> VideoGenerationUpdate:
    """Parse Runway task object to VideoGenerationUpdate.

    Args:
        task: Runway task object (Task type not exported by SDK)
    """
    # Map runway status to StatusType literals
    status_map: dict[str, Literal["queued", "processing", "completed", "failed"]] = {
        "PENDING": "queued",
        "THROTTLED": "queued",
        "RUNNING": "processing",
        "SUCCEEDED": "completed",
        "FAILED": "failed",
        "CANCELLED": "failed",
    }

    # Get status with typed default
    mapped_status: Literal["queued", "processing", "completed", "failed"] = (
        status_map.get(task.status, "processing")
    )

    return VideoGenerationUpdate(
        request_id=task.id,
        status=mapped_status,
        progress_percent=None,
        update={"raw_status": task.status},
    )


class RunwayProviderHandler:
    """Handler for Runway ML video generation."""

    def __init__(self):
        """Initialize handler (stateless, no config stored)."""
        if not has_runwayml:
            raise ImportError(
                "runwayml is required for Runway provider. "
                "Install with: pip install tarash-gateway[runway]"
            )

    @overload
    def _get_client(
        self, config: VideoGenerationConfig, client_type: Literal["async"]
    ) -> "AsyncRunwayML": ...

    @overload
    def _get_client(
        self, config: VideoGenerationConfig, client_type: Literal["sync"]
    ) -> "RunwayML": ...

    def _get_client(
        self, config: VideoGenerationConfig, client_type: Literal["sync", "async"]
    ) -> "RunwayML | AsyncRunwayML":
        """Create Runway client for the given config."""
        if not has_runwayml:
            raise ImportError(
                "runwayml is required for Runway provider. "
                "Install with: pip install tarash-gateway[runway]"
            )
        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)
        logger.debug(f"Creating new {client_type} Runway client")
        if client_type == "async":
            return AsyncRunwayML(api_key=config.api_key)
        else:
            return RunwayML(api_key=config.api_key)

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

        # Endpoint-specific parameters
        # Note: seed and content_moderation are only supported by image_to_video
        # and video_to_video endpoints, NOT text_to_video
        if endpoint == "text_to_video":
            # Remove params not supported by text_to_video endpoint
            params.pop("seed", None)
            params.pop("content_moderation", None)
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
            # image_to_video supports seed and content_moderation
            if request.seed is not None:
                params["seed"] = int(request.seed)
            if "content_moderation" in request.extra_params:
                params["content_moderation"] = request.extra_params[
                    "content_moderation"
                ]

        elif endpoint == "video_to_video":
            if not request.video:
                raise ValidationError(
                    "Video input is required for video-to-video",
                    provider=config.provider,
                )
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
            # video_to_video supports seed and content_moderation
            if request.seed is not None:
                params["seed"] = int(request.seed)
            if "content_moderation" in request.extra_params:
                params["content_moderation"] = request.extra_params[
                    "content_moderation"
                ]

        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)
        logger.info(
            "Mapped request to provider format",
            {"endpoint": endpoint, "converted_request": params},
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
        if has_runwayml and isinstance(ex, APITimeoutError):
            # APITimeoutError inherits from APIError which has .message
            error_message = str(ex)
            return TimeoutError(
                f"Request timed out: {error_message}",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response={"error": error_message},
                timeout_seconds=config.timeout,
            )

        # API connection errors
        if has_runwayml and isinstance(ex, APIConnectionError):
            # APIConnectionError inherits from APIError which has .message
            error_message = str(ex)
            return HTTPConnectionError(
                f"Connection error: {error_message}",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response={"error": error_message},
            )

        # API status errors (4xx, 5xx)
        if has_runwayml and isinstance(ex, APIStatusError):
            # APIStatusError has status_code, inherits message from APIError, and has body
            error_message = str(ex)
            # Use getattr to avoid type checking issues with dynamically imported exception types
            status_code = getattr(ex, "status_code", 0)
            raw_response: AnyDict = {
                "status_code": status_code,
                "message": error_message,
                "body": getattr(ex, "body", None),
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
                status_code=status_code,
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
        self, client: "RunwayML | AsyncRunwayML", endpoint: str, params: dict[str, Any]
    ) -> "NewTaskCreatedResponse | AsyncNewTaskCreatedResponse":
        """Call the appropriate Runway endpoint.

        Both sync and async clients have identical API structure, so we use the same
        endpoint_map for both. The difference is in what they return (sync value vs coroutine),
        which is handled at the call site with asyncio.iscoroutine().
        """
        endpoint_map = {
            "text_to_video": lambda: client.text_to_video.create(**params),
            "image_to_video": lambda: client.image_to_video.create(**params),
            "video_to_video": lambda: client.video_to_video.create(**params),
        }
        result = endpoint_map[endpoint]()
        return result  # type: ignore[return-value]

    async def _poll_until_complete(
        self,
        client: "RunwayML | AsyncRunwayML",
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

        task: Any = None  # Will hold the retrieved task object

        while poll_attempts < config.max_poll_attempts:
            # Wait and retrieve updated status
            if is_async:
                await asyncio.sleep(config.poll_interval)
                # AsyncRunwayML client returns awaitable - must await it
                task_response = client.tasks.retrieve(task_id)
                if asyncio.iscoroutine(task_response):
                    task = await task_response
                else:
                    task = task_response
            else:
                time.sleep(config.poll_interval)
                task = client.tasks.retrieve(task_id)

            poll_attempts += 1

            # Report progress
            if on_progress and task is not None:
                update = parse_runway_task_status(task)
                result = on_progress(update)
                if is_async and asyncio.iscoroutine(result):
                    await result

            # Check if terminal
            if task is not None and task.status in _TERMINAL_STATUSES:
                break

            # Log progress
            if task is not None:
                logger = ProviderLogger(
                    config.provider, config.model, _LOGGER_NAME, request_id
                )
                logger.info(
                    "Progress status update",
                    {
                        "status": task.status,
                        "poll_attempt": poll_attempts + 1,
                    },
                )

        # Check timeout
        if task is None or task.status not in _TERMINAL_STATUSES:
            timeout_seconds = config.max_poll_attempts * config.poll_interval
            logger = ProviderLogger(
                config.provider, config.model, _LOGGER_NAME, request_id
            )
            logger.error(
                "Runway video generation timed out",
                {
                    "poll_attempts": poll_attempts,
                    "max_attempts": config.max_poll_attempts,
                    "timeout_seconds": timeout_seconds,
                },
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

        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)
        logger.debug("Starting API call", {"endpoint": endpoint})

        task: "NewTaskCreatedResponse | AsyncNewTaskCreatedResponse | None" = None
        try:
            task_result = self._call_endpoint(client, endpoint, params)
            if asyncio.iscoroutine(task_result):
                task = await task_result
            else:
                task = task_result
            if task is None:
                raise GenerationFailedError(
                    "Failed to create task",
                    provider=config.provider,
                    model=config.model,
                    request_id="unknown",
                )
            request_id = task.id

            logger = logger.with_request_id(request_id)
            logger.debug("Request submitted", {"endpoint": endpoint})

            completed_task = await self._poll_until_complete(
                client, task.id, config, on_progress, is_async=True
            )

            logger.debug(
                "Request complete",
                {
                    "task_status": completed_task.status
                    if completed_task is not None
                    else "unknown",
                },
                redact=True,
            )

            response = self._convert_response(
                config, request, request_id, completed_task
            )

            logger.info("Final generated response", {"response": response}, redact=True)

            return response

        except Exception as ex:
            # Get request_id if task exists, otherwise use placeholder
            error_request_id = getattr(task, "id", None) if task is not None else None
            raise self._handle_error(config, request, error_request_id or "unknown", ex)

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

        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)
        logger.debug("Starting API call", {"endpoint": endpoint})

        task: "NewTaskCreatedResponse | AsyncNewTaskCreatedResponse | None" = None
        try:
            task = self._call_endpoint(client, endpoint, params)
            request_id = task.id

            logger = logger.with_request_id(request_id)
            logger.debug("Request submitted", {"endpoint": endpoint})

            # Use asyncio.run for the unified polling (it handles sync correctly)
            completed_task = asyncio.run(
                self._poll_until_complete(
                    client, task.id, config, on_progress, is_async=False
                )
            )

            logger.debug(
                "Request complete",
                {
                    "task_status": completed_task.status
                    if completed_task is not None
                    else "unknown",
                },
                redact=True,
            )

            if completed_task is None:
                raise GenerationFailedError(
                    "Task completed but result is None",
                    provider=config.provider,
                    model=config.model,
                    request_id=request_id,
                )
            response = self._convert_response(
                config, request, request_id, completed_task
            )

            logger.info("Final generated response", {"response": response}, redact=True)

            return response

        except Exception as ex:
            # Get request_id if task exists, otherwise use placeholder
            error_request_id = getattr(task, "id", None) if task is not None else None
            raise self._handle_error(config, request, error_request_id or "unknown", ex)

    # ==================== Image Generation (Not Supported) ====================

    async def generate_image_async(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """Runway image generation - not supported."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support image generation. "
            "Use Fal provider for image generation."
        )

    def generate_image(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """Runway image generation - not supported."""
        raise NotImplementedError(
            f"{self.__class__.__name__} does not support image generation. "
            "Use Fal provider for image generation."
        )
