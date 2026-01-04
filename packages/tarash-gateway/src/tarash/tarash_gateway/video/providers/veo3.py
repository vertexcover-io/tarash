"""Veo3 provider handler using google-genai."""

import asyncio
import time
import traceback
from typing import TYPE_CHECKING, Any, Literal

from google.genai.types import GenerateVideosOperation, VideoGenerationReferenceType
from typing_extensions import TypedDict

try:
    import httpx
    from google import genai
    from google.genai.client import AsyncClient, Client
    from google.genai.errors import ClientError
    from google.genai.types import GenerateVideosConfig

    # google.genai can use either httpx or aiohttp for async
    try:
        import aiohttp

        has_aiohttp = True
    except ImportError:
        aiohttp = None
        has_aiohttp = False
except ImportError:
    genai = None
    has_aiohttp = False


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
from tarash.tarash_gateway.video.utils import validate_model_params

if TYPE_CHECKING:
    from google.genai.client import AsyncClient, Client
    from google.genai.types import GenerateVideosConfig

# Logger name constant
_LOGGER_NAME = "tarash.tarash_gateway.video.providers.veo3"


class Veo3VideoParams(TypedDict, total=False):
    """Veo3-specific parameters."""

    person_generation: Literal["allow_all", "dont_allow", "allow_adult"]


def _convert_to_image(image: MediaType) -> dict[str, Any]:
    """Parse last frame image."""
    if isinstance(image, dict) and "content" in image and "content_type" in image:
        return {"image_bytes": image["content"], "mime_type": image["content_type"]}
    elif isinstance(image, str):
        return {"gcs_uri": image}
    return {}  # type: ignore[return-value]


def _convert_to_video(video: MediaType) -> dict[str, Any]:
    """Convert video to google-genai format."""
    if isinstance(video, dict) and "content" in video and "content_type" in video:
        return {"video_bytes": video["content"], "mime_type": video["content_type"]}
    elif isinstance(video, str):
        return {"uri": video}
    return {}  # type: ignore[return-value]


def parse_veo3_operation(operation: GenerateVideosOperation) -> VideoGenerationUpdate:
    """Parse google-genai operation to VideoGenerationUpdate."""
    # Check operation metadata for status
    status_str = "processing"
    metadata = operation.metadata
    if metadata and "state" in metadata:
        state = metadata["state"]
        if state in ["PENDING", "QUEUED"]:
            status_str = "queued"
        elif state in ["RUNNING", "PROCESSING"]:
            status_str = "processing"
    return VideoGenerationUpdate(
        request_id=operation.name or "unknown",
        status=status_str,
        update={"metadata": metadata},
    )


class Veo3ProviderHandler:
    """Handler for Veo3 provider using google-genai."""

    def __init__(self):
        """Initialize handler (stateless, no config stored)."""
        self._sync_client_cache: dict[str, Any] = {}
        self._async_client_cache: dict[str, Any] = {}
        try:
            import google.genai  # noqa: F401
        except ImportError:
            raise ImportError(
                "google-genai is required for Veo3 provider. "
                "Install with: pip install tarash-gateway[veo3]"
            )

    def _get_client(
        self, config: VideoGenerationConfig, client_type: str
    ) -> AsyncClient | Client:
        """
        Get or create google-genai client for the given config.

        Args:
            config: Provider configuration
            client_type: Type of client to return ("sync" or "async")

        Returns:
            genai.Client instance (sync) or genai.Client.aio (async)
        """
        if genai is None:
            raise ImportError(
                "google-genai is required for Veo3 provider. "
                "Install with: pip install tarash-gateway[veo3]"
            )

        # Use API key + base_url as cache key
        # base_url can be used to determine if using Vertex AI
        base_url = config.base_url
        use_vertex = base_url is not None and "vertex" in base_url.lower()

        # Extract project and location from base_url if using Vertex AI
        project = None
        location = None
        if use_vertex and base_url:
            # Try to parse project and location from base_url
            # Format: https://{location}-aiplatform.googleapis.com or similar
            import re

            match = re.search(r"([a-z0-9-]+)-aiplatform", base_url)
            if match:
                location = match.group(1)
            # For project, we might need to extract from a different format
            # For now, we'll use api_key as project if it looks like a project ID

        cache_key = f"{config.api_key}:{base_url or 'default'}:{client_type}"

        if client_type == "async":
            if cache_key not in self._async_client_cache:
                # Create a Client and access its .aio property for async operations
                client = Client(
                    api_key=config.api_key,
                    vertexai=use_vertex,
                    project=project,
                    location=location,
                )
                self._async_client_cache[cache_key] = client.aio
            return self._async_client_cache[cache_key]
        else:  # sync
            if cache_key not in self._sync_client_cache:
                self._sync_client_cache[cache_key] = Client(
                    api_key=config.api_key,
                    vertexai=use_vertex,
                    project=project,
                    location=location,
                )
            return self._sync_client_cache[cache_key]

    def _validate_params(
        self, config: VideoGenerationConfig, request: VideoGenerationRequest
    ) -> dict[str, Any]:
        """
        Validate model_params using Veo3-specific schema.

        Args:
            config: Provider configuration
            request: Video generation request

        Returns:
            Validated model parameters dict
        """
        if not request.extra_params:
            return {}

        # Use Veo3VideoParams schema for validation
        return validate_model_params(
            schema=Veo3VideoParams,
            data=request.extra_params,
            provider=config.provider,
            model=config.model,
        )

    def _convert_request(
        self, config: VideoGenerationConfig, request: VideoGenerationRequest
    ) -> dict[str, Any]:
        """
        Convert VideoGenerationRequest to google-genai format.

        Args:
            config: Provider configuration
            request: Video generation request

        Returns:
            Dict with prompt, image, video, and config keys ready for generate_videos(**kwargs)

        Raises:
            ValidationError: If more than 1 reference/first_frame or last_frame image is provided
        """
        # Start with validated model_params
        model_params = self._validate_params(config, request)

        # Build GenerateVideosConfig params
        config_params: dict[str, Any] = {
            **model_params,
        }

        # Separate parameters for generate_videos call
        prompt = request.prompt
        image = None
        video = None

        # Duration: convert to duration_seconds if it's a number or parse from string
        if request.duration_seconds is not None:
            config_params["duration_seconds"] = int(request.duration_seconds)

        # Aspect ratio
        if request.aspect_ratio is not None:
            config_params["aspect_ratio"] = request.aspect_ratio

        if request.number_of_videos is not None:
            config_params["number_of_videos"] = int(request.number_of_videos)

        if request.generate_audio is not None:
            config_params["generate_audio"] = request.generate_audio

        if request.seed is not None:
            config_params["seed"] = int(request.seed)

        if request.negative_prompt is not None:
            config_params["negative_prompt"] = request.negative_prompt

        if request.enhance_prompt is not None:
            config_params["enhance_prompt"] = request.enhance_prompt

        if request.video is not None:
            video = _convert_to_video(request.video)

        # Validate image constraints before processing
        if request.image_list:
            reference_count = sum(
                1
                for img in request.image_list
                if img["type"] in ("first_frame", "reference")
            )
            last_frame_count = sum(
                1 for img in request.image_list if img["type"] == "last_frame"
            )
            reference_images_count = sum(
                1 for img in request.image_list if img["type"] in ("asset", "style")
            )
            has_first_frame = any(
                img["type"] == "first_frame" for img in request.image_list
            )

            # Validate only 1 reference/first_frame image
            if reference_count > 1:
                raise ValidationError(
                    f"Veo3 only supports 1 reference/first_frame image, got {reference_count}",
                    provider=config.provider,
                )

            # Validate only 1 last_frame image
            if last_frame_count > 1:
                raise ValidationError(
                    f"Veo3 only supports 1 last_frame image, got {last_frame_count}",
                    provider=config.provider,
                )

            # Validate last_frame requires first_frame
            if last_frame_count > 0 and not has_first_frame:
                raise ValidationError(
                    "Veo3 requires first_frame when using last_frame (interpolation mode)",
                    provider=config.provider,
                )

            # Validate first_frame and reference images are mutually exclusive
            if has_first_frame and reference_images_count > 0:
                raise ValidationError(
                    "Veo3 does not allow reference images (asset/style) when using first_frame (interpolation mode)",
                    provider=config.provider,
                )

            # Validate maximum 3 reference images
            if reference_images_count > 3:
                raise ValidationError(
                    f"Veo3 only supports up to 3 reference images (asset/style), got {reference_images_count}",
                    provider=config.provider,
                )

        for image_item in request.image_list or []:
            match image_item["type"]:
                case "first_frame" | "reference":
                    image = _convert_to_image(image_item["image"])
                case "last_frame":
                    config_params["last_frame"] = _convert_to_image(image_item["image"])
                case "asset":
                    if "reference_images" not in config_params:
                        config_params["reference_images"] = []

                    config_params["reference_images"].append(
                        {
                            "image": _convert_to_image(image_item["image"]),
                            "reference_type": VideoGenerationReferenceType.ASSET,
                        }
                    )
                case "style":
                    if "reference_images" not in config_params:
                        config_params["reference_images"] = []

                    config_params["reference_images"].append(
                        {
                            "image": _convert_to_image(image_item["image"]),
                            "reference_type": VideoGenerationReferenceType.STYLE,
                        }
                    )

        # Create GenerateVideosConfig and return kwargs dict
        generation_config = GenerateVideosConfig(**config_params)

        api_payload = {
            "prompt": prompt,
            "image": image,
            "video": video,
            "config": generation_config,
        }

        log_info(
            "Mapped request to provider format",
            context={
                "provider": config.provider,
                "model": config.model,
                "converted_request": api_payload,
            },
            logger_name=_LOGGER_NAME,
            redact=True,
        )

        return api_payload

    def _convert_response(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        request_id: str,
        operation: GenerateVideosOperation,
    ) -> VideoGenerationResponse:
        """
        Convert google-genai operation response to VideoGenerationResponse.

        Args:
            config: Provider configuration
            request: Original video generation request
            request_id: Our request ID (operation name)
            operation: Completed operation from google-genai

        Returns:
            Normalized VideoGenerationResponse
        """
        if not operation.done:
            raise GenerationFailedError(
                "Operation is not completed",
                provider=config.provider,
                raw_response={"operation": str(operation)},
            )

        if operation.error:
            raise GenerationFailedError(
                f"Video generation failed: {operation.error}",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response=operation.model_dump(),
            )

        # Extract video from operation.response
        if not operation.response:
            raise GenerationFailedError(
                "No response in completed operation",
                provider=config.provider,
                raw_response={"operation": str(operation)},
            )

        generated_videos = operation.response.generated_videos
        if not generated_videos or len(generated_videos) == 0:
            raise GenerationFailedError(
                "No generated videos in response",
                provider=config.provider,
                raw_response={"operation": str(operation)},
            )

        video_obj = generated_videos[0].video
        if video_obj.uri:
            video = video_obj.uri
        else:
            video = {
                "content": video_obj.video_bytes,
                "content_type": video_obj.mime_type,
            }

        return VideoGenerationResponse(
            request_id=request_id,
            video=video,
            content_type=video_obj.mime_type,
            status="completed",
            raw_response=operation.model_dump(),
            provider_metadata={},
        )

    def _handle_error(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        request_id: str,
        ex: Exception,
    ) -> TarashException:
        """Handle errors from google-genai API."""
        if isinstance(ex, TarashException):
            return ex

        # httpx timeout errors
        if isinstance(ex, httpx.TimeoutException):
            return TimeoutError(
                f"Request timed out: {str(ex)}",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response={"error": str(ex)},
                timeout_seconds=config.timeout,
            )

        # httpx connection errors
        if isinstance(ex, (httpx.ConnectError, httpx.NetworkError)):
            return HTTPConnectionError(
                f"Connection error: {str(ex)}",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response={"error": str(ex)},
            )

        # aiohttp errors (if aiohttp is used instead of httpx)
        if has_aiohttp and aiohttp:
            # aiohttp timeout errors
            if isinstance(ex, (aiohttp.ServerTimeoutError, aiohttp.ClientTimeout)):
                return TimeoutError(
                    f"Request timed out: {str(ex)}",
                    provider=config.provider,
                    model=config.model,
                    request_id=request_id,
                    raw_response={"error": str(ex)},
                    timeout_seconds=config.timeout,
                )

            # aiohttp connection errors
            if isinstance(
                ex, (aiohttp.ClientConnectionError, aiohttp.ClientConnectorError)
            ):
                return HTTPConnectionError(
                    f"Connection error: {str(ex)}",
                    provider=config.provider,
                    model=config.model,
                    request_id=request_id,
                    raw_response={"error": str(ex)},
                )

        # Google GenAI ClientError (includes validation and HTTP errors)
        if isinstance(ex, ClientError):
            raw_response = {
                "status_code": ex.code,
                "message": ex.message,
                "error_details": ex.details,
                "error_type": ex.status,
            }

            # Validation errors (400, 422)
            if ex.code in (400, 422):
                return ValidationError(
                    ex.message,
                    provider=config.provider,
                    model=config.model,
                    request_id=request_id,
                    raw_response=raw_response,
                )

            # All other HTTP errors
            return HTTPError(
                ex.message,
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response=raw_response,
                status_code=ex.code,
            )

        # Unknown errors
        log_error(
            f"Veo3 unknown error: {str(ex)}",
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

    @handle_video_generation_errors
    async def generate_video_async(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """
        Generate video asynchronously via Veo3 with async progress callback.

        Args:
            config: Provider configuration
            request: Video generation request
            on_progress: Optional async callback for progress updates

        Returns:
            Final VideoGenerationResponse when complete
        """
        client = self._get_client(config, "async")
        # Build Veo3 input (let validation errors propagate)
        veo3_kwargs = self._convert_request(config, request)

        log_debug(
            "Starting API call",
            context={
                "provider": config.provider,
                "model": config.model,
            },
            logger_name=_LOGGER_NAME,
        )

        try:
            request_id = None
            operation = await client.models.generate_videos(
                model=config.model,
                **veo3_kwargs,
            )

            request_id = operation.name or "unknown"

            log_debug(
                "Request submitted",
                context={
                    "provider": config.provider,
                    "model": config.model,
                    "request_id": request_id,
                },
                logger_name=_LOGGER_NAME,
            )

            # Poll for completion
            poll_attempts = 0
            while poll_attempts < config.max_poll_attempts:
                if operation.done:
                    break

                start = time.time()
                if on_progress:
                    update = parse_veo3_operation(operation)
                    result = on_progress(update)
                    if asyncio.iscoroutine(result):
                        await result

                # Log progress
                log_info(
                    "Progress status update",
                    context={
                        "provider": config.provider,
                        "model": config.model,
                        "request_id": request_id,
                        "status": "processing" if not operation.done else "completed",
                    },
                    logger_name=_LOGGER_NAME,
                )

                end_time = time.time()
                if end_time - start < config.poll_interval:
                    await asyncio.sleep(config.poll_interval - (end_time - start))

                # Get updated operation status
                operation = await client.operations.get(operation)
                poll_attempts += 1

            if not operation.done:
                timeout_seconds = config.max_poll_attempts * config.poll_interval
                raise TimeoutError(
                    f"Video generation timed out after {config.max_poll_attempts} attempts ({timeout_seconds}s)",
                    provider=config.provider,
                    model=config.model,
                    request_id=request_id,
                    raw_response={
                        "status": "timeout",
                        "poll_attempts": config.max_poll_attempts,
                    },
                    timeout_seconds=timeout_seconds,
                )

            log_debug(
                "Request complete",
                context={
                    "provider": config.provider,
                    "model": config.model,
                    "request_id": request_id,
                    "response": operation,
                },
                logger_name=_LOGGER_NAME,
                redact=True,
            )

            # Convert final response
            response = self._convert_response(config, request, request_id, operation)

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

        # Build Veo3 input (let validation errors propagate)
        veo3_kwargs = self._convert_request(config, request)

        log_debug(
            "Starting API call",
            context={
                "provider": config.provider,
                "model": config.model,
            },
            logger_name=_LOGGER_NAME,
        )

        try:
            request_id = None
            operation = client.models.generate_videos(
                model=config.model,
                **veo3_kwargs,
            )

            request_id = operation.name or "unknown"

            log_debug(
                "Request submitted",
                context={
                    "provider": config.provider,
                    "model": config.model,
                    "request_id": request_id,
                },
                logger_name=_LOGGER_NAME,
            )

            # Poll for completion
            poll_attempts = 0
            while poll_attempts < config.max_poll_attempts:
                # Parse and report progress
                if on_progress:
                    update = parse_veo3_operation(operation)
                    on_progress(update)

                # Log progress
                log_info(
                    "Progress status update",
                    context={
                        "provider": config.provider,
                        "model": config.model,
                        "request_id": request_id,
                        "status": "processing" if not operation.done else "completed",
                    },
                    logger_name=_LOGGER_NAME,
                )

                # Check if done
                if operation.done:
                    break

                # Wait before next poll
                time.sleep(config.poll_interval)

                # Get updated operation status
                operation = client.operations.get(operation)
                poll_attempts += 1

            if not operation.done:
                timeout_seconds = config.max_poll_attempts * config.poll_interval
                raise TimeoutError(
                    f"Video generation timed out after {config.max_poll_attempts} attempts ({timeout_seconds}s)",
                    provider=config.provider,
                    model=config.model,
                    request_id=request_id,
                    raw_response={
                        "status": "timeout",
                        "poll_attempts": config.max_poll_attempts,
                    },
                    timeout_seconds=timeout_seconds,
                )

            log_debug(
                "Request complete",
                context={
                    "provider": config.provider,
                    "model": config.model,
                    "request_id": request_id,
                    "response": operation,
                },
                logger_name=_LOGGER_NAME,
                redact=True,
            )

            # Convert final response
            response = self._convert_response(config, request, request_id, operation)

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
            raise self._handle_error(config, request, request_id, ex) from ex
