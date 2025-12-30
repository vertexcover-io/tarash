"""Veo3 provider handler using google-genai."""

import asyncio
import time
import traceback
from typing import TYPE_CHECKING, Any, Literal

from google.genai.types import GenerateVideosOperation, VideoGenerationReferenceType
from typing_extensions import TypedDict

try:
    from google import genai
    from google.genai.client import AsyncClient, Client
    from google.genai.types import GenerateVideosConfig
except ImportError:
    genai = None  # type: ignore

from tarash.tarash_gateway.video.exceptions import (
    ProviderAPIError,
    ValidationError,
    VideoGenerationError,
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


class Veo3VideoParams(TypedDict, total=False):
    """Veo3-specific parameters."""

    person_generation: Literal["allow_all", "dont_allow", "allow_adult"]


def _convert_to_image(image: MediaType) -> dict[str, Any]:
    """Parse last frame image."""
    if isinstance(image, dict) and "content" in image and "content_type" in image:
        return {"image_bytes": image["content"], "mime_type": image["content_type"]}
    elif isinstance(image, str):
        return {"gcs_uri": image}


def _convert_to_video(video: MediaType) -> dict[str, Any]:
    """Convert video to google-genai format."""
    if isinstance(video, dict) and "content" in video and "content_type" in video:
        return {"video_bytes": video["content"], "mime_type": video["content_type"]}
    elif isinstance(video, str):
        return {"uri": video}


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

            if reference_count > 1:
                raise ValidationError(
                    f"Veo3 only supports 1 reference/first_frame image, got {reference_count}",
                    provider=config.provider,
                )
            if last_frame_count > 1:
                raise ValidationError(
                    f"Veo3 only supports 1 last_frame image, got {last_frame_count}",
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

        return {
            "prompt": prompt,
            "image": image,
            "video": video,
            "config": generation_config,
        }

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
            raise ProviderAPIError(
                "Operation is not completed",
                provider=config.provider,
                raw_response={"operation": str(operation)},
            )

        if operation.error:
            raise VideoGenerationError(
                f"Video generation failed: {operation.error}",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response=operation.model_dump(),
            )

        # Extract video from operation.response
        if not operation.response:
            raise ProviderAPIError(
                "No response in completed operation",
                provider=config.provider,
                raw_response={"operation": str(operation)},
            )

        generated_videos = operation.response.generated_videos
        if not generated_videos or len(generated_videos) == 0:
            raise ProviderAPIError(
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
    ) -> VideoGenerationError:
        """Handle errors from google-genai API."""
        if isinstance(ex, VideoGenerationError):
            return ex

        # Check for google-genai specific errors
        error_type = type(ex).__name__
        error_msg = str(ex)

        return VideoGenerationError(
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
        operation = await client.models.generate_videos(
            model=config.model,
            **veo3_kwargs,
        )

        request_id = operation.name or "unknown"

        try:
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

                end_time = time.time()
                if end_time - start < config.poll_interval:
                    await asyncio.sleep(config.poll_interval - (end_time - start))

                # Get updated operation status
                operation = await client.operations.get(operation)
                poll_attempts += 1

            if not operation.done:
                raise VideoGenerationError(
                    f"Video generation timed out after {config.max_poll_attempts} attempts",
                    provider=config.provider,
                    model=config.model,
                    request_id=request_id,
                    raw_response={"status": "timeout"},
                )

            # Convert final response
            response = self._convert_response(config, request, request_id, operation)
            return response

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
        operation = client.models.generate_videos(
            model=config.model,
            **veo3_kwargs,
        )

        request_id = operation.name or "unknown"

        try:
            # Poll for completion
            poll_attempts = 0
            while poll_attempts < config.max_poll_attempts:
                # Parse and report progress
                if on_progress:
                    update = parse_veo3_operation(operation)
                    on_progress(update)

                # Check if done
                if operation.done:
                    break

                # Wait before next poll
                time.sleep(config.poll_interval)

                # Get updated operation status
                operation = client.operations.get(operation)
                poll_attempts += 1

            if not operation.done:
                raise VideoGenerationError(
                    f"Video generation timed out after {config.max_poll_attempts} attempts",
                    provider=config.provider,
                    model=config.model,
                    request_id=request_id,
                    raw_response={"status": "timeout"},
                )

            # Convert final response
            response = self._convert_response(config, request, request_id, operation)
            return response

        except Exception as ex:
            raise self._handle_error(config, request, request_id, ex)
