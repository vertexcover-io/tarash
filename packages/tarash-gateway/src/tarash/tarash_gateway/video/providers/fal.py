"""Fal.ai provider handler."""

import asyncio
import traceback
from typing import Any, NotRequired

from fal_client.client import FalClientHTTPError

from tarash.tarash_gateway.video.exceptions import (
    ProviderAPIError,
    VideoGenerationError,
    handle_video_generation_errors,
)
from tarash.tarash_gateway.video.models import (
    BaseVideoParams,
    ProgressCallback,
    VideoGenerationConfig,
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoGenerationUpdate,
)
from tarash.tarash_gateway.video.utils import (
    convert_to_data_url,
    validate_model_params,
)

try:
    import fal_client
    from fal_client import (
        AsyncClient,
        Status,
        SyncClient,
        Completed,
        Queued,
        InProgress,
    )
except ImportError:
    pass


# Define Fal-specific params schema
class FalVideoParams(BaseVideoParams):
    """Fal-specific parameters."""

    auto_fix: NotRequired["bool"]


def parse_fal_status(request_id: str, status: Status) -> VideoGenerationUpdate:
    if isinstance(status, Completed):
        return VideoGenerationUpdate(
            request_id=request_id,
            status="completed",
            update={
                "metrics": status.metrics,
                "logs": status.logs,
            },
        )
    elif isinstance(status, Queued):
        return VideoGenerationUpdate(
            request_id=request_id,
            status="queued",
            update={"position": status.position},
        )
    elif isinstance(status, InProgress):
        return VideoGenerationUpdate(
            request_id=request_id,
            status="processing",
            update={"logs": status.logs},
        )

    else:
        raise ValueError(f"Unknown status: {status}")


class FalProviderHandler:
    """Handler for Fal.ai provider."""

    def __init__(self):
        """Initialize handler (stateless, no config stored)."""

        try:
            import fal_client  # noqa: F401
        except ImportError:
            raise ImportError(
                "fal-client is required for Fal provider. "
                "Install with: pip install tarash-gateway[fal]"
            )

        self._sync_client_cache: dict[str, Any] = {}
        self._async_client_cache: dict[str, Any] = {}

    def _get_client(
        self, config: VideoGenerationConfig, client_type: str
    ) -> AsyncClient | SyncClient:
        """
        Get or create Fal client for the given config.

        Args:
            config: Provider configuration
            client_type: Type of client to return ("sync" or "async")

        Returns:
            fal_client.SyncClient or fal_client.AsyncClient instance
        """

        # Use API key + base_url as cache key
        cache_key = f"{config.api_key}:{config.base_url or 'default'}"

        if client_type == "async":
            if cache_key not in self._async_client_cache:
                self._async_client_cache[cache_key] = fal_client.AsyncClient(
                    key=config.api_key,
                    default_timeout=config.timeout,
                )
            return self._async_client_cache[cache_key]
        else:  # sync
            if cache_key not in self._sync_client_cache:
                self._sync_client_cache[cache_key] = fal_client.SyncClient(
                    key=config.api_key,
                    default_timeout=config.timeout,
                )
            return self._sync_client_cache[cache_key]

    def _validate_params(
        self, config: VideoGenerationConfig, request: VideoGenerationRequest
    ) -> dict[str, Any]:
        """
        Validate model_params using Fal-specific schema.

        Args:
            config: Provider configuration
            request: Video generation request

        Returns:
            Validated model parameters dict
        """
        if not request.model_params:
            return {}

        # Use FalVideoParams schema for validation
        return validate_model_params(
            schema=FalVideoParams,
            data=request.model_params,
            provider=config.provider,
            model=config.model,
        )

    def _convert_request(
        self, config: VideoGenerationConfig, request: VideoGenerationRequest
    ) -> dict[str, Any]:
        """
        Convert VideoGenerationRequest to Fal API format.

        Args:
            config: Provider configuration
            request: Video generation request

        Returns:
            Fal-specific request payload
        """
        # Start with validated model_params
        fal_input: dict[str, Any] = self._validate_params(config, request)

        # Add common parameters
        fal_input["prompt"] = request.prompt

        if request.duration_seconds is not None:
            fal_input["duration"] = request.duration_seconds

        if request.resolution is not None:
            fal_input["resolution"] = request.resolution

        if request.aspect_ratio is not None:
            fal_input["aspect_ratio"] = request.aspect_ratio

        if request.image_list:
            # Convert ImageType list to image URLs
            # ImageType is TypedDict with 'image' (MediaType) and 'type' fields
            image_urls = []
            for img in request.image_list:
                media = img["image"]
                # Check if it's a MediaContent dict (TypedDict)
                if (
                    isinstance(media, dict)
                    and "content" in media
                    and "content_type" in media
                ):
                    # Convert to base64 data URL
                    image_urls.append(convert_to_data_url(media))
                else:
                    # String URL - pass through as-is
                    image_urls.append(str(media))
            fal_input["image_urls"] = image_urls

        if request.video:
            # Check if it's a MediaContent dict (TypedDict)
            if (
                isinstance(request.video, dict)
                and "content" in request.video
                and "content_type" in request.video
            ):
                # Convert to base64 data URL
                fal_input["video_url"] = convert_to_data_url(request.video)
            else:
                # String URL - pass through as-is
                fal_input["video_url"] = str(request.video)

        return fal_input

    def _convert_response(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        request_id: str,
        provider_response: Any,
    ) -> VideoGenerationResponse:
        """
        Convert Fal response to VideoGenerationResponse.

        Args:
            config: Provider configuration
            request: Original video generation request
            request_id: Our request ID
            provider_response: Raw Fal response

        Returns:
            Normalized VideoGenerationResponse
        """
        # Extract video URL from Fal response
        # Fal returns: {"video": {"url": "..."}, ...} or {"video_url": "..."}
        video_url = None
        audio_url = None

        if isinstance(provider_response, dict):
            # Try different response formats
            if "video" in provider_response and isinstance(
                provider_response["video"], dict
            ):
                video_url = provider_response["video"].get("url")
            elif "video_url" in provider_response:
                video_url = provider_response["video_url"]

            if "audio" in provider_response and isinstance(
                provider_response["audio"], dict
            ):
                audio_url = provider_response["audio"].get("url")
            elif "audio_url" in provider_response:
                audio_url = provider_response["audio_url"]

        if not video_url:
            raise ProviderAPIError(
                f"No video URL found in Fal response: {provider_response}",
                provider=config.provider,
                raw_response=provider_response
                if isinstance(provider_response, dict)
                else {},
            )

        return VideoGenerationResponse(
            request_id=request_id,
            video=video_url,
            audio_url=audio_url,
            duration=provider_response.get("duration")
            if isinstance(provider_response, dict)
            else None,
            resolution=provider_response.get("resolution")
            if isinstance(provider_response, dict)
            else None,
            aspect_ratio=provider_response.get("aspect_ratio")
            if isinstance(provider_response, dict)
            else None,
            status="completed",
            raw_response=provider_response
            if isinstance(provider_response, dict)
            else {"data": str(provider_response)},
            provider_metadata={},
        )

    def _handle_error(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        request_id: str,
        ex: Exception,
    ) -> VideoGenerationResponse:
        if isinstance(ex, VideoGenerationError):
            return ex

        elif isinstance(ex, FalClientHTTPError):
            return VideoGenerationError(
                f"Request Failed with status code {ex.status_code}: {ex.message}",
                provider=config.provider,
                raw_response={
                    "status_code": ex.status_code,
                    "response_headers": ex.response_headers,
                    "response": ex.response.content,
                    "traceback": traceback.format_exc(),
                },
            )
        else:
            raise VideoGenerationError(
                f"Unknown Error while generating video: {ex}",
                provider=config.provider,
                raw_response={
                    "error": str(ex),
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
        Generate video asynchronously via Fal with async progress callback.

        Args:
            config: Provider configuration
            request: Video generation request
            on_progress: Optional async callback for progress updates

        Returns:
            Final VideoGenerationResponse when complete
        """
        client = self._get_client(config, "async")
        # Build Fal input (let validation errors propagate)
        fal_input = self._convert_request(config, request)

        # Submit to Fal using async API
        handler = await client.submit(
            config.model,
            arguments=fal_input,
        )

        request_id = handler.request_id

        try:
            async for event in handler.iter_events(with_logs=True):
                if on_progress:
                    result = on_progress(parse_fal_status(request_id, event))
                    if asyncio.iscoroutine(result):
                        await result

            result = await handler.get()

            # Parse response
            fal_result = (
                result
                if isinstance(result, dict)
                else (result.data if hasattr(result, "data") else {})
            )
            response = self._convert_response(config, request, request_id, fal_result)

            return response

        except Exception as ex:
            raise self._handle_error(config, request, request_id, ex)

    @handle_video_generation_errors
    def generate_video(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: Any | None = None,
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

        # Build Fal input (let validation errors propagate)
        fal_input = self._convert_request(config, request)

        # Submit to Fal
        handler = client.submit(
            config.model,
            arguments=fal_input,
        )
        request_id = handler.request_id

        try:
            for event in handler.iter_events(with_logs=True):
                if on_progress:
                    on_progress(parse_fal_status(request_id, event))

            result = handler.get()

            # Parse response
            fal_result = (
                result
                if isinstance(result, dict)
                else (result.data if hasattr(result, "data") else {})
            )
            response = self._convert_response(config, request, request_id, fal_result)

            return response

        except Exception as ex:
            raise self._handle_error(config, request, request_id, ex)
