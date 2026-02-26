"""xAI provider handler for video and image generation."""

import asyncio
import time
from typing import TYPE_CHECKING, Any, Literal, overload

from tarash.tarash_gateway.logging import ProviderLogger, log_error
from tarash.tarash_gateway.exceptions import (
    ContentModerationError,
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
    ProgressCallback,
    SyncImageProgressCallback,
    VideoGenerationConfig,
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoGenerationUpdate,
)


has_xai_sdk = True
try:
    from xai_sdk import AsyncClient, Client
except ImportError:
    has_xai_sdk = False

if TYPE_CHECKING:
    from xai_sdk import AsyncClient, Client


_LOGGER_NAME = "tarash.tarash_gateway.providers.xai"

_VALID_VIDEO_RESOLUTIONS = frozenset({"720p", "480p"})
_VALID_IMAGE_RESOLUTIONS = frozenset({"1k", "2k"})
_VALID_VIDEO_DURATIONS = range(1, 16)  # 1-15 seconds inclusive

_STATUS_MAP: dict[str, Literal["queued", "processing", "completed", "failed"]] = {
    "pending": "processing",
    "done": "completed",
    "expired": "failed",
}


def parse_xai_video_status(
    request_id: str,
    raw_status: str,
) -> VideoGenerationUpdate:
    """Parse xAI video generation status string into a VideoGenerationUpdate."""
    mapped: Literal["queued", "processing", "completed", "failed"] = _STATUS_MAP.get(
        raw_status.lower(), "processing"
    )
    return VideoGenerationUpdate(
        request_id=request_id,
        status=mapped,
        progress_percent=None,
        update={"raw_status": raw_status},
    )


class XaiProviderHandler:
    """Handler for xAI video and image generation."""

    def __init__(self) -> None:
        """Initialize handler. Raises ImportError if xai-sdk is not installed."""
        if not has_xai_sdk:
            raise ImportError(
                "xai-sdk is required for xAI provider. "
                "Install with: pip install tarash-gateway[xai]"
            )

    @overload
    def _get_client(
        self, api_key: str | None, timeout: int, client_type: Literal["async"]
    ) -> "AsyncClient": ...

    @overload
    def _get_client(
        self, api_key: str | None, timeout: int, client_type: Literal["sync"]
    ) -> "Client": ...

    def _get_client(
        self, api_key: str | None, timeout: int, client_type: Literal["sync", "async"]
    ) -> "Client | AsyncClient":
        """Create an xAI client."""
        logger = ProviderLogger("xai", "xai", _LOGGER_NAME)
        if client_type == "async":
            logger.debug("Creating new async xAI client")
            return AsyncClient(api_key=api_key, timeout=timeout)

        logger.debug("Creating new sync xAI client")
        return Client(api_key=api_key, timeout=timeout)

    def _convert_video_request(
        self, config: VideoGenerationConfig, request: VideoGenerationRequest
    ) -> dict[str, Any]:
        """Convert VideoGenerationRequest to xAI video API parameters."""
        params: dict[str, Any] = {
            "prompt": request.prompt,
            "model": config.model,
        }

        # Duration — integer 1-15
        if request.duration_seconds is not None:
            if request.duration_seconds not in _VALID_VIDEO_DURATIONS:
                raise ValidationError(
                    f"Invalid duration for xAI: {request.duration_seconds}s. "
                    f"Supported: 1-15 seconds.",
                    provider=config.provider,
                )
            params["duration"] = request.duration_seconds

        # Resolution — "720p" or "480p"
        if request.resolution is not None:
            if request.resolution not in _VALID_VIDEO_RESOLUTIONS:
                supported = ", ".join(sorted(_VALID_VIDEO_RESOLUTIONS))
                raise ValidationError(
                    f"Invalid resolution for xAI: {request.resolution}. "
                    f"Supported: {supported}",
                    provider=config.provider,
                )
            params["resolution"] = request.resolution

        # Aspect ratio — pass-through
        if request.aspect_ratio is not None:
            params["aspect_ratio"] = request.aspect_ratio

        # Image-to-video: use first image in image_list as image_url
        if request.image_list:
            first_image = request.image_list[0]
            image_val = first_image.get("image", "")
            if image_val:
                params["image_url"] = str(image_val)

        # Video editing: convert video field to video_url
        # video field is MediaType: Base64 | HttpUrl | MediaContent (dict with content/content_type)
        if request.video is not None:
            if isinstance(request.video, dict):
                video_url = request.video.get("url") or request.video.get("content")
                if video_url:
                    params["video_url"] = str(video_url)
            else:
                params["video_url"] = str(request.video)

        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)
        logger.info(
            "Mapped request to provider format",
            {"converted_request": params},
            redact=True,
        )
        return params

    def _convert_video_response(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        request_id: str,
        xai_response: Any,  # xAI SDK video response (type not exported)
    ) -> VideoGenerationResponse:
        """Convert xAI video response to VideoGenerationResponse."""
        respect_moderation = getattr(xai_response, "respect_moderation", True)
        if not respect_moderation:
            raise ContentModerationError(
                "xAI content moderation rejected the video generation",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response={"request_id": request_id, "respect_moderation": False},
            )

        video_url = getattr(xai_response, "url", None)
        if not video_url:
            raise GenerationFailedError(
                "No video URL in xAI generation response",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response={"request_id": request_id},
            )

        return VideoGenerationResponse(
            request_id=request_id,
            video=str(video_url),
            content_type="video/mp4",
            status="completed",
            raw_response={
                "request_id": request_id,
                "duration": getattr(xai_response, "duration", None),
                "model": getattr(xai_response, "model", config.model),
            },
            provider_metadata={},
        )

    def _convert_image_request(
        self, config: ImageGenerationConfig, request: ImageGenerationRequest
    ) -> dict[str, Any]:
        """Convert ImageGenerationRequest to xAI image API parameters."""
        params: dict[str, Any] = {
            "prompt": request.prompt,
            "model": config.model,
        }

        # Resolution — "1k" or "2k" (passed via extra_params since ImageGenerationRequest
        # does not have a resolution field)
        resolution = request.extra_params.get("resolution")
        if resolution is not None:
            resolution_str = str(resolution)
            if resolution_str not in _VALID_IMAGE_RESOLUTIONS:
                supported = ", ".join(sorted(_VALID_IMAGE_RESOLUTIONS))
                raise ValidationError(
                    f"Invalid resolution for xAI image: {resolution_str}. "
                    f"Supported: {supported}",
                    provider=config.provider,
                )
            params["resolution"] = resolution_str

        # Aspect ratio — pass-through
        if request.aspect_ratio is not None:
            params["aspect_ratio"] = request.aspect_ratio

        # Image references: single → image_url, multiple → image_urls
        if request.image_list:
            image_urls = [
                str(img.get("image", ""))
                for img in request.image_list
                if img.get("image")
            ]
            if len(image_urls) == 1:
                params["image_url"] = image_urls[0]
            elif len(image_urls) > 1:
                params["image_urls"] = image_urls

        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)
        logger.info(
            "Mapped image request to provider format",
            {"converted_request": params},
            redact=True,
        )
        return params

    def _convert_image_response(
        self,
        config: ImageGenerationConfig,
        request_id: str,
        xai_response: Any,  # xAI SDK image response (type not exported)
    ) -> ImageGenerationResponse:
        """Convert xAI image response to ImageGenerationResponse."""
        respect_moderation = getattr(xai_response, "respect_moderation", True)
        if not respect_moderation:
            raise ContentModerationError(
                "xAI content moderation rejected the image generation",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response={"request_id": request_id, "respect_moderation": False},
            )

        image_url = getattr(xai_response, "url", None)
        if not image_url:
            raise GenerationFailedError(
                "No image URL in xAI generation response",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response={"request_id": request_id},
            )

        return ImageGenerationResponse(
            request_id=request_id,
            images=[str(image_url)],
            content_type="image/png",
            status="completed",
            raw_response={
                "request_id": request_id,
                "model": getattr(xai_response, "model", config.model),
            },
        )

    def _handle_error(
        self,
        config: VideoGenerationConfig | ImageGenerationConfig,
        request_id: str,
        ex: Exception,
    ) -> TarashException:
        """Map xAI SDK / gRPC exceptions to TarashException subclasses."""
        if isinstance(ex, TarashException):
            return ex

        provider = config.provider
        model = config.model

        if has_xai_sdk:
            try:
                import grpc as _grpc

                if isinstance(ex, _grpc.RpcError):
                    code = ex.code()  # type: ignore[attr-defined]
                    details = ex.details() or str(ex)  # type: ignore[attr-defined]

                    if code == _grpc.StatusCode.DEADLINE_EXCEEDED:
                        return TimeoutError(
                            f"Request timed out: {details}",
                            provider=provider,
                            model=model,
                            request_id=request_id,
                            raw_response={"error": details, "grpc_code": str(code)},
                            timeout_seconds=float(config.timeout),
                        )
                    if code == _grpc.StatusCode.UNAVAILABLE:
                        return HTTPConnectionError(
                            f"Connection error: {details}",
                            provider=provider,
                            model=model,
                            request_id=request_id,
                            raw_response={"error": details, "grpc_code": str(code)},
                        )
                    if code == _grpc.StatusCode.UNAUTHENTICATED:
                        return HTTPError(
                            f"Authentication failed: {details}",
                            provider=provider,
                            model=model,
                            request_id=request_id,
                            raw_response={"error": details, "grpc_code": str(code)},
                            status_code=401,
                        )
                    if code == _grpc.StatusCode.PERMISSION_DENIED:
                        return ContentModerationError(
                            f"Permission denied: {details}",
                            provider=provider,
                            model=model,
                            request_id=request_id,
                            raw_response={"error": details, "grpc_code": str(code)},
                        )
                    if code == _grpc.StatusCode.INVALID_ARGUMENT:
                        return ValidationError(
                            f"Invalid request: {details}",
                            provider=provider,
                            model=model,
                            request_id=request_id,
                            raw_response={"error": details, "grpc_code": str(code)},
                        )
                    return HTTPError(
                        f"xAI API error: {details}",
                        provider=provider,
                        model=model,
                        request_id=request_id,
                        raw_response={"error": details, "grpc_code": str(code)},
                        status_code=0,
                    )
            except ImportError:
                pass

        log_error(
            f"xAI unknown error: {str(ex)}",
            context={
                "provider": provider,
                "model": model,
                "request_id": request_id,
                "error_type": type(ex).__name__,
            },
            logger_name=_LOGGER_NAME,
            exc_info=True,
        )
        return GenerationFailedError(
            f"Error while generating: {str(ex)}",
            provider=provider,
            model=model,
            request_id=request_id,
            raw_response={"error": str(ex), "error_type": type(ex).__name__},
        )

    async def _poll_until_complete(
        self,
        client: "Client | AsyncClient",
        request_id: str,
        max_poll_attempts: int,
        poll_interval: int,
        provider: str,
        model: str,
        on_progress: ProgressCallback | None,
        is_async: bool,
    ) -> Any:
        """Poll xAI video generation until DONE or EXPIRED (unified for sync/async)."""
        poll_attempts = 0
        last_response: Any = None

        while poll_attempts < max_poll_attempts:
            if is_async:
                await asyncio.sleep(poll_interval)
                get_result = client.video.get(request_id)
                if asyncio.iscoroutine(get_result):
                    last_response = await get_result
                else:
                    last_response = get_result
            else:
                time.sleep(poll_interval)
                last_response = client.video.get(request_id)

            poll_attempts += 1

            raw_status = getattr(
                getattr(last_response, "status", None), "name", "PENDING"
            ).upper()

            if on_progress:
                update = parse_xai_video_status(request_id, raw_status.lower())
                result = on_progress(update)
                if is_async and asyncio.iscoroutine(result):
                    await result

            logger = ProviderLogger(provider, model, _LOGGER_NAME, request_id)
            logger.info(
                "Progress status update",
                {"status": raw_status, "poll_attempt": poll_attempts},
            )

            if raw_status == "DONE":
                return last_response

            if raw_status == "EXPIRED":
                raise GenerationFailedError(
                    "xAI video generation expired before completing",
                    provider=provider,
                    model=model,
                    request_id=request_id,
                    raw_response={"status": "expired", "poll_attempts": poll_attempts},
                )

        timeout_seconds = max_poll_attempts * poll_interval
        raise TimeoutError(
            f"Video generation timed out after {max_poll_attempts} attempts "
            f"({timeout_seconds}s)",
            provider=provider,
            model=model,
            request_id=request_id,
            raw_response={"status": "timeout", "poll_attempts": poll_attempts},
            timeout_seconds=float(timeout_seconds),
        )

    @handle_video_generation_errors
    async def generate_video_async(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """Generate video asynchronously via xAI."""
        client = self._get_client(config.api_key, config.timeout, "async")
        params = self._convert_video_request(config, request)

        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)
        logger.debug("Starting xAI video API call")

        request_id: str | None = None
        try:
            start_result = client.video.start(**params)
            if asyncio.iscoroutine(start_result):
                start_response = await start_result
            else:
                start_response = start_result

            request_id = str(getattr(start_response, "request_id", "unknown"))
            logger = logger.with_request_id(request_id)
            logger.debug("xAI video request submitted")

            completed = await self._poll_until_complete(
                client,
                request_id,
                config.max_poll_attempts,
                config.poll_interval,
                config.provider,
                config.model,
                on_progress,
                is_async=True,
            )

            response = self._convert_video_response(
                config, request, request_id, completed
            )
            logger.info("Final generated response", {"response": response}, redact=True)
            return response

        except Exception as ex:
            raise self._handle_error(config, request_id or "unknown", ex)

    @handle_video_generation_errors
    def generate_video(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """Generate video synchronously via xAI (blocking)."""
        client = self._get_client(config.api_key, config.timeout, "sync")
        params = self._convert_video_request(config, request)

        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)
        logger.debug("Starting xAI video API call (sync)")

        request_id: str | None = None
        try:
            start_response = client.video.start(**params)
            request_id = str(getattr(start_response, "request_id", "unknown"))

            logger = logger.with_request_id(request_id)
            logger.debug("xAI video request submitted")

            completed = asyncio.run(
                self._poll_until_complete(
                    client,
                    request_id,
                    config.max_poll_attempts,
                    config.poll_interval,
                    config.provider,
                    config.model,
                    on_progress,
                    is_async=False,
                )
            )

            response = self._convert_video_response(
                config, request, request_id, completed
            )
            logger.info("Final generated response", {"response": response}, redact=True)
            return response

        except Exception as ex:
            raise self._handle_error(config, request_id or "unknown", ex)

    @handle_video_generation_errors  # pyright: ignore[reportArgumentType, reportUntypedFunctionDecorator]
    async def generate_image_async(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        on_progress: ImageProgressCallback | None = None,
    ) -> ImageGenerationResponse:
        """Generate image asynchronously via xAI (grok-imagine-image)."""
        import uuid

        client = self._get_client(config.api_key, config.timeout, "async")
        params = self._convert_image_request(config, request)
        request_id = str(uuid.uuid4())

        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)
        logger.debug("Starting xAI image API call")

        try:
            sample_result = client.image.sample(**params)
            if asyncio.iscoroutine(sample_result):
                xai_response = await sample_result
            else:
                xai_response = sample_result

            response = self._convert_image_response(config, request_id, xai_response)
            logger.info(
                "Final generated image response", {"response": response}, redact=True
            )
            return response

        except Exception as ex:
            raise self._handle_error(config, request_id, ex)

    @handle_video_generation_errors  # pyright: ignore[reportArgumentType, reportUntypedFunctionDecorator]
    def generate_image(
        self,
        config: ImageGenerationConfig,
        request: ImageGenerationRequest,
        on_progress: SyncImageProgressCallback | None = None,
    ) -> ImageGenerationResponse:
        """Generate image synchronously via xAI (grok-imagine-image, blocking)."""
        import uuid

        client = self._get_client(config.api_key, config.timeout, "sync")
        params = self._convert_image_request(config, request)
        request_id = str(uuid.uuid4())

        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)
        logger.debug("Starting xAI image API call (sync)")

        try:
            xai_response = client.image.sample(**params)
            response = self._convert_image_response(config, request_id, xai_response)
            logger.info(
                "Final generated image response", {"response": response}, redact=True
            )
            return response

        except Exception as ex:
            raise self._handle_error(config, request_id, ex)
