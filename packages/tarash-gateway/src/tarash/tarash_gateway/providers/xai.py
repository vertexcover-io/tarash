"""xAI provider handler for video and image generation."""

import asyncio
import time
from typing import TYPE_CHECKING, Any, Literal, cast, overload

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
from tarash.tarash_gateway.pricing import resolve_cost
from tarash.tarash_gateway.providers.field_mappers import (
    FieldMapper,
    GenerationRequest,
    ImageListItem,
    apply_field_mappers,
    get_field_mappers_from_registry,
    passthrough_field_mapper,
    single_image_field_mapper,
    video_url_field_mapper,
)


has_xai_sdk = True
_DEFERRED_STATUS_DONE: int = 1
_DEFERRED_STATUS_EXPIRED: int = 2
_DEFERRED_STATUS_PENDING: int = 3
try:
    from xai_sdk import AsyncClient, Client
    from xai_sdk.proto import deferred_pb2
    from xai_sdk.video import VideoResponse as XaiVideoResponse

    _DEFERRED_STATUS_DONE = deferred_pb2.DeferredStatus.DONE
    _DEFERRED_STATUS_EXPIRED = deferred_pb2.DeferredStatus.EXPIRED
    _DEFERRED_STATUS_PENDING = deferred_pb2.DeferredStatus.PENDING
except ImportError:
    has_xai_sdk = False

if TYPE_CHECKING:
    from xai_sdk import AsyncClient, Client


_LOGGER_NAME = "tarash.tarash_gateway.providers.xai"


# ==================== Custom Converters ====================


def _xai_duration_converter(_request: GenerationRequest, value: object) -> int | None:
    """Validate xAI video duration (integer 1-15 seconds)."""
    if value is None:
        return None
    duration = int(value) if isinstance(value, (int, float, str)) else 0
    if duration < 1 or duration > 15:
        raise ValidationError(
            f"Invalid duration for xAI: {duration}s. Supported: 1-15 seconds.",
            provider="xai",
        )
    return duration


def _xai_video_resolution_converter(
    _request: GenerationRequest, value: object
) -> str | None:
    """Validate xAI video resolution (480p or 720p)."""
    if value is None:
        return None
    resolution = str(value)
    valid = {"480p", "720p"}
    if resolution not in valid:
        supported = ", ".join(sorted(valid))
        raise ValidationError(
            f"Invalid resolution for xAI: {resolution}. Supported: {supported}",
            provider="xai",
        )
    return resolution


def _xai_image_resolution_converter(
    _request: GenerationRequest, value: object
) -> str | None:
    """Extract and validate xAI image resolution from extra_params (1k or 2k)."""
    if not isinstance(value, dict):
        return None
    resolution = value.get("resolution")
    if resolution is None:
        return None
    resolution_str = str(resolution)
    valid = {"1k", "2k"}
    if resolution_str not in valid:
        supported = ", ".join(sorted(valid))
        raise ValidationError(
            f"Invalid resolution for xAI image: {resolution_str}. Supported: {supported}",
            provider="xai",
        )
    return resolution_str


def _xai_single_image_converter(
    _request: GenerationRequest, value: object
) -> str | None:
    """Extract single image URL from image_list (when exactly 1 image)."""
    if not value:
        return None
    image_list = cast(list[ImageListItem], value)
    image_urls = [str(img.get("image", "")) for img in image_list if img.get("image")]
    if len(image_urls) == 1:
        return image_urls[0]
    return None


def _xai_multi_image_converter(
    _request: GenerationRequest, value: object
) -> list[str] | None:
    """Extract multiple image URLs from image_list (when 2+ images)."""
    if not value:
        return None
    image_list = cast(list[ImageListItem], value)
    image_urls = [str(img.get("image", "")) for img in image_list if img.get("image")]
    if len(image_urls) > 1:
        return image_urls
    return None


# ==================== Field Mapper Definitions ====================

XAI_VIDEO_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "duration": FieldMapper(
        source_field="duration_seconds", converter=_xai_duration_converter
    ),
    "resolution": FieldMapper(
        source_field="resolution", converter=_xai_video_resolution_converter
    ),
    "aspect_ratio": passthrough_field_mapper("aspect_ratio"),
    "image_url": single_image_field_mapper(),
    "video_url": video_url_field_mapper(),
}

XAI_IMAGE_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "resolution": FieldMapper(
        source_field="extra_params", converter=_xai_image_resolution_converter
    ),
    "aspect_ratio": passthrough_field_mapper("aspect_ratio"),
    "image_url": FieldMapper(
        source_field="image_list", converter=_xai_single_image_converter
    ),
    "image_urls": FieldMapper(
        source_field="image_list", converter=_xai_multi_image_converter
    ),
}


# ==================== Model Registries ====================

XAI_VIDEO_MODEL_REGISTRY: dict[str, dict[str, FieldMapper]] = {
    "grok-imagine-video": XAI_VIDEO_FIELD_MAPPERS,
}

XAI_IMAGE_MODEL_REGISTRY: dict[str, dict[str, FieldMapper]] = {
    "grok-imagine-image": XAI_IMAGE_FIELD_MAPPERS,
    "grok-2-image": XAI_IMAGE_FIELD_MAPPERS,
}


def get_xai_video_field_mappers(model_name: str) -> dict[str, FieldMapper]:
    """Get field mappers for xAI video model."""
    return get_field_mappers_from_registry(
        model_name, XAI_VIDEO_MODEL_REGISTRY, XAI_VIDEO_FIELD_MAPPERS
    )


def get_xai_image_field_mappers(model_name: str) -> dict[str, FieldMapper]:
    """Get field mappers for xAI image model."""
    return get_field_mappers_from_registry(
        model_name, XAI_IMAGE_MODEL_REGISTRY, XAI_IMAGE_FIELD_MAPPERS
    )


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
        field_mappers = get_xai_video_field_mappers(config.model)
        params: dict[str, Any] = dict(apply_field_mappers(field_mappers, request))
        params["model"] = config.model

        # Merge extra_params for manual overrides
        if request.extra_params:
            for key, value in request.extra_params.items():
                if key not in params and value is not None:
                    params[key] = value

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

        # Resolve cost using output duration if available, else quantity=1.0
        duration = getattr(xai_response, "duration", None)
        quantity = float(duration) if duration is not None else 1.0
        cost = resolve_cost(config.provider, config.model, None, quantity)

        return VideoGenerationResponse(
            request_id=request_id,
            video=str(video_url),
            content_type="video/mp4",
            status="completed",
            cost=cost,
            raw_response={
                "request_id": request_id,
                "duration": duration,
                "model": getattr(xai_response, "model", config.model),
            },
        )

    def _convert_image_request(
        self, config: ImageGenerationConfig, request: ImageGenerationRequest
    ) -> dict[str, Any]:
        """Convert ImageGenerationRequest to xAI image API parameters."""
        field_mappers = get_xai_image_field_mappers(config.model)
        params: dict[str, Any] = dict(apply_field_mappers(field_mappers, request))
        params["model"] = config.model

        # Merge extra_params for manual overrides (skip "resolution" since it's
        # already handled by the field mapper)
        if request.extra_params:
            for key, value in request.extra_params.items():
                if key not in params and key != "resolution" and value is not None:
                    params[key] = value

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

        # Resolve cost with quantity=1 per image
        cost = resolve_cost(config.provider, config.model, None, 1.0)

        return ImageGenerationResponse(
            request_id=request_id,
            images=[str(image_url)],
            content_type="image/png",
            status="completed",
            cost=cost,
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

            status_int = getattr(last_response, "status", _DEFERRED_STATUS_PENDING)

            # Map protobuf enum int to human-readable string for logging
            _STATUS_INT_TO_STR = {
                _DEFERRED_STATUS_DONE: "DONE",
                _DEFERRED_STATUS_EXPIRED: "EXPIRED",
                _DEFERRED_STATUS_PENDING: "PENDING",
            }
            raw_status = _STATUS_INT_TO_STR.get(status_int, f"UNKNOWN({status_int})")

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

            if status_int == _DEFERRED_STATUS_DONE:
                return last_response

            if status_int == _DEFERRED_STATUS_EXPIRED:
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

            # GetDeferredVideoResponse has .response (VideoResponse proto);
            # wrap it with XaiVideoResponse to get .url, .duration, etc.
            video_resp = XaiVideoResponse(completed.response)
            response = self._convert_video_response(
                config, request, request_id, video_resp
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

            # GetDeferredVideoResponse has .response (VideoResponse proto);
            # wrap it with XaiVideoResponse to get .url, .duration, etc.
            video_resp = XaiVideoResponse(completed.response)
            response = self._convert_video_response(
                config, request, request_id, video_resp
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
