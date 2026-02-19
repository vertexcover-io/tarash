"""HeyGen provider handler for avatar-based video generation."""

import asyncio
import time
from typing import Any, Literal, overload

import httpx
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
    ProgressCallback,
    VideoGenerationConfig,
    VideoGenerationRequest,
    VideoGenerationResponse,
    VideoGenerationUpdate,
)
from tarash.tarash_gateway.utils import validate_model_params

_LOGGER_NAME = "tarash.tarash_gateway.providers.heygen"
_HEYGEN_CREATE_URL = "https://api.heygen.com/v2/video/generate"
_HEYGEN_STATUS_URL = "https://api.heygen.com/v1/video_status.get"

# Maps aspect_ratio string to (width, height) in pixels
_ASPECT_RATIO_DIMENSIONS: dict[str, tuple[int, int]] = {
    "16:9": (1920, 1080),
    "9:16": (1080, 1920),
    "1:1": (1080, 1080),
    "4:3": (1440, 1080),
    "21:9": (2560, 1080),
}

# Maps HeyGen status strings to internal StatusType literals
_STATUS_MAP: dict[str, Literal["queued", "processing", "completed", "failed"]] = {
    "pending": "queued",
    "waiting": "queued",
    "processing": "processing",
    "completed": "completed",
    "failed": "failed",
}

# Terminal statuses — polling stops when one of these is reached
_TERMINAL_STATUSES = frozenset({"completed", "failed"})


class HeyGenVideoParams(TypedDict, total=False):
    """HeyGen-specific parameters passable via extra_params."""

    avatar_id: str
    voice_id: str
    avatar_style: str  # "normal" | "circle" | "closeUp"
    background_type: str  # "color" | "image" | "video"
    background_value: str  # hex color or URL
    voice_speed: float  # 0.5-1.5
    voice_pitch: int  # -50 to 50
    voice_emotion: (
        str  # "Excited" | "Friendly" | "Serious" | "Soothing" | "Broadcaster"
    )
    caption: bool
    title: str
    matting: bool


def parse_heygen_status(video_id: str, status_data: AnyDict) -> VideoGenerationUpdate:
    """Parse HeyGen status response dict to VideoGenerationUpdate.

    Args:
        video_id: HeyGen video ID
        status_data: The ``data`` object from HeyGen's status polling response

    Returns:
        Normalized VideoGenerationUpdate
    """
    heygen_status = str(status_data.get("status", "")).lower()
    mapped: Literal["queued", "processing", "completed", "failed"] = _STATUS_MAP.get(
        heygen_status, "processing"
    )
    return VideoGenerationUpdate(
        request_id=video_id,
        status=mapped,
        progress_percent=None,
        update={"raw_status": heygen_status},
    )


class HeyGenProviderHandler:
    """Handler for HeyGen avatar-based video generation."""

    def __init__(self) -> None:
        """Initialize handler with empty client caches."""
        self._sync_client_cache: dict[str, httpx.Client] = {}
        self._async_client_cache: dict[str, httpx.AsyncClient] = {}

    @overload
    def _get_client(
        self, config: VideoGenerationConfig, client_type: Literal["async"]
    ) -> httpx.AsyncClient: ...

    @overload
    def _get_client(
        self, config: VideoGenerationConfig, client_type: Literal["sync"]
    ) -> httpx.Client: ...

    def _get_client(
        self, config: VideoGenerationConfig, client_type: Literal["sync", "async"]
    ) -> httpx.Client | httpx.AsyncClient:
        """Get or create an httpx client, cached by api_key and client_type."""
        api_key = config.api_key or ""
        cache_key = f"{api_key}:{client_type}"
        headers = {
            "x-api-key": api_key,
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        timeout = httpx.Timeout(float(config.timeout), connect=30.0)

        if client_type == "async":
            if cache_key not in self._async_client_cache:
                logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)
                logger.debug("Creating new async httpx client")
                self._async_client_cache[cache_key] = httpx.AsyncClient(
                    headers=headers, timeout=timeout
                )
            return self._async_client_cache[cache_key]
        else:
            if cache_key not in self._sync_client_cache:
                logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)
                logger.debug("Creating new sync httpx client")
                self._sync_client_cache[cache_key] = httpx.Client(
                    headers=headers, timeout=timeout
                )
            return self._sync_client_cache[cache_key]

    def _validate_params(
        self, config: VideoGenerationConfig, request: VideoGenerationRequest
    ) -> dict[str, Any]:
        """Validate extra_params against HeyGenVideoParams schema."""
        return (
            validate_model_params(
                schema=HeyGenVideoParams,
                data=request.extra_params,
                provider=config.provider,
                model=config.model,
            )
            if request.extra_params
            else {}
        )

    def _convert_request(
        self, config: VideoGenerationConfig, request: VideoGenerationRequest
    ) -> dict[str, Any]:
        """Convert VideoGenerationRequest to HeyGen API payload.

        avatar_id and voice_id are resolved by checking extra_params first,
        then falling back to provider_config defaults.

        duration_seconds, image_list, and video are silently ignored since
        HeyGen's avatar model does not support them.
        """
        params: dict[str, Any] = self._validate_params(config, request)
        pc = config.provider_config

        # Resolve required IDs — extra_params take precedence over provider_config
        avatar_id: str | None = (
            str(params.pop("avatar_id"))
            if "avatar_id" in params
            else pc.get("avatar_id")
        )
        voice_id: str | None = (
            str(params.pop("voice_id")) if "voice_id" in params else pc.get("voice_id")
        )

        if not avatar_id:
            raise ValidationError(
                "avatar_id is required. Set it in provider_config or pass via extra_params.",
                provider=config.provider,
                model=config.model,
            )
        if not voice_id:
            raise ValidationError(
                "voice_id is required. Set it in provider_config or pass via extra_params.",
                provider=config.provider,
                model=config.model,
            )

        # Resolve dimensions from aspect_ratio
        aspect_ratio = request.aspect_ratio
        if aspect_ratio is not None and aspect_ratio not in _ASPECT_RATIO_DIMENSIONS:
            supported = ", ".join(_ASPECT_RATIO_DIMENSIONS.keys())
            raise ValidationError(
                f"Invalid aspect ratio for HeyGen: {aspect_ratio}. Supported: {supported}",
                provider=config.provider,
                model=config.model,
            )
        width, height = _ASPECT_RATIO_DIMENSIONS.get(
            aspect_ratio or "16:9", (1920, 1080)
        )

        # Build character dict
        character: dict[str, Any] = {
            "type": "avatar",
            "avatar_id": avatar_id,
            "scale": 1.0,
        }
        if "avatar_style" in params:
            character["avatar_style"] = params.pop("avatar_style")
        if "matting" in params:
            character["matting"] = params.pop("matting")

        # Build voice dict
        voice: dict[str, Any] = {
            "type": "text",
            "voice_id": voice_id,
            "input_text": request.prompt,
            "speed": params.pop("voice_speed", 1.0),
            "pitch": params.pop("voice_pitch", 0),
        }
        if "voice_emotion" in params:
            voice["emotion"] = params.pop("voice_emotion")

        # Build background dict
        background: dict[str, Any] = {
            "type": params.pop("background_type", "color"),
            "value": params.pop("background_value", "#FFFFFF"),
        }

        payload: dict[str, Any] = {
            "video_inputs": [
                {
                    "character": character,
                    "voice": voice,
                    "background": background,
                }
            ],
            "dimension": {"width": width, "height": height},
            "caption": params.pop("caption", False),
        }
        if "title" in params:
            payload["title"] = params.pop("title")

        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)
        logger.info(
            "Mapped request to provider format",
            {"converted_request": payload},
            redact=True,
        )

        return payload

    def _convert_response(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        request_id: str,
        status_data: AnyDict,
    ) -> VideoGenerationResponse:
        """Convert HeyGen completed status data to VideoGenerationResponse."""
        status = str(status_data.get("status", "")).lower()

        if status == "failed":
            error_info = status_data.get("error") or {}
            error_msg = str(
                error_info.get("message", "Video generation failed")
                if isinstance(error_info, dict)
                else error_info
            )
            log_error(
                "HeyGen video generation failed",
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
                    "video_id": request_id,
                    "status": status,
                    "error": error_info,
                },
            )

        video_url = status_data.get("video_url")
        if not video_url:
            raise GenerationFailedError(
                "No video URL in completed response",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response=dict(status_data),
            )

        duration = status_data.get("duration")
        thumbnail_url = status_data.get("thumbnail_url")

        return VideoGenerationResponse(
            request_id=request_id,
            video=str(video_url),
            content_type="video/mp4",
            duration=float(duration) if duration is not None else None,
            status="completed",
            raw_response=dict(status_data),
            provider_metadata={"thumbnail_url": thumbnail_url} if thumbnail_url else {},
        )

    def _handle_error(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        request_id: str,
        ex: Exception,
    ) -> TarashException:
        """Map exceptions to TarashException subclasses."""
        if isinstance(ex, TarashException):
            return ex

        if isinstance(ex, httpx.TimeoutException):
            return TimeoutError(
                f"Request timed out: {ex}",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response={"error": str(ex)},
                timeout_seconds=float(config.timeout),
            )

        if isinstance(ex, (httpx.ConnectError, httpx.NetworkError)):
            return HTTPConnectionError(
                f"Connection error: {ex}",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response={"error": str(ex)},
            )

        if isinstance(ex, httpx.HTTPStatusError):
            status_code = ex.response.status_code
            try:
                body: object = ex.response.json()
            except Exception:
                body = ex.response.text
            raw: AnyDict = {"status_code": status_code, "body": body}
            if status_code in (400, 422):
                return ValidationError(
                    str(ex),
                    provider=config.provider,
                    model=config.model,
                    request_id=request_id,
                    raw_response=raw,
                )
            return HTTPError(
                str(ex),
                provider=config.provider,
                model=config.model,
                request_id=request_id,
                raw_response=raw,
                status_code=status_code,
            )

        log_error(
            f"HeyGen unknown error: {ex}",
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
            f"Error while generating video: {ex}",
            provider=config.provider,
            model=config.model,
            request_id=request_id,
            raw_response={"error": str(ex), "error_type": type(ex).__name__},
        )

    async def _poll_until_complete_async(
        self,
        client: httpx.AsyncClient,
        video_id: str,
        config: VideoGenerationConfig,
        on_progress: ProgressCallback | None,
    ) -> AnyDict:
        """Poll HeyGen status asynchronously until complete or failed.

        Returns:
            Final status data dict from HeyGen

        Raises:
            TimeoutError: If max_poll_attempts exceeded without terminal status
        """
        poll_attempts = 0
        status_data: AnyDict = {}

        while poll_attempts < config.max_poll_attempts:
            await asyncio.sleep(config.poll_interval)
            response = await client.get(
                _HEYGEN_STATUS_URL, params={"video_id": video_id}
            )
            response.raise_for_status()
            resp_json: dict[str, Any] = response.json()
            status_data = resp_json.get("data") or {}
            poll_attempts += 1

            if on_progress:
                update = parse_heygen_status(video_id, status_data)
                result = on_progress(update)
                if asyncio.iscoroutine(result):
                    await result

            heygen_status = str(status_data.get("status", "")).lower()
            mapped = _STATUS_MAP.get(heygen_status, "processing")

            logger = ProviderLogger(
                config.provider, config.model, _LOGGER_NAME, video_id
            )
            logger.info(
                "Progress status update",
                {"status": heygen_status, "poll_attempt": poll_attempts},
            )

            if mapped in _TERMINAL_STATUSES:
                break

        final_status = str(status_data.get("status", "")).lower()
        if final_status not in {"completed", "failed"}:
            timeout_seconds = config.max_poll_attempts * config.poll_interval
            raise TimeoutError(
                f"Video generation timed out after {config.max_poll_attempts} attempts ({timeout_seconds}s)",
                provider=config.provider,
                model=config.model,
                request_id=video_id,
                raw_response={"status": "timeout", "poll_attempts": poll_attempts},
                timeout_seconds=float(timeout_seconds),
            )

        return status_data

    def _poll_until_complete_sync(
        self,
        client: httpx.Client,
        video_id: str,
        config: VideoGenerationConfig,
        on_progress: ProgressCallback | None,
    ) -> AnyDict:
        """Poll HeyGen status synchronously until complete or failed.

        Returns:
            Final status data dict from HeyGen

        Raises:
            TimeoutError: If max_poll_attempts exceeded without terminal status
        """
        poll_attempts = 0
        status_data: AnyDict = {}

        while poll_attempts < config.max_poll_attempts:
            time.sleep(config.poll_interval)
            response = client.get(_HEYGEN_STATUS_URL, params={"video_id": video_id})
            response.raise_for_status()
            resp_json: dict[str, Any] = response.json()
            status_data = resp_json.get("data") or {}
            poll_attempts += 1

            if on_progress:
                update = parse_heygen_status(video_id, status_data)
                on_progress(update)

            heygen_status = str(status_data.get("status", "")).lower()
            mapped = _STATUS_MAP.get(heygen_status, "processing")

            logger = ProviderLogger(
                config.provider, config.model, _LOGGER_NAME, video_id
            )
            logger.info(
                "Progress status update",
                {"status": heygen_status, "poll_attempt": poll_attempts},
            )

            if mapped in _TERMINAL_STATUSES:
                break

        final_status = str(status_data.get("status", "")).lower()
        if final_status not in {"completed", "failed"}:
            timeout_seconds = config.max_poll_attempts * config.poll_interval
            raise TimeoutError(
                f"Video generation timed out after {config.max_poll_attempts} attempts ({timeout_seconds}s)",
                provider=config.provider,
                model=config.model,
                request_id=video_id,
                raw_response={"status": "timeout", "poll_attempts": poll_attempts},
                timeout_seconds=float(timeout_seconds),
            )

        return status_data

    @handle_video_generation_errors
    async def generate_video_async(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """Generate avatar video asynchronously via HeyGen."""
        client = self._get_client(config, "async")
        payload = self._convert_request(config, request)

        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)
        logger.debug("Starting API call")

        video_id: str | None = None
        try:
            response = await client.post(_HEYGEN_CREATE_URL, json=payload)
            response.raise_for_status()
            resp_json: dict[str, Any] = response.json()

            if resp_json.get("error"):
                raise GenerationFailedError(
                    str(resp_json["error"]),
                    provider=config.provider,
                    model=config.model,
                    request_id="unknown",
                    raw_response=resp_json,
                )

            video_id = str(resp_json["data"]["video_id"])
            logger = logger.with_request_id(video_id)
            logger.debug("Request submitted")

            if on_progress:
                initial_update = VideoGenerationUpdate(
                    request_id=video_id,
                    status="queued",
                    progress_percent=None,
                    update={"raw_status": "submitted"},
                )
                init_result = on_progress(initial_update)
                if asyncio.iscoroutine(init_result):
                    await init_result

            status_data = await self._poll_until_complete_async(
                client, video_id, config, on_progress
            )

            response_obj = self._convert_response(
                config, request, video_id, status_data
            )
            logger.info(
                "Final generated response", {"response": response_obj}, redact=True
            )
            return response_obj

        except Exception as ex:
            raise self._handle_error(config, request, video_id or "unknown", ex)

    @handle_video_generation_errors
    def generate_video(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        """Generate avatar video synchronously via HeyGen."""
        client = self._get_client(config, "sync")
        payload = self._convert_request(config, request)

        logger = ProviderLogger(config.provider, config.model, _LOGGER_NAME)
        logger.debug("Starting API call")

        video_id: str | None = None
        try:
            response = client.post(_HEYGEN_CREATE_URL, json=payload)
            response.raise_for_status()
            resp_json: dict[str, Any] = response.json()

            if resp_json.get("error"):
                raise GenerationFailedError(
                    str(resp_json["error"]),
                    provider=config.provider,
                    model=config.model,
                    request_id="unknown",
                    raw_response=resp_json,
                )

            video_id = str(resp_json["data"]["video_id"])
            logger = logger.with_request_id(video_id)
            logger.debug("Request submitted")

            if on_progress:
                initial_update = VideoGenerationUpdate(
                    request_id=video_id,
                    status="queued",
                    progress_percent=None,
                    update={"raw_status": "submitted"},
                )
                on_progress(initial_update)

            status_data = self._poll_until_complete_sync(
                client, video_id, config, on_progress
            )

            response_obj = self._convert_response(
                config, request, video_id, status_data
            )
            logger.info(
                "Final generated response", {"response": response_obj}, redact=True
            )
            return response_obj

        except Exception as ex:
            raise self._handle_error(config, request, video_id or "unknown", ex)
