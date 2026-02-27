"""ElevenLabs provider handler for TTS and STS audio generation."""

import base64
import io
import json
import uuid
from typing import Any, Literal

from tarash.tarash_gateway.logging import log_info
from tarash.tarash_gateway.utils import (
    download_media_from_url,
    download_media_from_url_async,
)
from tarash.tarash_gateway.exceptions import (
    ContentModerationError,
    HTTPConnectionError,
    HTTPError,
    TarashException,
    TimeoutError,
    ValidationError,
    handle_audio_generation_errors,
)
from tarash.tarash_gateway.models import (
    AudioGenerationConfig,
    STSProgressCallback,
    STSRequest,
    STSResponse,
    TTSProgressCallback,
    TTSRequest,
    TTSResponse,
)

has_elevenlabs = True
try:
    from elevenlabs.client import AsyncElevenLabs, ElevenLabs
    from elevenlabs.core import ApiError
except ImportError:
    has_elevenlabs = False

_LOGGER_NAME = "tarash.tarash_gateway.providers.elevenlabs"


def _generate_request_id() -> str:
    """Generate a unique request ID."""
    return uuid.uuid4().hex


def _output_format_to_content_type(output_format: str) -> str:
    """Map ElevenLabs output format to MIME content type."""
    if output_format.startswith("mp3"):
        return "audio/mpeg"
    if output_format.startswith("pcm"):
        return "audio/pcm"
    if output_format.startswith("wav"):
        return "audio/wav"
    if output_format.startswith("opus"):
        return "audio/opus"
    if output_format.startswith("ulaw") or output_format.startswith("alaw"):
        return "audio/basic"
    return "audio/mpeg"


class ElevenLabsProviderHandler:
    """Handler for ElevenLabs TTS and STS audio generation."""

    def __init__(self) -> None:
        if not has_elevenlabs:
            raise ImportError(
                "elevenlabs is required for ElevenLabs provider. "
                "Install with: pip install tarash-gateway[elevenlabs]"
            )

    def _get_client(
        self, config: AudioGenerationConfig, client_type: Literal["sync", "async"]
    ) -> Any:
        """Create a fresh ElevenLabs client.

        Fresh client each call to avoid event loop issues with async client.
        """
        kwargs: dict[str, Any] = {}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        kwargs["timeout"] = config.timeout

        if client_type == "async":
            return AsyncElevenLabs(**kwargs)
        return ElevenLabs(**kwargs)

    def _convert_tts_request(
        self, config: AudioGenerationConfig, request: TTSRequest
    ) -> dict[str, Any]:
        """Convert TTSRequest to ElevenLabs SDK kwargs."""
        kwargs: dict[str, Any] = {
            "voice_id": request.voice_id,
            "text": request.text,
            "model_id": config.model,
        }

        kwargs["output_format"] = request.output_format

        if request.voice_settings:
            kwargs["voice_settings"] = request.voice_settings

        if request.language_code:
            kwargs["language_code"] = request.language_code

        if request.seed is not None:
            kwargs["seed"] = request.seed

        if request.previous_text:
            kwargs["previous_text"] = request.previous_text

        if request.next_text:
            kwargs["next_text"] = request.next_text

        # Merge extra_params
        kwargs.update(request.extra_params)

        return kwargs

    @staticmethod
    def _resolve_audio_bytes(audio: Any) -> io.BytesIO:
        """Convert MediaType audio input to BytesIO for the SDK (sync).

        Handles MediaContent (dict with content bytes), base64 strings, and URLs.
        """
        if isinstance(audio, dict) and "content" in audio:
            return io.BytesIO(audio["content"])  # type: ignore[index]
        if isinstance(audio, str):
            if audio.startswith("http://") or audio.startswith("https://"):
                content, _ = download_media_from_url(audio, provider="elevenlabs")
                return io.BytesIO(content)
            # Base64 string
            return io.BytesIO(base64.b64decode(audio))
        return io.BytesIO(audio)  # type: ignore[arg-type]

    @staticmethod
    async def _resolve_audio_bytes_async(audio: Any) -> io.BytesIO:
        """Convert MediaType audio input to BytesIO for the SDK (async).

        Handles MediaContent (dict with content bytes), base64 strings, and URLs.
        """
        if isinstance(audio, dict) and "content" in audio:
            return io.BytesIO(audio["content"])  # type: ignore[index]
        if isinstance(audio, str):
            if audio.startswith("http://") or audio.startswith("https://"):
                content, _ = await download_media_from_url_async(
                    audio, provider="elevenlabs"
                )
                return io.BytesIO(content)
            # Base64 string
            return io.BytesIO(base64.b64decode(audio))
        return io.BytesIO(audio)  # type: ignore[arg-type]

    def _convert_sts_request(
        self, config: AudioGenerationConfig, request: STSRequest
    ) -> dict[str, Any]:
        """Convert STSRequest to ElevenLabs SDK kwargs (without audio).

        Audio is resolved separately via _resolve_audio_bytes / _resolve_audio_bytes_async
        because URL downloads require sync/async handling.
        """
        kwargs: dict[str, Any] = {
            "voice_id": request.voice_id,
        }

        kwargs["model_id"] = config.model

        kwargs["output_format"] = request.output_format

        if request.voice_settings:
            kwargs["voice_settings"] = json.dumps(request.voice_settings)

        if request.seed is not None:
            kwargs["seed"] = request.seed

        if request.remove_background_noise:
            kwargs["remove_background_noise"] = request.remove_background_noise

        # Merge extra_params
        kwargs.update(request.extra_params)

        return kwargs

    def _convert_tts_response(
        self,
        config: AudioGenerationConfig,
        request: TTSRequest,
        request_id: str,
        audio_bytes: bytes,
    ) -> TTSResponse:
        """Convert raw audio bytes to TTSResponse."""
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        content_type = _output_format_to_content_type(request.output_format)

        return TTSResponse(
            request_id=request_id,
            audio=audio_b64,
            content_type=content_type,
            duration=None,
            status="completed",
            raw_response={
                "audio_size_bytes": len(audio_bytes),
                "output_format": request.output_format,
                "model": config.model,
                "voice_id": request.voice_id,
            },
        )

    def _convert_sts_response(
        self,
        config: AudioGenerationConfig,
        request: STSRequest,
        request_id: str,
        audio_bytes: bytes,
    ) -> STSResponse:
        """Convert raw audio bytes to STSResponse."""
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        content_type = _output_format_to_content_type(request.output_format)

        return STSResponse(
            request_id=request_id,
            audio=audio_b64,
            content_type=content_type,
            duration=None,
            status="completed",
            raw_response={
                "audio_size_bytes": len(audio_bytes),
                "output_format": request.output_format,
                "model": config.model,
                "voice_id": request.voice_id,
            },
        )

    def _handle_error(
        self,
        config: AudioGenerationConfig,
        request: TTSRequest | STSRequest,
        request_id: str,
        ex: Exception,
    ) -> TarashException:
        """Map ElevenLabs errors to TarashException hierarchy."""
        provider = config.provider
        model = config.model

        if isinstance(ex, ApiError):
            status_code = ex.status_code
            body = str(ex.body) if ex.body else str(ex)

            if status_code == 400 or status_code == 422:
                return ValidationError(
                    f"Invalid request: {body}",
                    provider=provider,
                    model=model,
                    request_id=request_id,
                    raw_response={"error": body, "status_code": status_code},
                )
            elif status_code == 401:
                return HTTPError(
                    f"Authentication failed: {body}",
                    provider=provider,
                    model=model,
                    request_id=request_id,
                    raw_response={"error": body, "status_code": status_code},
                    status_code=status_code,
                )
            elif status_code == 403:
                return ContentModerationError(
                    f"Content policy violation: {body}",
                    provider=provider,
                    model=model,
                    request_id=request_id,
                    raw_response={"error": body, "status_code": status_code},
                )
            elif status_code == 429:
                return HTTPError(
                    f"Rate limit exceeded: {body}",
                    provider=provider,
                    model=model,
                    request_id=request_id,
                    raw_response={"error": body, "status_code": status_code},
                    status_code=status_code,
                )
            elif status_code is not None and status_code >= 500:
                return HTTPError(
                    f"Server error: {body}",
                    provider=provider,
                    model=model,
                    request_id=request_id,
                    raw_response={"error": body, "status_code": status_code},
                    status_code=status_code,
                )
            else:
                return HTTPError(
                    f"API error: {body}",
                    provider=provider,
                    model=model,
                    request_id=request_id,
                    raw_response={"error": body, "status_code": status_code},
                    status_code=status_code,
                )

        if isinstance(ex, ConnectionError):
            return HTTPConnectionError(
                f"Connection error: {ex}",
                provider=provider,
                model=model,
                request_id=request_id,
            )

        if isinstance(ex, TimeoutError):
            return ex

        if isinstance(ex, (TimeoutError, OSError)) and "timed out" in str(ex).lower():
            return TimeoutError(
                f"Request timed out: {ex}",
                provider=provider,
                model=model,
                request_id=request_id,
                timeout_seconds=config.timeout,
            )

        return TarashException(
            f"Unknown error: {ex}",
            provider=provider,
            model=model,
            request_id=request_id,
            raw_response={"error": str(ex), "error_type": type(ex).__name__},
        )

    # ==================== TTS Generation ====================

    @handle_audio_generation_errors
    async def generate_tts_async(
        self,
        config: AudioGenerationConfig,
        request: TTSRequest,
        on_progress: TTSProgressCallback
        | None = None,  # Unused: ElevenLabs SDK streams chunks directly with no status/polling events
    ) -> TTSResponse:
        """Generate speech from text asynchronously."""
        client = self._get_client(config, "async")
        kwargs = self._convert_tts_request(config, request)
        request_id = _generate_request_id()

        log_info(
            "Starting TTS generation (async)",
            context={
                "model": config.model,
                "voice_id": request.voice_id,
                "text_length": len(request.text),
                "output_format": request.output_format,
                "request_id": request_id,
            },
            logger_name=_LOGGER_NAME,
        )

        try:
            audio_iter = client.text_to_speech.convert(**kwargs)
            audio_bytes = b"".join([chunk async for chunk in audio_iter])

            log_info(
                "TTS generation completed (async)",
                context={
                    "request_id": request_id,
                    "audio_size_bytes": len(audio_bytes),
                },
                logger_name=_LOGGER_NAME,
            )

            return self._convert_tts_response(config, request, request_id, audio_bytes)
        except (TarashException, Exception) as ex:
            if isinstance(ex, TarashException):
                raise
            raise self._handle_error(config, request, request_id, ex)

    @handle_audio_generation_errors
    def generate_tts(
        self,
        config: AudioGenerationConfig,
        request: TTSRequest,
        on_progress: TTSProgressCallback
        | None = None,  # Unused: ElevenLabs SDK streams chunks directly with no status/polling events
    ) -> TTSResponse:
        """Generate speech from text synchronously."""
        client = self._get_client(config, "sync")
        kwargs = self._convert_tts_request(config, request)
        request_id = _generate_request_id()

        log_info(
            "Starting TTS generation (sync)",
            context={
                "model": config.model,
                "voice_id": request.voice_id,
                "text_length": len(request.text),
                "output_format": request.output_format,
                "request_id": request_id,
            },
            logger_name=_LOGGER_NAME,
        )

        try:
            audio_iter = client.text_to_speech.convert(**kwargs)
            audio_bytes = b"".join(audio_iter)

            log_info(
                "TTS generation completed (sync)",
                context={
                    "request_id": request_id,
                    "audio_size_bytes": len(audio_bytes),
                },
                logger_name=_LOGGER_NAME,
            )

            return self._convert_tts_response(config, request, request_id, audio_bytes)
        except (TarashException, Exception) as ex:
            if isinstance(ex, TarashException):
                raise
            raise self._handle_error(config, request, request_id, ex)

    # ==================== STS Generation ====================

    @handle_audio_generation_errors
    async def generate_sts_async(
        self,
        config: AudioGenerationConfig,
        request: STSRequest,
        on_progress: STSProgressCallback
        | None = None,  # Unused: ElevenLabs SDK streams chunks directly with no status/polling events
    ) -> STSResponse:
        """Convert speech to speech asynchronously."""
        client = self._get_client(config, "async")
        kwargs = self._convert_sts_request(config, request)
        kwargs["audio"] = await self._resolve_audio_bytes_async(request.audio)
        request_id = _generate_request_id()

        log_info(
            "Starting STS generation (async)",
            context={
                "model": config.model,
                "voice_id": request.voice_id,
                "output_format": request.output_format,
                "request_id": request_id,
            },
            logger_name=_LOGGER_NAME,
        )

        try:
            audio_iter = client.speech_to_speech.convert(**kwargs)
            audio_bytes = b"".join([chunk async for chunk in audio_iter])

            log_info(
                "STS generation completed (async)",
                context={
                    "request_id": request_id,
                    "audio_size_bytes": len(audio_bytes),
                },
                logger_name=_LOGGER_NAME,
            )

            return self._convert_sts_response(config, request, request_id, audio_bytes)
        except (TarashException, Exception) as ex:
            if isinstance(ex, TarashException):
                raise
            raise self._handle_error(config, request, request_id, ex)

    @handle_audio_generation_errors
    def generate_sts(
        self,
        config: AudioGenerationConfig,
        request: STSRequest,
        on_progress: STSProgressCallback
        | None = None,  # Unused: ElevenLabs SDK streams chunks directly with no status/polling events
    ) -> STSResponse:
        """Convert speech to speech synchronously."""
        client = self._get_client(config, "sync")
        kwargs = self._convert_sts_request(config, request)
        kwargs["audio"] = self._resolve_audio_bytes(request.audio)
        request_id = _generate_request_id()

        log_info(
            "Starting STS generation (sync)",
            context={
                "model": config.model,
                "voice_id": request.voice_id,
                "output_format": request.output_format,
                "request_id": request_id,
            },
            logger_name=_LOGGER_NAME,
        )

        try:
            audio_iter = client.speech_to_speech.convert(**kwargs)
            audio_bytes = b"".join(audio_iter)

            log_info(
                "STS generation completed (sync)",
                context={
                    "request_id": request_id,
                    "audio_size_bytes": len(audio_bytes),
                },
                logger_name=_LOGGER_NAME,
            )

            return self._convert_sts_response(config, request, request_id, audio_bytes)
        except (TarashException, Exception) as ex:
            if isinstance(ex, TarashException):
                raise
            raise self._handle_error(config, request, request_id, ex)
