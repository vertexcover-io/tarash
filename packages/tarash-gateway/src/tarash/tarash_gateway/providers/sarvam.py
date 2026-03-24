"""Sarvam AI provider handler for TTS audio generation."""

import uuid
from typing import Any, Literal

import httpx

from tarash.tarash_gateway.logging import log_info
from tarash.tarash_gateway.providers.field_mappers import (
    FieldMapper,
    apply_field_mappers,
    passthrough_field_mapper,
)
from tarash.tarash_gateway.exceptions import (
    ContentModerationError,
    GenerationFailedError,
    HTTPConnectionError,
    HTTPError,
    TarashException,
    TimeoutError,
    ValidationError,
    handle_audio_generation_errors,
)
from tarash.tarash_gateway.models import (
    AudioGenerationConfig,
    TTSProgressCallback,
    TTSRequest,
    TTSResponse,
    format_to_content_type,
)

has_sarvam = True
try:
    from sarvamai import AsyncSarvamAI, SarvamAI
    from sarvamai.core.api_error import ApiError as SarvamApiError
except ImportError:
    has_sarvam = False

_LOGGER_NAME = "tarash.tarash_gateway.providers.sarvam"

_DEFAULT_SAMPLE_RATE = 24000


def _generate_request_id() -> str:
    """Generate a unique request ID."""
    return uuid.uuid4().hex


def _sarvam_codec_converter(_request: TTSRequest, value: object) -> str | None:
    """Extract audio codec from AudioOutputFormat."""
    if value is None:
        return None
    fmt = value.format  # type: ignore[union-attr]
    return fmt if fmt else None


def _sarvam_sample_rate_converter(_request: TTSRequest, value: object) -> int:
    """Extract sample rate from AudioOutputFormat, defaulting to 24000."""
    if value is None:
        return _DEFAULT_SAMPLE_RATE
    sr = value.sample_rate  # type: ignore[union-attr]
    return sr if sr is not None else _DEFAULT_SAMPLE_RATE


SARVAM_TTS_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "text": passthrough_field_mapper("text", required=True),
    "target_language_code": passthrough_field_mapper("language_code", required=True),
    "speaker": passthrough_field_mapper("voice_id"),
    "output_audio_codec": FieldMapper(
        source_field="output_format",
        converter=_sarvam_codec_converter,
    ),
    "speech_sample_rate": FieldMapper(
        source_field="output_format",
        converter=_sarvam_sample_rate_converter,
    ),
}


class SarvamProviderHandler:
    """Handler for Sarvam AI TTS audio generation.

    Sarvam specializes in Indian-language TTS with support for 11 languages.
    STS (speech-to-speech) is not supported by Sarvam.
    """

    def __init__(self) -> None:
        if not has_sarvam:
            raise ImportError(
                "sarvamai is required for Sarvam provider. "
                "Install with: pip install tarash-gateway[sarvam]"
            )

    def _get_client(
        self, config: AudioGenerationConfig, client_type: Literal["sync", "async"]
    ) -> Any:
        """Create a fresh Sarvam client."""
        if not config.api_key:
            raise ValidationError(
                "api_key is required for Sarvam provider",
                provider=config.provider,
                model=config.model,
            )

        if client_type == "async":
            return AsyncSarvamAI(
                api_subscription_key=config.api_key, timeout=config.timeout_seconds
            )
        return SarvamAI(
            api_subscription_key=config.api_key, timeout=config.timeout_seconds
        )

    def _validate_request(
        self, config: AudioGenerationConfig, request: TTSRequest
    ) -> None:
        """Validate TTS request for Sarvam-specific requirements."""
        if not request.language_code:
            raise ValidationError(
                "language_code is required for Sarvam provider. "
                "Sarvam requires target_language_code (e.g. 'hi-IN', 'en-IN', 'ta-IN').",
                provider=config.provider,
                model=config.model,
            )

    def _convert_tts_request(
        self, config: AudioGenerationConfig, request: TTSRequest
    ) -> dict[str, Any]:
        """Convert TTSRequest to Sarvam SDK kwargs."""
        kwargs = apply_field_mappers(SARVAM_TTS_FIELD_MAPPERS, request)
        kwargs["model"] = config.model
        # voice_settings merged first; extra_params can override if needed
        if request.voice_settings:
            kwargs.update(request.voice_settings)
        kwargs.update(request.extra_params)
        return kwargs

    def _convert_tts_response(
        self,
        config: AudioGenerationConfig,
        request: TTSRequest,
        request_id: str,
        sarvam_result: Any,
    ) -> TTSResponse:
        """Convert Sarvam response to TTSResponse."""
        if not getattr(sarvam_result, "audios", None):
            raise GenerationFailedError(
                "Sarvam returned empty audio response",
                provider=config.provider,
                model=config.model,
                request_id=request_id,
            )
        audio_b64 = sarvam_result.audios[0]
        content_type = format_to_content_type(request.output_format.format)

        # Use Sarvam's request_id if available, otherwise use our generated one
        result_request_id = getattr(sarvam_result, "request_id", None) or request_id

        return TTSResponse(
            request_id=result_request_id,
            audio=audio_b64,
            content_type=content_type,
            duration=None,
            status="completed",
            raw_response={
                "model": config.model,
                "voice_id": request.voice_id,
                "target_language_code": request.language_code,
                "output_audio_codec": request.output_format.format,
                "audio_b64_length": len(audio_b64),
            },
        )

    def _handle_error(
        self,
        config: AudioGenerationConfig,
        request: TTSRequest,
        request_id: str,
        ex: Exception,
    ) -> TarashException:
        """Map Sarvam errors to TarashException hierarchy."""
        provider = config.provider
        model = config.model

        # Check timeout BEFORE connection (timeout is subclass of connection)
        if isinstance(ex, httpx.TimeoutException):
            return TimeoutError(
                f"Request timed out: {ex}",
                provider=provider,
                model=model,
                request_id=request_id,
                timeout_seconds=config.timeout_seconds,
            )

        if isinstance(ex, httpx.ConnectError):
            return HTTPConnectionError(
                f"Connection error: {ex}",
                provider=provider,
                model=model,
                request_id=request_id,
            )

        if isinstance(ex, SarvamApiError):
            status_code = ex.status_code
            body = str(ex.body) if hasattr(ex, "body") and ex.body else str(ex)

            if status_code in (400, 422):
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
            elif status_code == 500:
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
        | None = None,  # Unused: Sarvam returns complete audio in a single response
    ) -> TTSResponse:
        """Generate speech from text asynchronously."""
        self._validate_request(config, request)
        client = self._get_client(config, "async")
        kwargs = self._convert_tts_request(config, request)
        request_id = _generate_request_id()

        log_info(
            "Starting TTS generation (async)",
            context={
                "model": config.model,
                "voice_id": request.voice_id,
                "text_length": len(request.text),
                "language_code": request.language_code,
                "request_id": request_id,
            },
            logger_name=_LOGGER_NAME,
        )

        try:
            result = await client.text_to_speech.convert(**kwargs)

            log_info(
                "TTS generation completed (async)",
                context={"request_id": request_id},
                logger_name=_LOGGER_NAME,
            )

            return self._convert_tts_response(config, request, request_id, result)
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
        | None = None,  # Unused: Sarvam returns complete audio in a single response
    ) -> TTSResponse:
        """Generate speech from text synchronously."""
        self._validate_request(config, request)
        client = self._get_client(config, "sync")
        kwargs = self._convert_tts_request(config, request)
        request_id = _generate_request_id()

        log_info(
            "Starting TTS generation (sync)",
            context={
                "model": config.model,
                "voice_id": request.voice_id,
                "text_length": len(request.text),
                "language_code": request.language_code,
                "request_id": request_id,
            },
            logger_name=_LOGGER_NAME,
        )

        try:
            result = client.text_to_speech.convert(**kwargs)

            log_info(
                "TTS generation completed (sync)",
                context={"request_id": request_id},
                logger_name=_LOGGER_NAME,
            )

            return self._convert_tts_response(config, request, request_id, result)
        except (TarashException, Exception) as ex:
            if isinstance(ex, TarashException):
                raise
            raise self._handle_error(config, request, request_id, ex)
