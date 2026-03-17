"""Hume AI provider handler for TTS audio generation."""

import uuid
from typing import Any, Literal

import httpx

from tarash.tarash_gateway.logging import log_info
from tarash.tarash_gateway.providers.field_mappers import (
    FieldMapper,
    apply_field_mappers,
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
    GenerationCost,
    TTSProgressCallback,
    TTSRequest,
    TTSResponse,
    format_to_content_type,
)

has_hume = True
try:
    from hume.client import AsyncHumeClient, HumeClient
    from hume.core import ApiError as HumeApiError
    from hume.tts import PostedUtterance
except ImportError:
    has_hume = False

_LOGGER_NAME = "tarash.tarash_gateway.providers.hume"


def _generate_request_id() -> str:
    """Generate a unique request ID."""
    return uuid.uuid4().hex


def _model_to_version(model: str) -> str | None:
    """Extract Octave version from model name.

    Returns "1", "2", or None (let API auto-select).
    """
    model_lower = model.lower()
    if "v1" in model_lower or "octave-1" in model_lower:
        return "1"
    if "v2" in model_lower or "octave-2" in model_lower:
        return "2"
    return None


def _build_voice_spec(request: TTSRequest) -> dict[str, Any] | None:
    """Build Hume voice specification from request.

    By default, uses voice name with HUME_AI provider.
    If voice_settings["voice_id_mode"] == "id", uses voice ID lookup.
    If voice_settings["voice_provider"] is set, uses that provider.
    """
    if not request.voice_id:
        return None

    voice_settings = request.voice_settings or {}
    voice_id_mode = voice_settings.get("voice_id_mode", "name")
    voice_provider = voice_settings.get("voice_provider", "HUME_AI")

    if voice_id_mode == "id":
        return {"id": request.voice_id, "provider": voice_provider}
    return {"name": request.voice_id, "provider": voice_provider}


def _build_utterances(request: TTSRequest, _value: object) -> list:
    """Build list of PostedUtterance from the TTSRequest."""
    utterance_kwargs: dict[str, Any] = {"text": request.text}
    voice = _build_voice_spec(request)
    if voice:
        utterance_kwargs["voice"] = voice
    vs = request.voice_settings or {}
    for key in ("description", "speed", "trailing_silence"):
        if key in vs:
            utterance_kwargs[key] = vs[key]
    return [PostedUtterance(**utterance_kwargs)]


HUME_TTS_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "utterances": FieldMapper(
        source_field="text",
        converter=_build_utterances,
        required=True,
    ),
    "format": FieldMapper(
        source_field="output_format",
        converter=lambda req, val: {"type": req.output_format.format},
    ),
}


class HumeProviderHandler:
    """Handler for Hume AI TTS audio generation.

    Hume provides expressive TTS with emotional control via the Octave model.
    STS (speech-to-speech) is not supported by Hume.
    """

    def __init__(self) -> None:
        if not has_hume:
            raise ImportError(
                "hume is required for Hume provider. "
                "Install with: pip install tarash-gateway[hume]"
            )

    def _get_client(
        self, config: AudioGenerationConfig, client_type: Literal["sync", "async"]
    ) -> Any:
        """Create a fresh Hume client."""
        if not config.api_key:
            raise ValidationError(
                "api_key is required for Hume provider",
                provider=config.provider,
                model=config.model,
            )

        kwargs: dict[str, Any] = {"api_key": config.api_key}
        if config.timeout:
            kwargs["timeout"] = config.timeout

        if client_type == "async":
            return AsyncHumeClient(**kwargs)
        return HumeClient(**kwargs)

    def _convert_tts_request(
        self, config: AudioGenerationConfig, request: TTSRequest
    ) -> dict[str, Any]:
        """Convert TTSRequest to Hume SDK synthesize_json kwargs."""
        kwargs = apply_field_mappers(HUME_TTS_FIELD_MAPPERS, request)

        # Model version from config.model
        version = _model_to_version(config.model)
        if version is not None:
            kwargs["version"] = version

        kwargs.update(request.extra_params)

        return kwargs

    def _convert_tts_response(
        self,
        config: AudioGenerationConfig,
        request: TTSRequest,
        request_id: str,
        hume_result: Any,
    ) -> TTSResponse:
        """Convert Hume response to TTSResponse."""
        generation = hume_result.generations[0]
        audio_b64 = generation.audio
        content_type = format_to_content_type(request.output_format.format)

        # Extract duration if available
        duration = getattr(generation, "duration", None)

        # Use Hume's request_id if available, otherwise use our generated one
        result_request_id = getattr(hume_result, "request_id", None) or request_id

        # Build raw_response with available metadata
        raw_response: dict[str, Any] = {
            "model": config.model,
            "generation_id": str(generation.generation_id),
            "output_format": request.output_format.format,
        }

        file_size = getattr(generation, "file_size", None)
        if file_size is not None:
            raw_response["file_size"] = file_size

        encoding = getattr(generation, "encoding", None)
        if encoding:
            raw_response["encoding"] = {
                "format": getattr(encoding, "format", None),
                "sample_rate": getattr(encoding, "sample_rate", None),
            }

        cost = GenerationCost.from_pricing_table(
            config.provider, config.model, len(request.text)
        )

        return TTSResponse(
            request_id=result_request_id,
            audio=audio_b64,
            content_type=content_type,
            duration=duration,
            status="completed",
            cost=cost,
            raw_response=raw_response,
        )

    def _handle_error(
        self,
        config: AudioGenerationConfig,
        request_id: str,
        ex: Exception,
    ) -> TarashException:
        """Map Hume errors to TarashException hierarchy."""
        provider = config.provider
        model = config.model

        # Check timeout BEFORE connection (timeout is subclass of connection)
        if isinstance(ex, httpx.TimeoutException):
            return TimeoutError(
                f"Request timed out: {ex}",
                provider=provider,
                model=model,
                request_id=request_id,
                timeout_seconds=config.timeout,
            )

        if isinstance(ex, httpx.ConnectError):
            return HTTPConnectionError(
                f"Connection error: {ex}",
                provider=provider,
                model=model,
                request_id=request_id,
            )

        if isinstance(ex, HumeApiError):
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
            elif status_code and status_code >= 500:
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
        | None = None,  # Unused: Hume returns complete audio in a single response
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
                "request_id": request_id,
            },
            logger_name=_LOGGER_NAME,
        )

        try:
            result = await client.tts.synthesize_json(**kwargs)

            log_info(
                "TTS generation completed (async)",
                context={"request_id": request_id},
                logger_name=_LOGGER_NAME,
            )

            return self._convert_tts_response(config, request, request_id, result)
        except (TarashException, Exception) as ex:
            if isinstance(ex, TarashException):
                raise
            raise self._handle_error(config, request_id, ex)

    @handle_audio_generation_errors
    def generate_tts(
        self,
        config: AudioGenerationConfig,
        request: TTSRequest,
        on_progress: TTSProgressCallback
        | None = None,  # Unused: Hume returns complete audio in a single response
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
                "request_id": request_id,
            },
            logger_name=_LOGGER_NAME,
        )

        try:
            result = client.tts.synthesize_json(**kwargs)

            log_info(
                "TTS generation completed (sync)",
                context={"request_id": request_id},
                logger_name=_LOGGER_NAME,
            )

            return self._convert_tts_response(config, request, request_id, result)
        except (TarashException, Exception) as ex:
            if isinstance(ex, TarashException):
                raise
            raise self._handle_error(config, request_id, ex)
