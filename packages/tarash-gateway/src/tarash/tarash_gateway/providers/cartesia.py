"""Cartesia provider handler for TTS and STS (Voice Changer) audio generation."""

import base64
import io
import uuid
from typing import Any, Literal

from tarash.tarash_gateway.logging import log_info, log_warning
from tarash.tarash_gateway.pricing import resolve_cost
from tarash.tarash_gateway.providers.field_mappers import (
    FieldMapper,
    apply_field_mappers,
    passthrough_field_mapper,
)
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
    AudioOutputFormat,
    STSProgressCallback,
    STSRequest,
    STSResponse,
    TTSProgressCallback,
    TTSRequest,
    TTSResponse,
    format_to_content_type,
)

has_cartesia = True
try:
    from cartesia import AsyncCartesia, Cartesia
    from cartesia._exceptions import (
        APIConnectionError as CartesiaConnectionError,
        APITimeoutError as CartesiaTimeoutError,
        APIStatusError as CartesiaAPIStatusError,
        BadRequestError as CartesiaBadRequestError,
        AuthenticationError as CartesiaAuthenticationError,
        PermissionDeniedError as CartesiaPermissionDeniedError,
        UnprocessableEntityError as CartesiaUnprocessableEntityError,
        RateLimitError as CartesiaRateLimitError,
        InternalServerError as CartesiaInternalServerError,
    )
except ImportError:
    has_cartesia = False

_LOGGER_NAME = "tarash.tarash_gateway.providers.cartesia"


def _generate_request_id() -> str:
    """Generate a unique request ID."""
    return uuid.uuid4().hex


_DEFAULT_SAMPLE_RATE = 44100
_DEFAULT_BIT_RATE = 128000
_DEFAULT_ENCODING = "pcm_s16le"


def _output_format_to_cartesia_dict(
    _request: object, output_format: AudioOutputFormat
) -> dict[str, Any]:
    """Convert AudioOutputFormat to Cartesia structured dict.

    Maps format -> container (with pcm -> raw), adds encoding for wav/pcm,
    and converts bitrate kbps -> bit_rate bps. Logs warning if bitrate
    is set for wav/pcm (unsupported by Cartesia).
    """
    fmt = output_format.format
    container = "raw" if fmt == "pcm" else fmt
    sample_rate = output_format.sample_rate or _DEFAULT_SAMPLE_RATE

    result: dict[str, Any] = {"container": container, "sample_rate": sample_rate}

    if fmt in ("wav", "pcm"):
        result["encoding"] = _DEFAULT_ENCODING
        if output_format.bitrate is not None:
            log_warning(
                f"Cartesia: bitrate is not supported for '{fmt}' format, ignoring",
                logger_name=_LOGGER_NAME,
            )
    elif output_format.bitrate is not None:
        result["bit_rate"] = output_format.bitrate * 1000
    else:
        result["bit_rate"] = _DEFAULT_BIT_RATE

    return result


def _cartesia_voice_converter(
    _request: TTSRequest, value: object
) -> dict[str, str] | None:
    """Convert voice_id to Cartesia voice spec dict."""
    if value is None:
        return None
    return {"mode": "id", "id": str(value)}


def _sts_bit_rate_converter(_request: STSRequest, output_format: object) -> int | None:
    """Extract bit_rate for STS flat params, returning None when not applicable."""
    if not isinstance(output_format, AudioOutputFormat):
        return None
    fmt = output_format.format
    if fmt in ("wav", "pcm"):
        if output_format.bitrate is not None:
            log_warning(
                f"Cartesia: bitrate is not supported for '{fmt}' format, ignoring",
                logger_name=_LOGGER_NAME,
            )
        return None
    if output_format.bitrate is not None:
        return output_format.bitrate * 1000
    return _DEFAULT_BIT_RATE


CARTESIA_TTS_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "transcript": passthrough_field_mapper("text", required=True),
    "voice": FieldMapper(
        source_field="voice_id",
        converter=_cartesia_voice_converter,
        required=True,
    ),
    "output_format": FieldMapper(
        source_field="output_format",
        converter=_output_format_to_cartesia_dict,
    ),
    "language": passthrough_field_mapper("language_code"),
}

CARTESIA_STS_FIELD_MAPPERS: dict[str, FieldMapper] = {
    "voice_id": passthrough_field_mapper("voice_id", required=True),
    "output_format_container": FieldMapper(
        source_field="output_format",
        converter=lambda _req, val: ("raw" if val.format == "pcm" else val.format)
        if isinstance(val, AudioOutputFormat)
        else None,
    ),
    "output_format_sample_rate": FieldMapper(
        source_field="output_format",
        converter=lambda _req, val: (val.sample_rate or _DEFAULT_SAMPLE_RATE)
        if isinstance(val, AudioOutputFormat)
        else _DEFAULT_SAMPLE_RATE,
    ),
    "output_format_encoding": FieldMapper(
        source_field="output_format",
        converter=lambda _req, val: _DEFAULT_ENCODING
        if isinstance(val, AudioOutputFormat) and val.format in ("wav", "pcm")
        else None,
    ),
    "output_format_bit_rate": FieldMapper(
        source_field="output_format",
        converter=_sts_bit_rate_converter,
    ),
}


class CartesiaProviderHandler:
    """Handler for Cartesia TTS and STS (Voice Changer) audio generation."""

    def __init__(self) -> None:
        if not has_cartesia:
            raise ImportError(
                "cartesia is required for Cartesia provider. "
                "Install with: pip install tarash-gateway[cartesia]"
            )

    def _get_client(
        self, config: AudioGenerationConfig, client_type: Literal["sync", "async"]
    ) -> Any:
        """Create a fresh Cartesia client.

        Fresh client each call to avoid event loop issues with async client.
        """
        kwargs: dict[str, Any] = {}
        if config.api_key:
            kwargs["api_key"] = config.api_key
        kwargs["timeout"] = config.timeout

        if client_type == "async":
            return AsyncCartesia(**kwargs)
        return Cartesia(**kwargs)

    def _convert_tts_request(
        self, config: AudioGenerationConfig, request: TTSRequest
    ) -> dict[str, Any]:
        """Convert TTSRequest to Cartesia SDK kwargs."""
        kwargs = apply_field_mappers(CARTESIA_TTS_FIELD_MAPPERS, request)
        kwargs["model_id"] = config.model
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
        content_type = format_to_content_type(request.output_format.format)
        cost = resolve_cost(config.provider, config.model, None, len(request.text))

        return TTSResponse(
            request_id=request_id,
            audio=audio_b64,
            content_type=content_type,
            duration=None,  # Cartesia API does not return duration metadata
            status="completed",
            cost=cost,
            raw_response={
                "audio_size_bytes": len(audio_bytes),
                "output_format": request.output_format.model_dump(),
                "model": config.model,
                "voice_id": request.voice_id,
            },
        )

    @staticmethod
    def _resolve_audio_local(audio: Any) -> io.BytesIO | None:
        """Resolve audio from local sources (dict, base64, raw bytes).

        Returns None for URLs that need downloading.
        """
        if isinstance(audio, dict) and "content" in audio:
            return io.BytesIO(audio["content"])
        if isinstance(audio, str):
            if audio.startswith(("http://", "https://")):
                return None
            return io.BytesIO(base64.b64decode(audio))
        return io.BytesIO(audio)

    def _resolve_audio_bytes(self, audio: Any) -> io.BytesIO:
        """Convert MediaType audio input to BytesIO for the SDK (sync)."""
        result = self._resolve_audio_local(audio)
        if result is not None:
            return result
        content, _ = download_media_from_url(audio, provider="cartesia")
        return io.BytesIO(content)

    async def _resolve_audio_bytes_async(self, audio: Any) -> io.BytesIO:
        """Convert MediaType audio input to BytesIO for the SDK (async)."""
        result = self._resolve_audio_local(audio)
        if result is not None:
            return result
        content, _ = await download_media_from_url_async(audio, provider="cartesia")
        return io.BytesIO(content)

    def _convert_sts_request(self, request: STSRequest) -> dict[str, Any]:
        """Convert STSRequest to Cartesia Voice Changer SDK kwargs (without clip).

        Clip (audio) is resolved separately via _resolve_audio_bytes / _resolve_audio_bytes_async.
        Voice Changer uses flat output_format_* parameters — multiple FieldMappers
        share the same source_field="output_format" to fan out into separate API keys.
        """
        kwargs = apply_field_mappers(CARTESIA_STS_FIELD_MAPPERS, request)
        kwargs.update(request.extra_params)
        return kwargs

    def _convert_sts_response(
        self,
        config: AudioGenerationConfig,
        request: STSRequest,
        request_id: str,
        audio_bytes: bytes,
    ) -> STSResponse:
        """Convert raw audio bytes to STSResponse."""
        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")
        content_type = format_to_content_type(request.output_format.format)

        return STSResponse(
            request_id=request_id,
            audio=audio_b64,
            content_type=content_type,
            duration=None,  # Cartesia API does not return duration metadata
            status="completed",
            raw_response={
                "audio_size_bytes": len(audio_bytes),
                "output_format": request.output_format.model_dump(),
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
        """Map Cartesia errors to TarashException hierarchy."""
        provider = config.provider
        model = config.model

        # Check timeout BEFORE connection (timeout is subclass of connection)
        if isinstance(ex, CartesiaTimeoutError):
            return TimeoutError(
                f"Request timed out: {ex}",
                provider=provider,
                model=model,
                request_id=request_id,
                timeout_seconds=config.timeout,
            )

        if isinstance(ex, CartesiaConnectionError):
            return HTTPConnectionError(
                f"Connection error: {ex}",
                provider=provider,
                model=model,
                request_id=request_id,
            )

        if isinstance(ex, CartesiaAPIStatusError):
            status_code = ex.status_code
            body = str(ex.body) if ex.body else str(ex)

            if isinstance(
                ex, (CartesiaBadRequestError, CartesiaUnprocessableEntityError)
            ):
                return ValidationError(
                    f"Invalid request: {body}",
                    provider=provider,
                    model=model,
                    request_id=request_id,
                    raw_response={"error": body, "status_code": status_code},
                )
            elif isinstance(ex, CartesiaAuthenticationError):
                return HTTPError(
                    f"Authentication failed: {body}",
                    provider=provider,
                    model=model,
                    request_id=request_id,
                    raw_response={"error": body, "status_code": status_code},
                    status_code=status_code,
                )
            elif isinstance(ex, CartesiaPermissionDeniedError):
                return ContentModerationError(
                    f"Content policy violation: {body}",
                    provider=provider,
                    model=model,
                    request_id=request_id,
                    raw_response={"error": body, "status_code": status_code},
                )
            elif isinstance(ex, CartesiaRateLimitError):
                return HTTPError(
                    f"Rate limit exceeded: {body}",
                    provider=provider,
                    model=model,
                    request_id=request_id,
                    raw_response={"error": body, "status_code": status_code},
                    status_code=status_code,
                )
            elif isinstance(ex, CartesiaInternalServerError):
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
        | None = None,  # Unused: Cartesia bytes endpoint returns complete audio
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
                "output_format": request.output_format.model_dump(),
                "request_id": request_id,
            },
            logger_name=_LOGGER_NAME,
        )

        try:
            response = await client.tts.generate(**kwargs)
            audio_bytes = b"".join([chunk async for chunk in response.iter_bytes()])

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
        | None = None,  # Unused: Cartesia bytes endpoint returns complete audio
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
                "output_format": request.output_format.model_dump(),
                "request_id": request_id,
            },
            logger_name=_LOGGER_NAME,
        )

        try:
            response = client.tts.generate(**kwargs)
            audio_bytes = b"".join(response.iter_bytes())

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

    # ==================== STS Generation (Voice Changer) ====================

    @handle_audio_generation_errors
    async def generate_sts_async(
        self,
        config: AudioGenerationConfig,
        request: STSRequest,
        on_progress: STSProgressCallback
        | None = None,  # Unused: Cartesia bytes endpoint returns complete audio
    ) -> STSResponse:
        """Convert speech to speech asynchronously using Voice Changer."""
        client = self._get_client(config, "async")
        kwargs = self._convert_sts_request(request)
        kwargs["clip"] = await self._resolve_audio_bytes_async(request.audio)
        request_id = _generate_request_id()

        log_info(
            "Starting STS generation (async)",
            context={
                "model": config.model,
                "voice_id": request.voice_id,
                "output_format": request.output_format.model_dump(),
                "request_id": request_id,
            },
            logger_name=_LOGGER_NAME,
        )

        try:
            response = await client.voice_changer.change_voice_bytes(**kwargs)
            audio_bytes = b"".join([chunk async for chunk in response.iter_bytes()])

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
        | None = None,  # Unused: Cartesia bytes endpoint returns complete audio
    ) -> STSResponse:
        """Convert speech to speech synchronously using Voice Changer."""
        client = self._get_client(config, "sync")
        kwargs = self._convert_sts_request(request)
        kwargs["clip"] = self._resolve_audio_bytes(request.audio)
        request_id = _generate_request_id()

        log_info(
            "Starting STS generation (sync)",
            context={
                "model": config.model,
                "voice_id": request.voice_id,
                "output_format": request.output_format.model_dump(),
                "request_id": request_id,
            },
            logger_name=_LOGGER_NAME,
        )

        try:
            response = client.voice_changer.change_voice_bytes(**kwargs)
            audio_bytes = b"".join(response.iter_bytes())

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
