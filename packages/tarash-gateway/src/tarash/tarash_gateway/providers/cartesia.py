"""Cartesia provider handler for TTS and STS (Voice Changer) audio generation."""

import base64
import io
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


def _parse_output_format(output_format: str | None) -> dict[str, Any]:
    """Parse output format string to Cartesia structured dict.

    Handles full, partial, or bare container strings — missing parts get defaults.

    Examples:
        "mp3_44100_128" -> {"container": "mp3", "sample_rate": 44100, "bit_rate": 128000}
        "mp3_22050"     -> {"container": "mp3", "sample_rate": 22050, "bit_rate": 128000}
        "mp3"           -> {"container": "mp3", "sample_rate": 44100, "bit_rate": 128000}
        "wav_16000"     -> {"container": "wav", "encoding": "pcm_s16le", "sample_rate": 16000}
        "wav"           -> {"container": "wav", "encoding": "pcm_s16le", "sample_rate": 44100}
        "pcm_16000"     -> {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 16000}
        "pcm"           -> {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 44100}
        None            -> {"container": "mp3", "sample_rate": 44100, "bit_rate": 128000} (default)
    """
    if not output_format:
        return {
            "container": "mp3",
            "sample_rate": _DEFAULT_SAMPLE_RATE,
            "bit_rate": _DEFAULT_BIT_RATE,
        }

    parts = output_format.split("_")
    container = parts[0]

    sample_rate = int(parts[1]) if len(parts) >= 2 else _DEFAULT_SAMPLE_RATE

    if container == "mp3":
        bit_rate = int(parts[2]) * 1000 if len(parts) >= 3 else _DEFAULT_BIT_RATE
        return {"container": "mp3", "sample_rate": sample_rate, "bit_rate": bit_rate}
    elif container == "wav":
        return {
            "container": "wav",
            "encoding": _DEFAULT_ENCODING,
            "sample_rate": sample_rate,
        }
    elif container == "pcm":
        return {
            "container": "raw",
            "encoding": _DEFAULT_ENCODING,
            "sample_rate": sample_rate,
        }
    else:
        # Unknown container — pass through as-is, let Cartesia API validate
        return {"container": container, "sample_rate": sample_rate}


def _output_format_to_content_type(output_format: str | None) -> str:
    """Map output format string to MIME content type."""
    if not output_format:
        return "audio/mpeg"
    if output_format.startswith("mp3"):
        return "audio/mpeg"
    if output_format.startswith("wav"):
        return "audio/wav"
    if output_format.startswith("pcm"):
        return "audio/pcm"
    return "audio/mpeg"


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
        kwargs: dict[str, Any] = {
            "transcript": request.text,
            "voice": {"mode": "id", "id": request.voice_id},
            "model_id": config.model,
            "output_format": _parse_output_format(request.output_format),
        }

        if request.language_code:
            kwargs["language"] = request.language_code

        # Merge extra_params (generation_config, pronunciation_dict_id, etc.)
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
            duration=None,  # Cartesia API does not return duration metadata
            status="completed",
            raw_response={
                "audio_size_bytes": len(audio_bytes),
                "output_format": request.output_format,
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
        Voice Changer uses flat parameters, not nested output_format dict.
        """
        parsed_format = _parse_output_format(request.output_format)

        kwargs: dict[str, Any] = {
            "voice_id": request.voice_id,
            "output_format_container": parsed_format["container"],
            "output_format_sample_rate": parsed_format["sample_rate"],
        }

        if "encoding" in parsed_format:
            kwargs["output_format_encoding"] = parsed_format["encoding"]
        if "bit_rate" in parsed_format:
            kwargs["output_format_bit_rate"] = parsed_format["bit_rate"]

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
        content_type = _output_format_to_content_type(request.output_format)

        return STSResponse(
            request_id=request_id,
            audio=audio_b64,
            content_type=content_type,
            duration=None,  # Cartesia API does not return duration metadata
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
                "output_format": request.output_format,
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
                "output_format": request.output_format,
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
                "output_format": request.output_format,
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
                "output_format": request.output_format,
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
