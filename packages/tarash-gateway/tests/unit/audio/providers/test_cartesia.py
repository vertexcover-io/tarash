"""Unit tests for Cartesia TTS/STS provider handler."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tarash.tarash_gateway.models import (
    AudioGenerationConfig,
    STSRequest,
    TTSRequest,
    TTSResponse,
    STSResponse,
)
from tarash.tarash_gateway.exceptions import (
    ContentModerationError,
    HTTPConnectionError,
    HTTPError,
    TarashException,
    TimeoutError,
    ValidationError,
)
from tarash.tarash_gateway.providers.cartesia import (
    CartesiaProviderHandler,
    _generate_request_id,
    _parse_output_format,
    _output_format_to_content_type,
)


# ==================== Fixtures ====================


@pytest.fixture
def handler():
    return CartesiaProviderHandler()


@pytest.fixture
def base_config():
    return AudioGenerationConfig(
        model="sonic-3",
        provider="cartesia",
        api_key="test-api-key",
        timeout=240,
    )


@pytest.fixture
def tts_request():
    return TTSRequest(text="Hello world", voice_id="test-voice-id")


@pytest.fixture
def sts_request():
    return STSRequest(
        audio={"content": b"fake-audio-bytes", "content_type": "audio/wav"},
        voice_id="test-voice-id",
    )


# ==================== Request ID Generation ====================


def test_generate_request_id_is_hex_string():
    rid = _generate_request_id()
    assert len(rid) == 32
    assert all(c in "0123456789abcdef" for c in rid)


def test_generate_request_id_unique():
    ids = {_generate_request_id() for _ in range(100)}
    assert len(ids) == 100


# ==================== Output Format Parsing ====================


@pytest.mark.parametrize(
    ("format_str", "expected"),
    [
        # Full strings (ElevenLabs-style)
        (
            "mp3_44100_128",
            {"container": "mp3", "sample_rate": 44100, "bit_rate": 128000},
        ),
        ("mp3_22050_32", {"container": "mp3", "sample_rate": 22050, "bit_rate": 32000}),
        (
            "wav_44100",
            {"container": "wav", "encoding": "pcm_s16le", "sample_rate": 44100},
        ),
        (
            "wav_16000",
            {"container": "wav", "encoding": "pcm_s16le", "sample_rate": 16000},
        ),
        (
            "pcm_16000",
            {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 16000},
        ),
        (
            "pcm_44100",
            {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 44100},
        ),
        # Partial strings (sample_rate only, defaults filled in)
        ("mp3_22050", {"container": "mp3", "sample_rate": 22050, "bit_rate": 128000}),
        # Bare container strings (all defaults)
        ("mp3", {"container": "mp3", "sample_rate": 44100, "bit_rate": 128000}),
        ("wav", {"container": "wav", "encoding": "pcm_s16le", "sample_rate": 44100}),
        ("pcm", {"container": "raw", "encoding": "pcm_s16le", "sample_rate": 44100}),
        # None -> default
        (None, {"container": "mp3", "sample_rate": 44100, "bit_rate": 128000}),
        # Unrecognized -> default
        (
            "opus_48000_64",
            {"container": "mp3", "sample_rate": 44100, "bit_rate": 128000},
        ),
    ],
)
def test_parse_output_format(format_str, expected):
    assert _parse_output_format(format_str) == expected


# ==================== Content Type Mapping ====================


@pytest.mark.parametrize(
    ("output_format", "expected_content_type"),
    [
        ("mp3_44100_128", "audio/mpeg"),
        ("mp3", "audio/mpeg"),
        ("wav_44100", "audio/wav"),
        ("wav", "audio/wav"),
        ("pcm_16000", "audio/pcm"),
        ("pcm", "audio/pcm"),
        (None, "audio/mpeg"),
    ],
)
def test_output_format_to_content_type(output_format, expected_content_type):
    assert _output_format_to_content_type(output_format) == expected_content_type


# ==================== TTS Request Conversion ====================


def test_convert_tts_request_minimal(handler, base_config, tts_request):
    kwargs = handler._convert_tts_request(base_config, tts_request)

    assert kwargs["transcript"] == "Hello world"
    assert kwargs["voice"] == {"mode": "id", "id": "test-voice-id"}
    assert kwargs["model_id"] == "sonic-3"
    assert kwargs["output_format"] == {
        "container": "mp3",
        "sample_rate": 44100,
        "bit_rate": 128000,
    }
    assert "language" not in kwargs
    assert "generation_config" not in kwargs


def test_convert_tts_request_full(handler, base_config):
    request = TTSRequest(
        text="Bonjour le monde",
        voice_id="test-voice-id",
        output_format="wav_44100",
        language_code="fr",
        extra_params={
            "generation_config": {"emotion": "happy", "speed": 1.2, "volume": 0.8},
            "pronunciation_dict_id": "dict-123",
        },
    )
    kwargs = handler._convert_tts_request(base_config, request)

    assert kwargs["transcript"] == "Bonjour le monde"
    assert kwargs["language"] == "fr"
    assert kwargs["output_format"] == {
        "container": "wav",
        "encoding": "pcm_s16le",
        "sample_rate": 44100,
    }
    assert kwargs["generation_config"] == {
        "emotion": "happy",
        "speed": 1.2,
        "volume": 0.8,
    }
    assert kwargs["pronunciation_dict_id"] == "dict-123"


def test_convert_tts_request_extra_params(handler, base_config):
    request = TTSRequest(
        text="Hello",
        voice_id="v1",
        extra_params={"custom_param": "value"},
    )
    kwargs = handler._convert_tts_request(base_config, request)
    assert kwargs["custom_param"] == "value"


# ==================== TTS Response Conversion ====================


def test_convert_tts_response(handler, base_config, tts_request):
    audio_bytes = b"fake-audio-output"
    response = handler._convert_tts_response(
        base_config, tts_request, "req-123", audio_bytes
    )

    assert isinstance(response, TTSResponse)
    assert response.request_id == "req-123"
    assert response.audio == base64.b64encode(audio_bytes).decode("utf-8")
    assert response.content_type == "audio/mpeg"
    assert response.status == "completed"
    assert response.raw_response["audio_size_bytes"] == len(audio_bytes)
    assert response.raw_response["model"] == "sonic-3"
    assert response.raw_response["voice_id"] == "test-voice-id"


def test_convert_tts_response_wav_format(handler, base_config):
    request = TTSRequest(text="test", voice_id="v1", output_format="wav_44100")
    response = handler._convert_tts_response(
        base_config, request, "req-456", b"wav-data"
    )

    assert response.content_type == "audio/wav"


# ==================== Audio Resolution ====================


def test_resolve_audio_bytes_media_content(handler):
    audio = {"content": b"audio-data", "content_type": "audio/wav"}
    result = handler._resolve_audio_bytes(audio)
    assert result.read() == b"audio-data"


def test_resolve_audio_bytes_base64(handler):
    audio_b64 = base64.b64encode(b"audio-data").decode()
    result = handler._resolve_audio_bytes(audio_b64)
    assert result.read() == b"audio-data"


def test_resolve_audio_bytes_url(handler):
    with patch(
        "tarash.tarash_gateway.providers.cartesia.download_media_from_url",
        return_value=(b"downloaded-audio", "audio/mpeg"),
    ) as mock_download:
        result = handler._resolve_audio_bytes("https://example.com/audio.mp3")
        assert result.read() == b"downloaded-audio"
        mock_download.assert_called_once_with(
            "https://example.com/audio.mp3", provider="cartesia"
        )


@pytest.mark.asyncio
async def test_resolve_audio_bytes_async_url(handler):
    with patch(
        "tarash.tarash_gateway.providers.cartesia.download_media_from_url_async",
        return_value=(b"downloaded-audio", "audio/mpeg"),
    ) as mock_download:
        result = await handler._resolve_audio_bytes_async(
            "https://example.com/audio.mp3"
        )
        assert result.read() == b"downloaded-audio"
        mock_download.assert_called_once_with(
            "https://example.com/audio.mp3", provider="cartesia"
        )


# ==================== STS Request Conversion ====================


def test_convert_sts_request_basic(handler, base_config, sts_request):
    kwargs = handler._convert_sts_request(base_config, sts_request)

    assert kwargs["voice_id"] == "test-voice-id"
    assert kwargs["output_format_container"] == "mp3"
    assert kwargs["output_format_sample_rate"] == 44100
    assert kwargs["output_format_bit_rate"] == 128000
    assert "clip" not in kwargs  # Audio handled separately


def test_convert_sts_request_wav_format(handler, base_config):
    request = STSRequest(
        audio={"content": b"data", "content_type": "audio/wav"},
        voice_id="v1",
        output_format="wav_44100",
    )
    kwargs = handler._convert_sts_request(base_config, request)

    assert kwargs["output_format_container"] == "wav"
    assert kwargs["output_format_sample_rate"] == 44100
    assert kwargs["output_format_encoding"] == "pcm_s16le"
    assert "output_format_bit_rate" not in kwargs


# ==================== STS Response Conversion ====================


def test_convert_sts_response(handler, base_config, sts_request):
    audio_bytes = b"fake-sts-output"
    response = handler._convert_sts_response(
        base_config, sts_request, "req-789", audio_bytes
    )

    assert isinstance(response, STSResponse)
    assert response.request_id == "req-789"
    assert response.audio == base64.b64encode(audio_bytes).decode("utf-8")
    assert response.content_type == "audio/mpeg"
    assert response.status == "completed"


# ==================== Error Handling ====================


def _make_cartesia_status_error(status_code, message="error"):
    """Create a mock Cartesia API status error."""
    import httpx
    from cartesia._exceptions import (
        BadRequestError,
        AuthenticationError,
        PermissionDeniedError,
        UnprocessableEntityError,
        RateLimitError,
        InternalServerError,
    )

    mock_response = MagicMock(spec=httpx.Response)
    mock_response.status_code = status_code
    mock_response.json.return_value = {"error": message}
    mock_response.headers = {}
    mock_response.text = message

    error_map = {
        400: BadRequestError,
        401: AuthenticationError,
        403: PermissionDeniedError,
        422: UnprocessableEntityError,
        429: RateLimitError,
        500: InternalServerError,
    }

    error_cls = error_map.get(status_code)
    if error_cls:
        return error_cls(
            message=message, response=mock_response, body={"error": message}
        )
    raise ValueError(f"Unsupported status code for test helper: {status_code}")


def test_handle_error_400_validation(handler, base_config, tts_request):
    ex = _make_cartesia_status_error(400, "Bad request")
    result = handler._handle_error(base_config, tts_request, "req-1", ex)
    assert isinstance(result, ValidationError)
    assert result.provider == "cartesia"
    assert result.model == "sonic-3"


def test_handle_error_401_auth(handler, base_config, tts_request):
    ex = _make_cartesia_status_error(401, "Unauthorized")
    result = handler._handle_error(base_config, tts_request, "req-1", ex)
    assert isinstance(result, HTTPError)
    assert result.status_code == 401


def test_handle_error_403_moderation(handler, base_config, tts_request):
    ex = _make_cartesia_status_error(403, "Forbidden")
    result = handler._handle_error(base_config, tts_request, "req-1", ex)
    assert isinstance(result, ContentModerationError)


def test_handle_error_422_validation(handler, base_config, tts_request):
    ex = _make_cartesia_status_error(422, "Unprocessable")
    result = handler._handle_error(base_config, tts_request, "req-1", ex)
    assert isinstance(result, ValidationError)


def test_handle_error_429_rate_limit(handler, base_config, tts_request):
    ex = _make_cartesia_status_error(429, "Rate limited")
    result = handler._handle_error(base_config, tts_request, "req-1", ex)
    assert isinstance(result, HTTPError)
    assert result.status_code == 429


def test_handle_error_500_server(handler, base_config, tts_request):
    ex = _make_cartesia_status_error(500, "Server error")
    result = handler._handle_error(base_config, tts_request, "req-1", ex)
    assert isinstance(result, HTTPError)
    assert result.status_code == 500


def test_handle_error_connection_error(handler, base_config, tts_request):
    from cartesia._exceptions import APIConnectionError as CartesiaConnError

    ex = CartesiaConnError(request=MagicMock())
    result = handler._handle_error(base_config, tts_request, "req-1", ex)
    assert isinstance(result, HTTPConnectionError)


def test_handle_error_timeout(handler, base_config, tts_request):
    from cartesia._exceptions import APITimeoutError as CartesiaTimeout

    ex = CartesiaTimeout(request=MagicMock())
    result = handler._handle_error(base_config, tts_request, "req-1", ex)
    assert isinstance(result, TimeoutError)


def test_handle_error_unknown(handler, base_config, tts_request):
    ex = RuntimeError("Something unexpected")
    result = handler._handle_error(base_config, tts_request, "req-1", ex)
    assert isinstance(result, TarashException)
    assert "Something unexpected" in result.message


# ==================== Integration Tests (Mocked SDK) ====================


def test_generate_tts_sync_success(handler, base_config, tts_request):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.iter_bytes.return_value = iter([b"chunk1", b"chunk2"])
    mock_client.tts.generate.return_value = mock_response

    with patch.object(handler, "_get_client", return_value=mock_client):
        response = handler.generate_tts(base_config, tts_request)

    assert isinstance(response, TTSResponse)
    assert response.status == "completed"
    expected_audio = base64.b64encode(b"chunk1chunk2").decode("utf-8")
    assert response.audio == expected_audio
    assert response.content_type == "audio/mpeg"
    assert len(response.request_id) == 32


@pytest.mark.asyncio
async def test_generate_tts_async_success(handler, base_config, tts_request):
    mock_client = MagicMock()
    mock_response = MagicMock()

    async def mock_iter_bytes():
        yield b"async_chunk1"
        yield b"async_chunk2"

    mock_response.iter_bytes.return_value = mock_iter_bytes()
    mock_client.tts.generate = AsyncMock(return_value=mock_response)

    with patch.object(handler, "_get_client", return_value=mock_client):
        response = await handler.generate_tts_async(base_config, tts_request)

    assert isinstance(response, TTSResponse)
    assert response.status == "completed"
    expected_audio = base64.b64encode(b"async_chunk1async_chunk2").decode("utf-8")
    assert response.audio == expected_audio


@pytest.mark.asyncio
async def test_generate_tts_async_api_error(handler, base_config, tts_request):
    mock_client = MagicMock()
    mock_client.tts.generate = AsyncMock(
        side_effect=_make_cartesia_status_error(400, "Bad request")
    )

    with patch.object(handler, "_get_client", return_value=mock_client):
        with pytest.raises(ValidationError, match="Invalid request"):
            await handler.generate_tts_async(base_config, tts_request)


def test_generate_sts_sync_success(handler, base_config, sts_request):
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.iter_bytes.return_value = iter([b"sts_chunk1", b"sts_chunk2"])
    mock_client.voice_changer.change_voice_bytes.return_value = mock_response

    with patch.object(handler, "_get_client", return_value=mock_client):
        response = handler.generate_sts(base_config, sts_request)

    assert isinstance(response, STSResponse)
    assert response.status == "completed"
    expected_audio = base64.b64encode(b"sts_chunk1sts_chunk2").decode("utf-8")
    assert response.audio == expected_audio


@pytest.mark.asyncio
async def test_generate_sts_async_success(handler, base_config, sts_request):
    mock_client = MagicMock()
    mock_response = MagicMock()

    async def mock_iter_bytes():
        yield b"sts_async1"
        yield b"sts_async2"

    mock_response.iter_bytes.return_value = mock_iter_bytes()
    mock_client.voice_changer.change_voice_bytes = AsyncMock(return_value=mock_response)

    with patch.object(handler, "_get_client", return_value=mock_client):
        response = await handler.generate_sts_async(base_config, sts_request)

    assert isinstance(response, STSResponse)
    assert response.status == "completed"


# ==================== Client Creation ====================


def test_get_client_sync(handler, base_config):
    with patch("tarash.tarash_gateway.providers.cartesia.Cartesia") as mock_cls:
        handler._get_client(base_config, "sync")
        mock_cls.assert_called_once_with(api_key="test-api-key", timeout=240)


def test_get_client_async(handler, base_config):
    with patch("tarash.tarash_gateway.providers.cartesia.AsyncCartesia") as mock_cls:
        handler._get_client(base_config, "async")
        mock_cls.assert_called_once_with(api_key="test-api-key", timeout=240)


def test_get_client_no_api_key(handler):
    config = AudioGenerationConfig(model="sonic-3", provider="cartesia")
    with patch("tarash.tarash_gateway.providers.cartesia.Cartesia") as mock_cls:
        handler._get_client(config, "sync")
        mock_cls.assert_called_once_with(timeout=240)
