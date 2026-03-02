"""Unit tests for ElevenLabs provider handler."""

import base64
from unittest.mock import MagicMock, patch

import pytest

from tarash.tarash_gateway.exceptions import (
    ContentModerationError,
    HTTPConnectionError,
    HTTPError,
    TarashException,
    TimeoutError,
    ValidationError,
)
from tarash.tarash_gateway.models import (
    AudioGenerationConfig,
    AudioOutputFormat,
    STSRequest,
    STSResponse,
    TTSRequest,
    TTSResponse,
    format_to_content_type,
)
from tarash.tarash_gateway.providers.elevenlabs import (
    ElevenLabsProviderHandler,
    _generate_request_id,
    _output_format_to_elevenlabs_string,
)


# ==================== Fixtures ====================


@pytest.fixture
def handler():
    return ElevenLabsProviderHandler()


@pytest.fixture
def base_config():
    return AudioGenerationConfig(
        model="eleven_multilingual_v2",
        provider="elevenlabs",
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
    """Request IDs are 32-char hex strings."""
    rid = _generate_request_id()
    assert len(rid) == 32
    assert all(c in "0123456789abcdef" for c in rid)


def test_generate_request_id_unique():
    """Each call generates a unique ID."""
    ids = {_generate_request_id() for _ in range(100)}
    assert len(ids) == 100


# ==================== Content Type Mapping ====================


@pytest.mark.parametrize(
    ("fmt", "expected_content_type"),
    [
        ("mp3", "audio/mpeg"),
        ("pcm", "audio/pcm"),
        ("wav", "audio/wav"),
        ("opus", "audio/opus"),
        ("ulaw", "audio/basic"),
        ("alaw", "audio/basic"),
        ("flac", "audio/flac"),
        ("unknown", "audio/mpeg"),
    ],
)
def test_format_to_content_type(fmt, expected_content_type):
    """Format strings map to correct MIME types."""
    assert format_to_content_type(fmt) == expected_content_type


# ==================== ElevenLabs String Conversion ====================


@pytest.mark.parametrize(
    ("output_format", "expected_string"),
    [
        (
            AudioOutputFormat(format="mp3", sample_rate=44100, bitrate=128),
            "mp3_44100_128",
        ),
        (
            AudioOutputFormat(format="mp3", sample_rate=22050, bitrate=32),
            "mp3_22050_32",
        ),
        (AudioOutputFormat(format="pcm", sample_rate=16000), "pcm_16000"),
        (AudioOutputFormat(format="wav", sample_rate=44100), "wav_44100"),
        (
            AudioOutputFormat(format="opus", sample_rate=48000, bitrate=64),
            "opus_48000_64",
        ),
        (AudioOutputFormat(format="mp3"), "mp3"),
    ],
)
def test_output_format_to_elevenlabs_string(output_format, expected_string):
    """AudioOutputFormat converts to ElevenLabs SDK format string."""
    assert _output_format_to_elevenlabs_string(None, output_format) == expected_string


# ==================== TTS Request Conversion ====================


def test_convert_tts_request_minimal(handler, base_config, tts_request):
    """Minimal TTS request includes text, voice_id, model_id, output_format."""
    kwargs = handler._convert_tts_request(base_config, tts_request)

    assert kwargs["text"] == "Hello world"
    assert kwargs["voice_id"] == "test-voice-id"
    assert kwargs["model_id"] == "eleven_multilingual_v2"
    assert kwargs["output_format"] == "mp3_44100_128"


def test_convert_tts_request_full(handler, base_config):
    """Full TTS request with all optional fields."""
    request = TTSRequest(
        text="Hello world",
        voice_id="test-voice-id",
        output_format=AudioOutputFormat(format="wav", sample_rate=44100),
        language_code="en",
        voice_settings={"stability": 0.5, "similarity_boost": 0.8},
        extra_params={
            "seed": 42,
            "previous_text": "Before",
            "next_text": "After",
        },
    )
    kwargs = handler._convert_tts_request(base_config, request)

    assert kwargs["text"] == "Hello world"
    assert kwargs["voice_id"] == "test-voice-id"
    assert kwargs["model_id"] == "eleven_multilingual_v2"
    assert kwargs["output_format"] == "wav_44100"
    assert kwargs["language_code"] == "en"
    assert kwargs["voice_settings"] == {"stability": 0.5, "similarity_boost": 0.8}
    assert kwargs["seed"] == 42
    assert kwargs["previous_text"] == "Before"
    assert kwargs["next_text"] == "After"


def test_convert_tts_request_extra_params(handler, base_config):
    """Extra params are merged into SDK kwargs."""
    request = TTSRequest(
        text="Hello",
        voice_id="v1",
        extra_params={"custom_param": "value"},
    )
    kwargs = handler._convert_tts_request(base_config, request)
    assert kwargs["custom_param"] == "value"


# ==================== STS Request Conversion ====================


def test_resolve_audio_bytes_media_content(handler):
    """Resolves MediaContent dict to BytesIO wrapping content bytes."""
    audio = {"content": b"audio-data", "content_type": "audio/wav"}
    result = handler._resolve_audio_bytes(audio)
    assert result.read() == b"audio-data"


def test_resolve_audio_bytes_base64(handler):
    """Resolves base64 string to BytesIO with decoded bytes."""
    audio_b64 = base64.b64encode(b"audio-data").decode()
    result = handler._resolve_audio_bytes(audio_b64)
    assert result.read() == b"audio-data"


def test_resolve_audio_bytes_url(handler):
    """Resolves URL by downloading content."""
    with patch(
        "tarash.tarash_gateway.providers.elevenlabs.download_media_from_url",
        return_value=(b"downloaded-audio", "audio/mpeg"),
    ) as mock_download:
        result = handler._resolve_audio_bytes("https://example.com/audio.mp3")
        assert result.read() == b"downloaded-audio"
        mock_download.assert_called_once_with(
            "https://example.com/audio.mp3", provider="elevenlabs"
        )


@pytest.mark.asyncio
async def test_resolve_audio_bytes_async_media_content(handler):
    """Async: resolves MediaContent dict to BytesIO."""
    audio = {"content": b"audio-data", "content_type": "audio/wav"}
    result = await handler._resolve_audio_bytes_async(audio)
    assert result.read() == b"audio-data"


@pytest.mark.asyncio
async def test_resolve_audio_bytes_async_url(handler):
    """Async: resolves URL by downloading content."""
    with patch(
        "tarash.tarash_gateway.providers.elevenlabs.download_media_from_url_async",
        return_value=(b"downloaded-audio", "audio/mpeg"),
    ) as mock_download:
        result = await handler._resolve_audio_bytes_async(
            "https://example.com/audio.mp3"
        )
        assert result.read() == b"downloaded-audio"
        mock_download.assert_called_once_with(
            "https://example.com/audio.mp3", provider="elevenlabs"
        )


def test_convert_sts_request_basic(handler, base_config, sts_request):
    """STS request conversion includes voice_id and model_id, not audio."""
    kwargs = handler._convert_sts_request(base_config, sts_request)

    assert kwargs["voice_id"] == "test-voice-id"
    assert kwargs["model_id"] == "eleven_multilingual_v2"
    assert "audio" not in kwargs


def test_convert_sts_request_uses_config_model(handler):
    """STS request always uses config.model for model_id."""
    config = AudioGenerationConfig(
        model="eleven_multilingual_sts_v2",
        provider="elevenlabs",
        api_key="test-api-key",
    )
    request = STSRequest(
        audio={"content": b"data", "content_type": "audio/wav"},
        voice_id="v1",
    )
    kwargs = handler._convert_sts_request(config, request)
    assert kwargs["model_id"] == "eleven_multilingual_sts_v2"


def test_convert_sts_request_voice_settings_json(handler, base_config):
    """STS request JSON-encodes voice_settings for multipart form."""
    request = STSRequest(
        audio={"content": b"data", "content_type": "audio/wav"},
        voice_id="v1",
        voice_settings={"stability": 0.5},
    )
    kwargs = handler._convert_sts_request(base_config, request)
    assert kwargs["voice_settings"] == '{"stability": 0.5}'


def test_convert_sts_request_remove_background_noise(handler, base_config):
    """STS request passes remove_background_noise."""
    request = STSRequest(
        audio={"content": b"data", "content_type": "audio/wav"},
        voice_id="v1",
        extra_params={"remove_background_noise": True},
    )
    kwargs = handler._convert_sts_request(base_config, request)
    assert kwargs["remove_background_noise"] is True


# ==================== TTS Response Conversion ====================


def test_convert_tts_response(handler, base_config, tts_request):
    """TTS response has base64 audio, content_type, and metadata."""
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
    assert response.raw_response["model"] == "eleven_multilingual_v2"


def test_convert_tts_response_wav_format(handler, base_config):
    """TTS response derives correct content_type for wav format."""
    request = TTSRequest(
        text="Hi",
        voice_id="v1",
        output_format=AudioOutputFormat(format="wav", sample_rate=44100),
    )
    response = handler._convert_tts_response(base_config, request, "req-1", b"wav-data")
    assert response.content_type == "audio/wav"


# ==================== STS Response Conversion ====================


def test_convert_sts_response(handler, base_config, sts_request):
    """STS response has base64 audio, content_type, and metadata."""
    audio_bytes = b"sts-output"
    response = handler._convert_sts_response(
        base_config, sts_request, "req-456", audio_bytes
    )

    assert isinstance(response, STSResponse)
    assert response.request_id == "req-456"
    assert response.audio == base64.b64encode(audio_bytes).decode("utf-8")
    assert response.content_type == "audio/mpeg"
    assert response.status == "completed"


# ==================== Error Handling ====================


def _make_api_error(status_code, body="error"):
    """Create a mock ApiError."""
    from elevenlabs.core import ApiError

    err = ApiError(status_code=status_code, body=body)
    return err


def test_handle_error_400_validation(handler, base_config, tts_request):
    """400 status maps to ValidationError."""
    ex = _make_api_error(400, "Bad request")
    result = handler._handle_error(base_config, tts_request, "req-1", ex)
    assert isinstance(result, ValidationError)
    assert result.provider == "elevenlabs"


def test_handle_error_401_auth(handler, base_config, tts_request):
    """401 status maps to HTTPError."""
    ex = _make_api_error(401, "Unauthorized")
    result = handler._handle_error(base_config, tts_request, "req-1", ex)
    assert isinstance(result, HTTPError)
    assert result.status_code == 401
    assert result.provider == "elevenlabs"
    assert result.model == "eleven_multilingual_v2"


def test_handle_error_403_moderation(handler, base_config, tts_request):
    """403 status maps to ContentModerationError."""
    ex = _make_api_error(403, "Forbidden")
    result = handler._handle_error(base_config, tts_request, "req-1", ex)
    assert isinstance(result, ContentModerationError)


def test_handle_error_422_validation(handler, base_config, tts_request):
    """422 status maps to ValidationError."""
    ex = _make_api_error(422, "Unprocessable")
    result = handler._handle_error(base_config, tts_request, "req-1", ex)
    assert isinstance(result, ValidationError)


def test_handle_error_429_rate_limit(handler, base_config, tts_request):
    """429 status maps to HTTPError."""
    ex = _make_api_error(429, "Rate limited")
    result = handler._handle_error(base_config, tts_request, "req-1", ex)
    assert isinstance(result, HTTPError)
    assert result.status_code == 429
    assert result.provider == "elevenlabs"
    assert result.model == "eleven_multilingual_v2"


def test_handle_error_500_server(handler, base_config, tts_request):
    """500 status maps to HTTPError."""
    ex = _make_api_error(500, "Server error")
    result = handler._handle_error(base_config, tts_request, "req-1", ex)
    assert isinstance(result, HTTPError)
    assert result.status_code == 500
    assert result.provider == "elevenlabs"
    assert result.model == "eleven_multilingual_v2"


def test_handle_error_connection_error(handler, base_config, tts_request):
    """ConnectionError maps to HTTPConnectionError."""
    ex = ConnectionError("Connection refused")
    result = handler._handle_error(base_config, tts_request, "req-1", ex)
    assert isinstance(result, HTTPConnectionError)


def test_handle_error_unknown_status_code(handler, base_config, tts_request):
    """Unknown ApiError status code maps to generic HTTPError."""
    ex = _make_api_error(502, "Bad gateway")
    result = handler._handle_error(base_config, tts_request, "req-1", ex)
    assert isinstance(result, HTTPError)
    assert result.status_code == 502


def test_handle_error_timeout(handler, base_config, tts_request):
    """OSError with 'timed out' maps to TimeoutError."""
    ex = OSError("Connection timed out")
    result = handler._handle_error(base_config, tts_request, "req-1", ex)
    assert isinstance(result, TimeoutError)
    assert result.provider == "elevenlabs"
    assert result.model == "eleven_multilingual_v2"


def test_handle_error_unknown(handler, base_config, tts_request):
    """Unknown errors map to TarashException."""
    ex = RuntimeError("Something unexpected")
    result = handler._handle_error(base_config, tts_request, "req-1", ex)
    assert isinstance(result, TarashException)
    assert "Something unexpected" in result.message


# ==================== Integration: generate_tts (mocked SDK) ====================


def test_generate_tts_sync_success(handler, base_config, tts_request):
    """Sync TTS generation with mocked SDK returns valid TTSResponse."""
    mock_client = MagicMock()
    mock_client.text_to_speech.convert.return_value = iter([b"chunk1", b"chunk2"])

    with patch.object(handler, "_get_client", return_value=mock_client):
        response = handler.generate_tts(base_config, tts_request)

    assert isinstance(response, TTSResponse)
    assert response.status == "completed"
    expected_audio = base64.b64encode(b"chunk1chunk2").decode("utf-8")
    assert response.audio == expected_audio
    assert response.content_type == "audio/mpeg"
    assert len(response.request_id) == 32

    # Verify SDK was called with correct args
    call_kwargs = mock_client.text_to_speech.convert.call_args.kwargs
    assert call_kwargs["text"] == "Hello world"
    assert call_kwargs["voice_id"] == "test-voice-id"
    assert call_kwargs["model_id"] == "eleven_multilingual_v2"


async def test_generate_tts_async_success(handler, base_config, tts_request):
    """Async TTS generation with mocked SDK returns valid TTSResponse."""
    captured_kwargs = {}

    async def mock_iter(*args, **kwargs):
        captured_kwargs.update(kwargs)
        yield b"async_chunk1"
        yield b"async_chunk2"

    mock_client = MagicMock()
    mock_client.text_to_speech.convert = mock_iter

    with patch.object(handler, "_get_client", return_value=mock_client):
        response = await handler.generate_tts_async(base_config, tts_request)

    assert isinstance(response, TTSResponse)
    assert response.status == "completed"
    expected_audio = base64.b64encode(b"async_chunk1async_chunk2").decode("utf-8")
    assert response.audio == expected_audio

    # Verify SDK was called with correct args
    assert captured_kwargs["text"] == "Hello world"
    assert captured_kwargs["voice_id"] == "test-voice-id"
    assert captured_kwargs["model_id"] == "eleven_multilingual_v2"


def test_generate_tts_sync_api_error(handler, base_config, tts_request):
    """Sync TTS generation raises ValidationError on 400."""
    from elevenlabs.core import ApiError

    mock_client = MagicMock()
    mock_client.text_to_speech.convert.side_effect = ApiError(
        status_code=400, body="Bad request"
    )

    with patch.object(handler, "_get_client", return_value=mock_client):
        with pytest.raises(ValidationError, match="Invalid request"):
            handler.generate_tts(base_config, tts_request)


async def test_generate_tts_async_api_error(handler, base_config, tts_request):
    """Async TTS generation raises ValidationError on 400."""
    from elevenlabs.core import ApiError

    async def mock_error_iter(*args, **kwargs):
        raise ApiError(status_code=400, body="Bad request")
        yield  # make it an async generator  # noqa: RUF027

    mock_client = MagicMock()
    mock_client.text_to_speech.convert = mock_error_iter

    with patch.object(handler, "_get_client", return_value=mock_client):
        with pytest.raises(ValidationError, match="Invalid request"):
            await handler.generate_tts_async(base_config, tts_request)


# ==================== Integration: generate_sts (mocked SDK) ====================


def test_generate_sts_sync_success(handler, base_config, sts_request):
    """Sync STS generation with mocked SDK returns valid STSResponse."""
    mock_client = MagicMock()
    mock_client.speech_to_speech.convert.return_value = iter([b"sts_chunk"])

    with patch.object(handler, "_get_client", return_value=mock_client):
        response = handler.generate_sts(base_config, sts_request)

    assert isinstance(response, STSResponse)
    assert response.status == "completed"
    expected_audio = base64.b64encode(b"sts_chunk").decode("utf-8")
    assert response.audio == expected_audio

    # Verify SDK was called with correct args
    call_kwargs = mock_client.speech_to_speech.convert.call_args.kwargs
    assert call_kwargs["voice_id"] == "test-voice-id"
    assert call_kwargs["model_id"] == "eleven_multilingual_v2"
    assert "audio" in call_kwargs


async def test_generate_sts_async_success(handler, base_config, sts_request):
    """Async STS generation with mocked SDK returns valid STSResponse."""
    captured_kwargs = {}

    async def mock_iter(*args, **kwargs):
        captured_kwargs.update(kwargs)
        yield b"sts_async"

    mock_client = MagicMock()
    mock_client.speech_to_speech.convert = mock_iter

    with patch.object(handler, "_get_client", return_value=mock_client):
        response = await handler.generate_sts_async(base_config, sts_request)

    assert isinstance(response, STSResponse)
    assert response.status == "completed"

    # Verify SDK was called with correct args
    assert captured_kwargs["voice_id"] == "test-voice-id"
    assert captured_kwargs["model_id"] == "eleven_multilingual_v2"
    assert "audio" in captured_kwargs


# ==================== Client Creation ====================


def test_get_client_sync(handler, base_config):
    """Sync client creation passes correct kwargs."""
    with patch("tarash.tarash_gateway.providers.elevenlabs.ElevenLabs") as mock_cls:
        handler._get_client(base_config, "sync")
        mock_cls.assert_called_once_with(api_key="test-api-key", timeout=240)


def test_get_client_async(handler, base_config):
    """Async client creation passes correct kwargs."""
    with patch(
        "tarash.tarash_gateway.providers.elevenlabs.AsyncElevenLabs"
    ) as mock_cls:
        handler._get_client(base_config, "async")
        mock_cls.assert_called_once_with(api_key="test-api-key", timeout=240)


def test_get_client_no_api_key(handler):
    """Client creation raises ValidationError without api_key."""
    config = AudioGenerationConfig(
        model="model",
        provider="elevenlabs",
    )
    with pytest.raises(ValidationError, match="api_key is required"):
        handler._get_client(config, "sync")
