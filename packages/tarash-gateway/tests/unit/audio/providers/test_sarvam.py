"""Unit tests for Sarvam AI TTS provider handler."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from tarash.tarash_gateway.models import (
    AudioGenerationConfig,
    AudioOutputFormat,
    TTSRequest,
    TTSResponse,
)
from tarash.tarash_gateway.exceptions import (
    ContentModerationError,
    HTTPConnectionError,
    HTTPError,
    TarashException,
    TimeoutError,
    ValidationError,
)
from tarash.tarash_gateway.providers.sarvam import SarvamProviderHandler


# ==================== Fixtures ====================


@pytest.fixture
def handler():
    return SarvamProviderHandler()


@pytest.fixture
def base_config():
    return AudioGenerationConfig(
        model="bulbul:v3",
        provider="sarvam",
        api_key="test-api-key",
        timeout=240,
    )


@pytest.fixture
def tts_request():
    return TTSRequest(
        text="Namaste duniya",
        voice_id="shubh",
        language_code="hi-IN",
    )


def _mock_client_with_result(audios=None, request_id="sarvam-req-1"):
    """Create a mock Sarvam client that returns a successful result."""
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.audios = audios or ["dGVzdC1hdWRpbw=="]
    mock_result.request_id = request_id
    mock_client.text_to_speech.convert.return_value = mock_result
    return mock_client


def _mock_async_client_with_result(audios=None, request_id="sarvam-req-1"):
    """Create a mock async Sarvam client that returns a successful result."""
    mock_client = MagicMock()
    mock_result = MagicMock()
    mock_result.audios = audios or ["dGVzdC1hdWRpbw=="]
    mock_result.request_id = request_id
    mock_client.text_to_speech.convert = AsyncMock(return_value=mock_result)
    return mock_client


def _mock_client_with_error(error):
    """Create a mock Sarvam client that raises an error."""
    mock_client = MagicMock()
    mock_client.text_to_speech.convert.side_effect = error
    return mock_client


def _make_sarvam_api_error(status_code, message="error"):
    """Create a Sarvam API error."""
    from sarvamai.core.api_error import ApiError

    return ApiError(status_code=status_code, body={"error": message})


# ==================== Client Creation ====================


def test_get_client_sync(handler, base_config):
    with patch("tarash.tarash_gateway.providers.sarvam.SarvamAI") as mock_cls:
        handler._get_client(base_config, "sync")
        mock_cls.assert_called_once_with(api_subscription_key="test-api-key")


def test_get_client_async(handler, base_config):
    with patch("tarash.tarash_gateway.providers.sarvam.AsyncSarvamAI") as mock_cls:
        handler._get_client(base_config, "async")
        mock_cls.assert_called_once_with(api_subscription_key="test-api-key")


# ==================== Request Conversion ====================


def test_convert_tts_request_basic(handler, base_config, tts_request):
    kwargs = handler._convert_tts_request(base_config, tts_request)
    assert kwargs["text"] == "Namaste duniya"
    assert kwargs["target_language_code"] == "hi-IN"
    assert kwargs["speaker"] == "shubh"
    assert kwargs["model"] == "bulbul:v3"
    assert kwargs["speech_sample_rate"] == 44100  # TTSRequest default
    assert kwargs["output_audio_codec"] == "mp3"  # default format


def test_convert_tts_request_custom_output_format(handler, base_config):
    request = TTSRequest(
        text="Hi",
        voice_id="v",
        language_code="en-IN",
        output_format=AudioOutputFormat(format="wav", sample_rate=22050),
    )
    kwargs = handler._convert_tts_request(base_config, request)
    assert kwargs["output_audio_codec"] == "wav"
    assert kwargs["speech_sample_rate"] == 22050


def test_convert_tts_request_voice_settings(handler, base_config):
    request = TTSRequest(
        text="Hi",
        voice_id="v",
        language_code="en-IN",
        voice_settings={"pace": 1.2, "temperature": 0.8},
    )
    kwargs = handler._convert_tts_request(base_config, request)
    assert kwargs["pace"] == 1.2
    assert kwargs["temperature"] == 0.8


def test_convert_tts_request_extra_params(handler, base_config):
    request = TTSRequest(
        text="Hi",
        voice_id="v",
        language_code="en-IN",
        extra_params={"custom_key": "custom_value"},
    )
    kwargs = handler._convert_tts_request(base_config, request)
    assert kwargs["custom_key"] == "custom_value"


# ==================== Response Conversion ====================


def test_convert_tts_response_basic(handler, base_config, tts_request):
    mock_result = MagicMock()
    mock_result.audios = ["dGVzdA=="]
    mock_result.request_id = "sarvam-123"

    response = handler._convert_tts_response(
        base_config, tts_request, "gen-456", mock_result
    )

    assert response.audio == "dGVzdA=="
    assert response.request_id == "sarvam-123"  # uses Sarvam's ID
    assert response.content_type == "audio/mpeg"
    assert response.status == "completed"


def test_convert_tts_response_fallback_request_id(handler, base_config, tts_request):
    mock_result = MagicMock()
    mock_result.audios = ["dGVzdA=="]
    mock_result.request_id = None

    response = handler._convert_tts_response(
        base_config, tts_request, "gen-456", mock_result
    )

    assert response.request_id == "gen-456"  # falls back to generated ID


# ==================== TTS Success ====================


def test_generate_tts_sync_success(handler, base_config, tts_request):
    mock_client = _mock_client_with_result()

    with patch.object(handler, "_get_client", return_value=mock_client):
        response = handler.generate_tts(base_config, tts_request)

    assert isinstance(response, TTSResponse)
    assert response.status == "completed"
    assert response.audio == "dGVzdC1hdWRpbw=="
    assert response.content_type == "audio/mpeg"
    assert response.request_id == "sarvam-req-1"

    # Verify SDK was called with correct args
    call_kwargs = mock_client.text_to_speech.convert.call_args.kwargs
    assert call_kwargs["text"] == "Namaste duniya"
    assert call_kwargs["target_language_code"] == "hi-IN"
    assert call_kwargs["speaker"] == "shubh"
    assert call_kwargs["model"] == "bulbul:v3"


@pytest.mark.asyncio
async def test_generate_tts_async_success(handler, base_config, tts_request):
    mock_client = _mock_async_client_with_result(
        audios=["YXN5bmMtYXVkaW8="], request_id="sarvam-req-async"
    )

    with patch.object(handler, "_get_client", return_value=mock_client):
        response = await handler.generate_tts_async(base_config, tts_request)

    assert isinstance(response, TTSResponse)
    assert response.status == "completed"
    assert response.audio == "YXN5bmMtYXVkaW8="

    # Verify SDK was called with correct args
    call_kwargs = mock_client.text_to_speech.convert.call_args.kwargs
    assert call_kwargs["text"] == "Namaste duniya"
    assert call_kwargs["target_language_code"] == "hi-IN"
    assert call_kwargs["speaker"] == "shubh"
    assert call_kwargs["model"] == "bulbul:v3"


# ==================== TTS Errors ====================


def test_missing_language_code_raises_validation_error(handler, base_config):
    request = TTSRequest(text="Hello", voice_id="shubh")

    with pytest.raises(ValidationError, match="language_code is required"):
        handler.generate_tts(base_config, request)


def test_missing_api_key_raises_validation_error(handler):
    config = AudioGenerationConfig(model="bulbul:v3", provider="sarvam")
    request = TTSRequest(text="Hello", voice_id="shubh", language_code="hi-IN")

    with pytest.raises(ValidationError, match="api_key is required"):
        handler.generate_tts(config, request)


def test_bad_request_raises_validation_error(handler, base_config, tts_request):
    mock_client = _mock_client_with_error(
        _make_sarvam_api_error(400, "Invalid speaker")
    )

    with patch.object(handler, "_get_client", return_value=mock_client):
        with pytest.raises(ValidationError, match="Invalid request"):
            handler.generate_tts(base_config, tts_request)


def test_unprocessable_raises_validation_error(handler, base_config, tts_request):
    mock_client = _mock_client_with_error(
        _make_sarvam_api_error(422, "Unprocessable entity")
    )

    with patch.object(handler, "_get_client", return_value=mock_client):
        with pytest.raises(ValidationError, match="Invalid request"):
            handler.generate_tts(base_config, tts_request)


def test_auth_failure_raises_http_error_401(handler, base_config, tts_request):
    mock_client = _mock_client_with_error(
        _make_sarvam_api_error(401, "Invalid API key")
    )

    with patch.object(handler, "_get_client", return_value=mock_client):
        with pytest.raises(HTTPError, match="Authentication failed") as exc_info:
            handler.generate_tts(base_config, tts_request)
        assert exc_info.value.status_code == 401
        assert exc_info.value.provider == "sarvam"
        assert exc_info.value.model == "bulbul:v3"


def test_forbidden_raises_content_moderation_error(handler, base_config, tts_request):
    mock_client = _mock_client_with_error(
        _make_sarvam_api_error(403, "Content policy violation")
    )

    with patch.object(handler, "_get_client", return_value=mock_client):
        with pytest.raises(ContentModerationError):
            handler.generate_tts(base_config, tts_request)


def test_rate_limit_raises_http_error_429(handler, base_config, tts_request):
    mock_client = _mock_client_with_error(_make_sarvam_api_error(429, "Rate limited"))

    with patch.object(handler, "_get_client", return_value=mock_client):
        with pytest.raises(HTTPError, match="Rate limit exceeded") as exc_info:
            handler.generate_tts(base_config, tts_request)
        assert exc_info.value.status_code == 429
        assert exc_info.value.provider == "sarvam"
        assert exc_info.value.model == "bulbul:v3"


def test_server_error_raises_http_error_500(handler, base_config, tts_request):
    mock_client = _mock_client_with_error(
        _make_sarvam_api_error(500, "Internal server error")
    )

    with patch.object(handler, "_get_client", return_value=mock_client):
        with pytest.raises(HTTPError, match="Server error") as exc_info:
            handler.generate_tts(base_config, tts_request)
        assert exc_info.value.status_code == 500
        assert exc_info.value.provider == "sarvam"
        assert exc_info.value.model == "bulbul:v3"


def test_unknown_status_code_raises_http_error(handler, base_config, tts_request):
    mock_client = _mock_client_with_error(_make_sarvam_api_error(502, "Bad gateway"))

    with patch.object(handler, "_get_client", return_value=mock_client):
        with pytest.raises(HTTPError, match="API error") as exc_info:
            handler.generate_tts(base_config, tts_request)
        assert exc_info.value.status_code == 502


def test_timeout_raises_timeout_error(handler, base_config, tts_request):
    mock_client = _mock_client_with_error(httpx.ReadTimeout("Connection timed out"))

    with patch.object(handler, "_get_client", return_value=mock_client):
        with pytest.raises(TimeoutError):
            handler.generate_tts(base_config, tts_request)


def test_connect_timeout_raises_timeout_error_not_connection_error(
    handler, base_config, tts_request
):
    """ConnectTimeout is both TimeoutException and ConnectError - must map to TimeoutError."""
    mock_client = _mock_client_with_error(httpx.ConnectTimeout("Connection timed out"))

    with patch.object(handler, "_get_client", return_value=mock_client):
        with pytest.raises(TimeoutError):
            handler.generate_tts(base_config, tts_request)


def test_connection_failure_raises_connection_error(handler, base_config, tts_request):
    mock_client = _mock_client_with_error(httpx.ConnectError("Connection refused"))

    with patch.object(handler, "_get_client", return_value=mock_client):
        with pytest.raises(HTTPConnectionError):
            handler.generate_tts(base_config, tts_request)


def test_unknown_error_raises_tarash_exception(handler, base_config, tts_request):
    mock_client = _mock_client_with_error(RuntimeError("Something unexpected"))

    with patch.object(handler, "_get_client", return_value=mock_client):
        with pytest.raises(TarashException, match="Something unexpected"):
            handler.generate_tts(base_config, tts_request)


@pytest.mark.asyncio
async def test_async_api_error_raises_validation_error(
    handler, base_config, tts_request
):
    mock_client = MagicMock()
    mock_client.text_to_speech.convert = AsyncMock(
        side_effect=_make_sarvam_api_error(400, "Bad request")
    )

    with patch.object(handler, "_get_client", return_value=mock_client):
        with pytest.raises(ValidationError, match="Invalid request"):
            await handler.generate_tts_async(base_config, tts_request)
