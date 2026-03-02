"""Unit tests for Hume AI TTS provider handler."""

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
from tarash.tarash_gateway.providers.hume import (
    HumeProviderHandler,
    _build_voice_spec,
    _model_to_version,
)


# ==================== Fixtures ====================


@pytest.fixture
def handler():
    return HumeProviderHandler()


@pytest.fixture
def base_config():
    return AudioGenerationConfig(
        model="octave-v1",
        provider="hume",
        api_key="test-api-key",
        timeout=240,
    )


@pytest.fixture
def tts_request():
    return TTSRequest(
        text="Hello world",
        voice_id="Kora",
    )


def _mock_generation(
    audio="dGVzdC1hdWRpbw==",
    generation_id="gen-123",
    duration=1.5,
    file_size=1024,
):
    """Create a mock Hume generation object."""
    gen = MagicMock()
    gen.audio = audio
    gen.generation_id = generation_id
    gen.duration = duration
    gen.file_size = file_size
    gen.encoding = MagicMock()
    gen.encoding.format = "mp3"
    gen.encoding.sample_rate = 48000
    return gen


def _mock_result(generations=None, request_id="hume-req-1"):
    """Create a mock Hume synthesize_json result."""
    result = MagicMock()
    result.generations = generations or [_mock_generation()]
    result.request_id = request_id
    return result


def _mock_client_with_result(result=None):
    """Create a mock sync Hume client."""
    mock_client = MagicMock()
    mock_client.tts.synthesize_json.return_value = result or _mock_result()
    return mock_client


def _mock_async_client_with_result(result=None):
    """Create a mock async Hume client."""
    mock_client = MagicMock()
    mock_client.tts.synthesize_json = AsyncMock(return_value=result or _mock_result())
    return mock_client


def _mock_client_with_error(error):
    """Create a mock Hume client that raises an error."""
    mock_client = MagicMock()
    mock_client.tts.synthesize_json.side_effect = error
    return mock_client


def _make_hume_api_error(status_code, message="error"):
    """Create a Hume API error."""
    from hume.core import ApiError

    return ApiError(status_code=status_code, body={"error": message})


# ==================== Helper Functions ====================


def test_model_to_version_v1():
    assert _model_to_version("octave-v1") == "1"


def test_model_to_version_v2():
    assert _model_to_version("octave-v2") == "2"


def test_model_to_version_auto():
    assert _model_to_version("octave") is None


def test_model_to_version_octave_1():
    assert _model_to_version("octave-1") == "1"


def test_build_voice_spec_name_default():
    request = TTSRequest(text="Hi", voice_id="Kora")
    result = _build_voice_spec(request)
    assert result == {"name": "Kora", "provider": "HUME_AI"}


def test_build_voice_spec_custom_provider():
    request = TTSRequest(
        text="Hi",
        voice_id="my-voice",
        voice_settings={"voice_provider": "CUSTOM_VOICE"},
    )
    result = _build_voice_spec(request)
    assert result == {"name": "my-voice", "provider": "CUSTOM_VOICE"}


def test_build_voice_spec_id_mode():
    request = TTSRequest(
        text="Hi",
        voice_id="abc-123",
        voice_settings={"voice_id_mode": "id"},
    )
    result = _build_voice_spec(request)
    assert result == {"id": "abc-123", "provider": "HUME_AI"}


def test_build_voice_spec_no_voice():
    request = TTSRequest(text="Hi", voice_id="")
    result = _build_voice_spec(request)
    assert result is None


# ==================== Client Creation ====================


def test_get_client_sync(handler, base_config):
    with patch("tarash.tarash_gateway.providers.hume.HumeClient") as mock_cls:
        handler._get_client(base_config, "sync")
        mock_cls.assert_called_once_with(api_key="test-api-key", timeout=240)


def test_get_client_async(handler, base_config):
    with patch("tarash.tarash_gateway.providers.hume.AsyncHumeClient") as mock_cls:
        handler._get_client(base_config, "async")
        mock_cls.assert_called_once_with(api_key="test-api-key", timeout=240)


def test_get_client_missing_api_key_raises(handler):
    config = AudioGenerationConfig(model="octave-v1", provider="hume")
    with pytest.raises(ValidationError, match="api_key is required"):
        handler._get_client(config, "sync")


# ==================== Request Conversion ====================


def test_convert_tts_request_basic(handler, base_config, tts_request):
    kwargs = handler._convert_tts_request(base_config, tts_request)
    assert len(kwargs["utterances"]) == 1
    utterance = kwargs["utterances"][0]
    assert utterance.text == "Hello world"
    assert kwargs["version"] == "1"
    assert kwargs["format"] == {"type": "mp3"}


def test_convert_tts_request_with_description(handler, base_config):
    request = TTSRequest(
        text="Hello",
        voice_id="Kora",
        voice_settings={"description": "calm, soothing"},
    )
    kwargs = handler._convert_tts_request(base_config, request)
    utterance = kwargs["utterances"][0]
    assert utterance.description == "calm, soothing"


def test_convert_tts_request_with_speed(handler, base_config):
    request = TTSRequest(
        text="Hello",
        voice_id="Kora",
        voice_settings={"speed": 1.5},
    )
    kwargs = handler._convert_tts_request(base_config, request)
    utterance = kwargs["utterances"][0]
    assert utterance.speed == 1.5


def test_convert_tts_request_with_trailing_silence(handler, base_config):
    request = TTSRequest(
        text="Hello",
        voice_id="Kora",
        voice_settings={"trailing_silence": 2.0},
    )
    kwargs = handler._convert_tts_request(base_config, request)
    utterance = kwargs["utterances"][0]
    assert utterance.trailing_silence == 2.0


def test_convert_tts_request_wav_format(handler, base_config):
    request = TTSRequest(
        text="Hello",
        voice_id="Kora",
        output_format=AudioOutputFormat(format="wav"),
    )
    kwargs = handler._convert_tts_request(base_config, request)
    assert kwargs["format"] == {"type": "wav"}


def test_convert_tts_request_v2_model(handler):
    config = AudioGenerationConfig(
        model="octave-v2",
        provider="hume",
        api_key="test-key",
    )
    request = TTSRequest(text="Hello", voice_id="Kora")
    kwargs = handler._convert_tts_request(config, request)
    assert kwargs["version"] == "2"


def test_convert_tts_request_extra_params(handler, base_config):
    request = TTSRequest(
        text="Hello",
        voice_id="Kora",
        extra_params={"num_generations": 3, "instant_mode": False},
    )
    kwargs = handler._convert_tts_request(base_config, request)
    assert kwargs["num_generations"] == 3
    assert kwargs["instant_mode"] is False


def test_convert_tts_request_context_via_extra_params(handler, base_config):
    request = TTSRequest(
        text="Hello",
        voice_id="Kora",
        extra_params={"context": {"generation_id": "prev-gen-123"}},
    )
    kwargs = handler._convert_tts_request(base_config, request)
    assert kwargs["context"] == {"generation_id": "prev-gen-123"}


# ==================== Response Conversion ====================


def test_convert_tts_response_basic(handler, base_config, tts_request):
    result = _mock_result()
    response = handler._convert_tts_response(
        base_config, tts_request, "req-456", result
    )

    assert isinstance(response, TTSResponse)
    assert response.audio == "dGVzdC1hdWRpbw=="
    assert response.request_id == "hume-req-1"
    assert response.content_type == "audio/mpeg"
    assert response.status == "completed"
    assert response.duration == 1.5
    assert response.raw_response["generation_id"] == "gen-123"
    assert response.raw_response["model"] == "octave-v1"


def test_convert_tts_response_fallback_request_id(handler, base_config, tts_request):
    result = _mock_result(request_id=None)
    response = handler._convert_tts_response(
        base_config, tts_request, "req-456", result
    )
    assert response.request_id == "req-456"


def test_convert_tts_response_wav_content_type(handler, base_config):
    request = TTSRequest(
        text="Hi",
        voice_id="Kora",
        output_format=AudioOutputFormat(format="wav"),
    )
    result = _mock_result()
    response = handler._convert_tts_response(base_config, request, "req-789", result)
    assert response.content_type == "audio/wav"


# ==================== TTS Success ====================


def test_generate_tts_sync_success(handler, base_config, tts_request):
    mock_client = _mock_client_with_result()

    with patch.object(handler, "_get_client", return_value=mock_client):
        response = handler.generate_tts(base_config, tts_request)

    assert isinstance(response, TTSResponse)
    assert response.status == "completed"
    assert response.audio == "dGVzdC1hdWRpbw=="
    assert response.content_type == "audio/mpeg"
    assert response.duration == 1.5

    # Verify SDK was called
    mock_client.tts.synthesize_json.assert_called_once()
    call_kwargs = mock_client.tts.synthesize_json.call_args.kwargs
    assert len(call_kwargs["utterances"]) == 1
    assert call_kwargs["utterances"][0].text == "Hello world"


@pytest.mark.asyncio
async def test_generate_tts_async_success(handler, base_config, tts_request):
    mock_client = _mock_async_client_with_result()

    with patch.object(handler, "_get_client", return_value=mock_client):
        response = await handler.generate_tts_async(base_config, tts_request)

    assert isinstance(response, TTSResponse)
    assert response.status == "completed"
    assert response.audio == "dGVzdC1hdWRpbw=="

    # Verify SDK was called
    mock_client.tts.synthesize_json.assert_called_once()


# ==================== TTS Errors ====================


def test_missing_api_key_raises_validation_error(handler):
    config = AudioGenerationConfig(model="octave-v1", provider="hume")
    request = TTSRequest(text="Hello", voice_id="Kora")

    with pytest.raises(ValidationError, match="api_key is required"):
        handler.generate_tts(config, request)


def test_bad_request_raises_validation_error(handler, base_config, tts_request):
    mock_client = _mock_client_with_error(
        _make_hume_api_error(400, "Invalid utterance")
    )

    with patch.object(handler, "_get_client", return_value=mock_client):
        with pytest.raises(ValidationError, match="Invalid request"):
            handler.generate_tts(base_config, tts_request)


def test_unprocessable_raises_validation_error(handler, base_config, tts_request):
    mock_client = _mock_client_with_error(
        _make_hume_api_error(422, "Unprocessable entity")
    )

    with patch.object(handler, "_get_client", return_value=mock_client):
        with pytest.raises(ValidationError, match="Invalid request"):
            handler.generate_tts(base_config, tts_request)


def test_auth_failure_raises_http_error_401(handler, base_config, tts_request):
    mock_client = _mock_client_with_error(_make_hume_api_error(401, "Invalid API key"))

    with patch.object(handler, "_get_client", return_value=mock_client):
        with pytest.raises(HTTPError, match="Authentication failed") as exc_info:
            handler.generate_tts(base_config, tts_request)
        assert exc_info.value.status_code == 401
        assert exc_info.value.provider == "hume"
        assert exc_info.value.model == "octave-v1"


def test_forbidden_raises_content_moderation_error(handler, base_config, tts_request):
    mock_client = _mock_client_with_error(
        _make_hume_api_error(403, "Content policy violation")
    )

    with patch.object(handler, "_get_client", return_value=mock_client):
        with pytest.raises(ContentModerationError):
            handler.generate_tts(base_config, tts_request)


def test_rate_limit_raises_http_error_429(handler, base_config, tts_request):
    mock_client = _mock_client_with_error(_make_hume_api_error(429, "Rate limited"))

    with patch.object(handler, "_get_client", return_value=mock_client):
        with pytest.raises(HTTPError, match="Rate limit exceeded") as exc_info:
            handler.generate_tts(base_config, tts_request)
        assert exc_info.value.status_code == 429


def test_server_error_raises_http_error_500(handler, base_config, tts_request):
    mock_client = _mock_client_with_error(
        _make_hume_api_error(500, "Internal server error")
    )

    with patch.object(handler, "_get_client", return_value=mock_client):
        with pytest.raises(HTTPError, match="Server error") as exc_info:
            handler.generate_tts(base_config, tts_request)
        assert exc_info.value.status_code == 500


def test_timeout_raises_timeout_error(handler, base_config, tts_request):
    mock_client = _mock_client_with_error(httpx.ReadTimeout("Connection timed out"))

    with patch.object(handler, "_get_client", return_value=mock_client):
        with pytest.raises(TimeoutError):
            handler.generate_tts(base_config, tts_request)


def test_connect_timeout_raises_timeout_error_not_connection_error(
    handler, base_config, tts_request
):
    """ConnectTimeout is both TimeoutException and ConnectError — must map to TimeoutError."""
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
    mock_client.tts.synthesize_json = AsyncMock(
        side_effect=_make_hume_api_error(400, "Bad request")
    )

    with patch.object(handler, "_get_client", return_value=mock_client):
        with pytest.raises(ValidationError, match="Invalid request"):
            await handler.generate_tts_async(base_config, tts_request)
