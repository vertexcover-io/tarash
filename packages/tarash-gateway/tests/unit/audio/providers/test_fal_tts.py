"""Unit tests for Fal TTS provider handler (minimax speech models)."""

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tarash.tarash_gateway.models import (
    AudioGenerationConfig,
    AudioOutputFormat,
    TTSRequest,
    TTSResponse,
    TTSUpdate,
)
from tarash.tarash_gateway.exceptions import (
    ContentModerationError,
    GenerationFailedError,
    HTTPConnectionError,
    HTTPError,
    TimeoutError,
    ValidationError,
)
from tarash.tarash_gateway.providers.fal import (
    FalProviderHandler,
    _extract_tts_audio_url,
    _output_format_to_fal_audio_setting,
    parse_fal_tts_status,
)


# ==================== Fixtures ====================


@pytest.fixture
def handler():
    return FalProviderHandler()


@pytest.fixture
def base_config():
    return AudioGenerationConfig(
        model="fal-ai/minimax/speech-2.8-hd",
        provider="fal",
        api_key="test-api-key",
        timeout=240,
    )


@pytest.fixture
def tts_request():
    return TTSRequest(text="Hello world", voice_id="Wise_Woman")


# ==================== Output Format Conversion ====================


@pytest.mark.parametrize(
    ("output_format", "expected"),
    [
        (
            AudioOutputFormat(format="mp3", sample_rate=44100, bitrate=128),
            {"format": "mp3", "sample_rate": 44100, "bitrate": 128000},
        ),
        (
            AudioOutputFormat(format="mp3", sample_rate=22050, bitrate=32),
            {"format": "mp3", "sample_rate": 22050, "bitrate": 32000},
        ),
        (AudioOutputFormat(format="mp3"), {"format": "mp3"}),
        (AudioOutputFormat(format="flac"), {"format": "flac"}),
        (
            AudioOutputFormat(format="pcm", sample_rate=16000),
            {"format": "pcm", "sample_rate": 16000},
        ),
        (
            AudioOutputFormat(format="pcm", sample_rate=44100),
            {"format": "pcm", "sample_rate": 44100},
        ),
    ],
)
def test_output_format_to_fal_audio_setting(output_format, expected):
    assert _output_format_to_fal_audio_setting(None, output_format) == expected


# ==================== TTS Request Conversion ====================


def test_convert_tts_request_minimal(handler, base_config, tts_request):
    """Minimal TTS request with just text and voice_id."""
    kwargs = handler._convert_tts_request(base_config, tts_request)

    assert kwargs["prompt"] == "Hello world"
    assert kwargs["voice_setting"] == {"voice_id": "Wise_Woman"}
    assert kwargs["output_format"] == "url"
    assert kwargs["audio_setting"] == {
        "format": "mp3",
        "sample_rate": 44100,
        "bitrate": 128000,
    }
    assert "language_boost" not in kwargs


def test_convert_tts_request_full(handler, base_config):
    """Full TTS request with voice_settings, language, and extra_params."""
    request = TTSRequest(
        text="Bonjour le monde",
        voice_id="Friendly_Person",
        output_format=AudioOutputFormat(format="flac"),
        language_code="French",
        voice_settings={
            "speed": 1.2,
            "vol": 0.8,
            "emotion": "happy",
            "pitch": 3,
            "english_normalization": True,
        },
        extra_params={
            "voice_modify": {"pitch": 10, "intensity": 20, "timbre": -5},
            "pronunciation_dict": {"tone_list": ["hello/(heh-loh)"]},
        },
    )
    kwargs = handler._convert_tts_request(base_config, request)

    assert kwargs["prompt"] == "Bonjour le monde"
    assert kwargs["voice_setting"] == {
        "voice_id": "Friendly_Person",
        "speed": 1.2,
        "vol": 0.8,
        "emotion": "happy",
        "pitch": 3,
        "english_normalization": True,
    }
    assert kwargs["language_boost"] == "French"
    assert kwargs["audio_setting"] == {"format": "flac"}
    assert kwargs["voice_modify"] == {"pitch": 10, "intensity": 20, "timbre": -5}
    assert kwargs["pronunciation_dict"] == {"tone_list": ["hello/(heh-loh)"]}


def test_convert_tts_request_extra_params_override(handler, base_config):
    """Extra params override computed fields, but output_format is always forced to 'url'."""
    request = TTSRequest(
        text="Test",
        voice_id="v1",
        extra_params={"language_boost": "fr", "output_format": "hex"},
    )
    kwargs = handler._convert_tts_request(base_config, request)
    # extra_params can override computed fields
    assert kwargs["language_boost"] == "fr"
    # output_format is forced to "url" after extra_params merge (download logic depends on it)
    assert kwargs["output_format"] == "url"


def test_convert_tts_request_extra_params_override_audio_setting(handler, base_config):
    """audio_setting from extra_params replaces the computed one entirely."""
    request = TTSRequest(
        text="Test",
        voice_id="v1",
        output_format=AudioOutputFormat(format="mp3", sample_rate=44100, bitrate=128),
        extra_params={"audio_setting": {"format": "wav", "channel": 2}},
    )
    kwargs = handler._convert_tts_request(base_config, request)
    # extra_params override replaces computed audio_setting
    assert kwargs["audio_setting"] == {"format": "wav", "channel": 2}


# ==================== TTS Response Conversion ====================


def test_convert_tts_response_with_duration(handler, base_config, tts_request):
    """Response with duration_ms correctly converts to seconds."""
    audio_bytes = b"fake-audio-output"
    fal_result = {
        "audio": {"url": "https://example.com/audio.mp3", "content_type": "audio/mpeg"},
        "duration_ms": 2500,
    }
    response = handler._convert_tts_response(
        base_config, tts_request, "req-123", fal_result, audio_bytes
    )

    assert isinstance(response, TTSResponse)
    assert response.request_id == "req-123"
    assert response.audio == base64.b64encode(audio_bytes).decode("utf-8")
    assert response.content_type == "audio/mpeg"
    assert response.duration == 2.5
    assert response.status == "completed"
    assert response.raw_response == fal_result


def test_convert_tts_response_without_duration(handler, base_config, tts_request):
    """Response without duration_ms has None duration."""
    fal_result = {
        "audio": {"url": "https://example.com/audio.mp3"},
    }
    response = handler._convert_tts_response(
        base_config, tts_request, "req-456", fal_result, b"data"
    )

    assert response.duration is None


def test_convert_tts_response_flac_format(handler, base_config):
    """Flac output format maps to audio/flac content type."""
    request = TTSRequest(
        text="test",
        voice_id="v1",
        output_format=AudioOutputFormat(format="flac"),
    )
    fal_result = {
        "audio": {"url": "https://example.com/audio.flac"},
        "duration_ms": 1000,
    }
    response = handler._convert_tts_response(
        base_config, request, "req-789", fal_result, b"flac-data"
    )

    assert response.content_type == "audio/flac"
    assert response.duration == 1.0


# ==================== Audio URL Extraction ====================


def test_extract_tts_audio_url_success():
    """Extract audio URL from standard Fal response."""
    fal_result = {
        "audio": {
            "url": "https://fal.ai/files/audio.mp3",
            "content_type": "audio/mpeg",
        },
        "duration_ms": 3000,
    }
    assert _extract_tts_audio_url(fal_result) == "https://fal.ai/files/audio.mp3"


def test_extract_tts_audio_url_missing():
    """Raises GenerationFailedError when no audio URL in response."""
    with pytest.raises(GenerationFailedError, match="No audio URL found"):
        _extract_tts_audio_url({"some_field": "value"})


def test_extract_tts_audio_url_empty_audio_dict():
    """Raises GenerationFailedError when audio dict has no url."""
    with pytest.raises(GenerationFailedError, match="No audio URL found"):
        _extract_tts_audio_url({"audio": {"content_type": "audio/mpeg"}})


# ==================== TTS Status Parsing ====================


def test_parse_fal_tts_status_completed():
    """Completed status maps correctly."""
    status = MagicMock()
    status.__class__ = type("Completed", (), {})
    # Use a real Completed-like object
    from fal_client import Completed

    completed = Completed(metrics={}, logs=[])
    update = parse_fal_tts_status("req-1", completed)

    assert isinstance(update, TTSUpdate)
    assert update.request_id == "req-1"
    assert update.status == "completed"
    assert update.progress_percent == 100


def test_parse_fal_tts_status_queued():
    """Queued status maps correctly."""
    from fal_client import Queued

    queued = Queued(position=3)
    update = parse_fal_tts_status("req-2", queued)

    assert update.status == "queued"
    assert update.progress_percent is None
    assert update.update == {"position": 3}


def test_parse_fal_tts_status_in_progress():
    """InProgress status maps correctly."""
    from fal_client import InProgress

    in_progress = InProgress(logs=[])
    update = parse_fal_tts_status("req-3", in_progress)

    assert update.status == "processing"
    assert update.progress_percent is None


# ==================== Error Handling ====================


def test_handle_tts_error_tarash_exception(handler, base_config, tts_request):
    """TarashException passes through unchanged."""
    ex = ValidationError("test", provider="fal", model="test")
    result = handler._handle_tts_error(base_config, "req-1", ex)
    assert result is ex


def test_handle_tts_error_timeout(handler, base_config, tts_request):
    """httpx TimeoutException maps to TimeoutError."""
    import httpx

    ex = httpx.ReadTimeout("timed out")
    result = handler._handle_tts_error(base_config, "req-1", ex)
    assert isinstance(result, TimeoutError)
    assert result.provider == "fal"


def test_handle_tts_error_connection(handler, base_config, tts_request):
    """httpx ConnectError maps to HTTPConnectionError."""
    import httpx

    ex = httpx.ConnectError("refused")
    result = handler._handle_tts_error(base_config, "req-1", ex)
    assert isinstance(result, HTTPConnectionError)


def test_handle_tts_error_fal_400(handler, base_config, tts_request):
    """Fal 400 error maps to ValidationError."""
    from fal_client.client import FalClientHTTPError

    mock_response = MagicMock()
    mock_response.content = b"bad request"
    ex = FalClientHTTPError.__new__(FalClientHTTPError)
    ex.status_code = 400
    ex.message = "Bad request: invalid voice_id"
    ex.response_headers = {}
    ex.response = mock_response

    result = handler._handle_tts_error(base_config, "req-1", ex)
    assert isinstance(result, ValidationError)
    assert result.request_id == "req-1"


def test_handle_tts_error_fal_422_content_policy(handler, base_config, tts_request):
    """Fal 422 content_policy_violation maps to ContentModerationError."""
    from fal_client.client import FalClientHTTPError

    mock_response = MagicMock()
    mock_response.content = b"policy violation"
    ex = FalClientHTTPError.__new__(FalClientHTTPError)
    ex.status_code = 422
    ex.message = "content_policy_violation: text violates policy"
    ex.response_headers = {}
    ex.response = mock_response

    result = handler._handle_tts_error(base_config, "req-1", ex)
    assert isinstance(result, ContentModerationError)


def test_handle_tts_error_fal_500(handler, base_config, tts_request):
    """Fal 500 error maps to HTTPError."""
    from fal_client.client import FalClientHTTPError

    mock_response = MagicMock()
    mock_response.content = b"server error"
    ex = FalClientHTTPError.__new__(FalClientHTTPError)
    ex.status_code = 500
    ex.message = "Internal server error"
    ex.response_headers = {}
    ex.response = mock_response

    result = handler._handle_tts_error(base_config, "req-1", ex)
    assert isinstance(result, HTTPError)
    assert result.status_code == 500


def test_handle_tts_error_unknown(handler, base_config, tts_request):
    """Unknown exception maps to GenerationFailedError."""
    ex = RuntimeError("something unexpected")
    result = handler._handle_tts_error(base_config, "req-1", ex)
    assert isinstance(result, GenerationFailedError)
    assert "something unexpected" in result.message


# ==================== Integration Tests (Mocked SDK) ====================


@pytest.mark.asyncio
async def test_generate_tts_async_success(handler, base_config, tts_request):
    """Async TTS generation with mocked Fal client."""
    from fal_client import Completed

    mock_handle = AsyncMock()
    mock_handle.request_id = "fal-req-123"

    # Simulate iter_events yielding Completed
    completed = Completed(metrics={}, logs=[])

    async def mock_iter_events(**kwargs):
        yield completed

    mock_handle.iter_events = mock_iter_events
    mock_handle.get = AsyncMock(
        return_value={
            "audio": {
                "url": "https://fal.ai/files/audio.mp3",
                "content_type": "audio/mpeg",
            },
            "duration_ms": 3500,
        }
    )

    mock_client = AsyncMock()
    mock_client.submit = AsyncMock(return_value=mock_handle)

    with (
        patch.object(handler, "_get_client", return_value=mock_client),
        patch(
            "tarash.tarash_gateway.providers.fal.download_media_from_url_async",
            return_value=(b"audio-bytes-here", "audio/mpeg"),
        ) as mock_download,
    ):
        response = await handler.generate_tts_async(base_config, tts_request)

    assert isinstance(response, TTSResponse)
    assert response.request_id == "fal-req-123"
    assert response.status == "completed"
    assert response.content_type == "audio/mpeg"
    assert response.duration == 3.5
    assert response.audio == base64.b64encode(b"audio-bytes-here").decode("utf-8")

    mock_download.assert_called_once_with(
        "https://fal.ai/files/audio.mp3", provider="fal"
    )


def test_generate_tts_sync_success(handler, base_config, tts_request):
    """Sync TTS generation with mocked Fal client."""
    from fal_client import Completed

    mock_handle = MagicMock()
    mock_handle.request_id = "fal-req-456"

    completed = Completed(metrics={}, logs=[])
    mock_handle.iter_events = MagicMock(return_value=iter([completed]))
    mock_handle.get = MagicMock(
        return_value={
            "audio": {"url": "https://fal.ai/files/audio.mp3"},
            "duration_ms": 1200,
        }
    )

    mock_client = MagicMock()
    mock_client.submit = MagicMock(return_value=mock_handle)

    with (
        patch.object(handler, "_get_client", return_value=mock_client),
        patch(
            "tarash.tarash_gateway.providers.fal.download_media_from_url",
            return_value=(b"sync-audio", "audio/mpeg"),
        ),
    ):
        response = handler.generate_tts(base_config, tts_request)

    assert isinstance(response, TTSResponse)
    assert response.request_id == "fal-req-456"
    assert response.status == "completed"
    assert response.duration == 1.2
    assert response.audio == base64.b64encode(b"sync-audio").decode("utf-8")


@pytest.mark.asyncio
async def test_generate_tts_async_with_progress(handler, base_config, tts_request):
    """Async TTS generation tracks progress callbacks."""
    from fal_client import Completed, Queued

    mock_handle = AsyncMock()
    mock_handle.request_id = "fal-req-789"

    queued = Queued(position=2)
    completed = Completed(metrics={}, logs=[])

    async def mock_iter_events(**kwargs):
        yield queued
        yield completed

    mock_handle.iter_events = mock_iter_events
    mock_handle.get = AsyncMock(
        return_value={
            "audio": {"url": "https://fal.ai/files/audio.mp3"},
            "duration_ms": 2000,
        }
    )

    mock_client = AsyncMock()
    mock_client.submit = AsyncMock(return_value=mock_handle)

    progress_updates = []

    async def progress_callback(update):
        progress_updates.append(update)

    with (
        patch.object(handler, "_get_client", return_value=mock_client),
        patch(
            "tarash.tarash_gateway.providers.fal.download_media_from_url_async",
            return_value=(b"audio", "audio/mpeg"),
        ),
    ):
        response = await handler.generate_tts_async(
            base_config, tts_request, on_progress=progress_callback
        )

    assert response.status == "completed"
    assert len(progress_updates) == 2
    assert progress_updates[0].status == "queued"
    assert progress_updates[1].status == "completed"


# ==================== Qwen 3 TTS Tests ====================


@pytest.fixture
def qwen_tts_config():
    return AudioGenerationConfig(
        model="fal-ai/qwen-3-tts/text-to-speech/1.7b",
        provider="fal",
        api_key="test-api-key",
        timeout=240,
    )


@pytest.fixture
def qwen_voice_design_config():
    return AudioGenerationConfig(
        model="fal-ai/qwen-3-tts/voice-design/1.7b",
        provider="fal",
        api_key="test-api-key",
        timeout=240,
    )


@pytest.fixture
def qwen_tts_request():
    return TTSRequest(text="Hello, how are you?", voice_id="Vivian")


# ==================== Qwen TTS Request Conversion ====================


def test_convert_qwen_tts_request_text_to_speech_minimal(
    handler, qwen_tts_config, qwen_tts_request
):
    """Minimal Qwen text-to-speech request with text and voice."""
    kwargs = handler._convert_tts_request(qwen_tts_config, qwen_tts_request)

    assert kwargs["text"] == "Hello, how are you?"
    assert kwargs["voice"] == "Vivian"
    assert "prompt" not in kwargs
    assert "voice_setting" not in kwargs
    assert "audio_setting" not in kwargs
    assert "output_format" not in kwargs


def test_convert_qwen_tts_request_text_to_speech_full(handler, qwen_tts_config):
    """Full Qwen text-to-speech request with language, voice_settings, and extra_params."""
    request = TTSRequest(
        text="Bonjour le monde",
        voice_id="Serena",
        language_code="French",
        voice_settings={
            "temperature": 0.7,
            "top_k": 30,
            "top_p": 0.9,
            "repetition_penalty": 1.1,
            "max_new_tokens": 500,
        },
        extra_params={
            "speaker_voice_embedding_file_url": "https://example.com/embedding.safetensors",
            "reference_text": "Sample reference text",
            "subtalker_dosample": False,
        },
    )
    kwargs = handler._convert_tts_request(qwen_tts_config, request)

    assert kwargs["text"] == "Bonjour le monde"
    assert kwargs["voice"] == "Serena"
    assert kwargs["language"] == "French"
    # voice_settings passed as individual fields
    assert kwargs["temperature"] == 0.7
    assert kwargs["top_k"] == 30
    assert kwargs["top_p"] == 0.9
    assert kwargs["repetition_penalty"] == 1.1
    assert kwargs["max_new_tokens"] == 500
    # extra_params merged
    assert (
        kwargs["speaker_voice_embedding_file_url"]
        == "https://example.com/embedding.safetensors"
    )
    assert kwargs["reference_text"] == "Sample reference text"
    assert kwargs["subtalker_dosample"] is False


def test_convert_qwen_tts_request_voice_design(handler, qwen_voice_design_config):
    """Qwen voice-design passes prompt via extra_params, no voice_id needed."""
    request = TTSRequest(
        text="The quick brown fox jumps over the lazy dog.",
        language_code="English",
        voice_settings={"temperature": 0.8},
        extra_params={
            "prompt": "Speak in an incredulous tone, but with a hint of panic."
        },
    )
    kwargs = handler._convert_tts_request(qwen_voice_design_config, request)

    assert kwargs["text"] == "The quick brown fox jumps over the lazy dog."
    assert kwargs["prompt"] == "Speak in an incredulous tone, but with a hint of panic."
    assert "voice" not in kwargs
    assert kwargs["language"] == "English"
    assert kwargs["temperature"] == 0.8


def test_convert_qwen_tts_request_0_6b_variant(handler):
    """Qwen 0.6b text-to-speech variant uses same conversion as 1.7b."""
    config = AudioGenerationConfig(
        model="fal-ai/qwen-3-tts/text-to-speech/0.6b",
        provider="fal",
        api_key="test-api-key",
        timeout=240,
    )
    request = TTSRequest(text="Test", voice_id="Dylan")
    kwargs = handler._convert_tts_request(config, request)

    assert kwargs["text"] == "Test"
    assert kwargs["voice"] == "Dylan"


# ==================== Qwen TTS Response Conversion ====================


def test_convert_qwen_tts_response_with_duration(
    handler, qwen_tts_config, qwen_tts_request
):
    """Qwen response extracts duration from audio dict (in seconds)."""
    audio_bytes = b"fake-audio-output"
    fal_result = {
        "audio": {
            "url": "https://fal.ai/files/audio.mp3",
            "content_type": "audio/mpeg",
            "duration": 3.45,
            "sample_rate": 24000,
            "channels": 1,
        },
    }
    response = handler._convert_tts_response(
        qwen_tts_config, qwen_tts_request, "req-q1", fal_result, audio_bytes
    )

    assert isinstance(response, TTSResponse)
    assert response.request_id == "req-q1"
    assert response.audio == base64.b64encode(audio_bytes).decode("utf-8")
    assert response.content_type == "audio/mpeg"
    assert response.duration == 3.45
    assert response.status == "completed"


def test_convert_qwen_tts_response_without_duration(
    handler, qwen_tts_config, qwen_tts_request
):
    """Qwen response without duration in audio dict has None duration."""
    fal_result = {
        "audio": {
            "url": "https://fal.ai/files/audio.mp3",
        },
    }
    response = handler._convert_tts_response(
        qwen_tts_config, qwen_tts_request, "req-q2", fal_result, b"data"
    )

    assert response.duration is None


# ==================== Qwen Integration Tests (Mocked SDK) ====================


@pytest.mark.asyncio
async def test_qwen_tts_async_success(handler, qwen_tts_config, qwen_tts_request):
    """Async Qwen TTS generation with mocked Fal client."""
    from fal_client import Completed

    mock_handle = AsyncMock()
    mock_handle.request_id = "fal-qwen-123"

    completed = Completed(metrics={}, logs=[])

    async def mock_iter_events(**kwargs):
        yield completed

    mock_handle.iter_events = mock_iter_events
    mock_handle.get = AsyncMock(
        return_value={
            "audio": {
                "url": "https://fal.ai/files/qwen-audio.mp3",
                "content_type": "audio/mpeg",
                "duration": 2.1,
                "sample_rate": 24000,
                "channels": 1,
            },
        }
    )

    mock_client = AsyncMock()
    mock_client.submit = AsyncMock(return_value=mock_handle)

    with (
        patch.object(handler, "_get_client", return_value=mock_client),
        patch(
            "tarash.tarash_gateway.providers.fal.download_media_from_url_async",
            return_value=(b"qwen-audio-bytes", "audio/mpeg"),
        ) as mock_download,
    ):
        response = await handler.generate_tts_async(qwen_tts_config, qwen_tts_request)

    assert isinstance(response, TTSResponse)
    assert response.request_id == "fal-qwen-123"
    assert response.status == "completed"
    assert response.content_type == "audio/mpeg"
    assert response.duration == 2.1
    assert response.audio == base64.b64encode(b"qwen-audio-bytes").decode("utf-8")

    mock_download.assert_called_once_with(
        "https://fal.ai/files/qwen-audio.mp3", provider="fal"
    )
