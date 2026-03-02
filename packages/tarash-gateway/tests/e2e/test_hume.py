"""End-to-end tests for Hume AI TTS provider.

Requires HUME_API_KEY environment variable to be set.
Run with: uv run pytest tests/e2e/test_hume.py --e2e -v
"""

import base64
import os

import pytest

from tarash.tarash_gateway.api import (
    generate_tts,
    generate_tts_async,
)
from tarash.tarash_gateway.models import (
    AudioGenerationConfig,
    AudioOutputFormat,
    TTSRequest,
    TTSResponse,
)


@pytest.fixture(scope="module")
def hume_api_key():
    """Get Hume API key from environment."""
    api_key = os.getenv("HUME_API_KEY")
    if not api_key:
        pytest.skip("HUME_API_KEY environment variable not set")
    return api_key


# ==================== TTS Tests ====================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_tts_async_v1_with_acting_instructions(hume_api_key):
    """Async TTS with octave-v1: acting instructions, speed, trailing_silence, mp3, voice by name."""
    config = AudioGenerationConfig(
        provider="hume",
        model="octave-v1",
        api_key=hume_api_key,
    )
    request = TTSRequest(
        text="The art of speech synthesis has come a long way. Each word carries emotion and meaning.",
        voice_id="Kora",
        output_format=AudioOutputFormat(format="mp3"),
        voice_settings={
            "description": "calm, warm, pedagogical",
            "speed": 0.9,
            "trailing_silence": 0.5,
        },
    )

    response = await generate_tts_async(config, request)

    assert isinstance(response, TTSResponse)
    assert response.status == "completed"
    assert response.request_id is not None
    assert response.content_type == "audio/mpeg"
    assert response.is_mock is False

    # Validate audio is valid base64
    audio_bytes = base64.b64decode(response.audio)
    assert len(audio_bytes) > 0

    # Validate duration is present
    assert response.duration is not None
    assert response.duration > 0

    # Validate raw_response metadata
    assert response.raw_response["generation_id"] is not None
    assert response.raw_response["model"] == "octave-v1"

    # Validate execution metadata
    assert response.execution_metadata is not None
    assert response.execution_metadata.total_attempts == 1

    print(
        f"  Async v1 TTS: {len(audio_bytes)} bytes, duration={response.duration:.2f}s"
    )


@pytest.mark.e2e
def test_tts_sync_v2_wav_output(hume_api_key):
    """Sync TTS with octave-v2: wav output, different voice, speed control."""
    config = AudioGenerationConfig(
        provider="hume",
        model="octave-v2",
        api_key=hume_api_key,
    )
    request = TTSRequest(
        text="Welcome to the future of voice technology. Every sentence is unique.",
        voice_id="Ava Song",
        output_format=AudioOutputFormat(format="wav"),
        voice_settings={
            "speed": 1.2,
        },
    )

    response = generate_tts(config, request)

    assert isinstance(response, TTSResponse)
    assert response.status == "completed"
    assert response.content_type == "audio/wav"

    audio_bytes = base64.b64decode(response.audio)
    assert len(audio_bytes) > 0

    # Validate duration
    assert response.duration is not None
    assert response.duration > 0

    print(f"  Sync v2 TTS: {len(audio_bytes)} bytes, duration={response.duration:.2f}s")
