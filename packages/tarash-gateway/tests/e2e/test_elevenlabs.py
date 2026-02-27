"""End-to-end tests for ElevenLabs TTS and STS provider."""

import base64
import os

import pytest

from tarash.tarash_gateway import (
    generate_tts,
    generate_tts_async,
    generate_sts_async,
)
from tarash.tarash_gateway.models import (
    AudioGenerationConfig,
    STSRequest,
    TTSRequest,
    TTSResponse,
    STSResponse,
)


@pytest.fixture(scope="module")
def elevenlabs_api_key():
    api_key = os.getenv("ELEVENLABS_API_KEY")
    if not api_key:
        pytest.skip("ELEVENLABS_API_KEY environment variable not set")
    return api_key


# Rachel voice — a commonly available default voice
RACHEL_VOICE_ID = "21m00Tcm4TlvDq8ikWAM"


# ==================== TTS Tests ====================


@pytest.mark.e2e
async def test_tts_async_multilingual(elevenlabs_api_key):
    """Async TTS with eleven_multilingual_v2, custom voice_settings, wav output, language_code."""
    config = AudioGenerationConfig(
        provider="elevenlabs",
        model="eleven_multilingual_v2",
        api_key=elevenlabs_api_key,
    )
    request = TTSRequest(
        text="Hello, this is a test of the ElevenLabs multilingual TTS integration.",
        voice_id=RACHEL_VOICE_ID,
        output_format="mp3_44100_128",
        language_code="en",
        voice_settings={"stability": 0.5, "similarity_boost": 0.75},
    )

    response = await generate_tts_async(config, request)

    assert isinstance(response, TTSResponse)
    assert response.status == "completed"
    assert response.request_id is not None
    assert response.content_type == "audio/mpeg"
    assert response.is_mock is False

    # Verify audio is valid base64
    audio_bytes = base64.b64decode(response.audio)
    assert len(audio_bytes) > 0

    # Verify execution metadata
    assert response.execution_metadata is not None
    assert response.execution_metadata.total_attempts == 1

    print(f"  TTS async multilingual: {len(audio_bytes)} bytes audio")


@pytest.mark.e2e
def test_tts_sync_flash(elevenlabs_api_key):
    """Sync TTS with eleven_flash_v2_5, default output format, seed parameter."""
    config = AudioGenerationConfig(
        provider="elevenlabs",
        model="eleven_flash_v2_5",
        api_key=elevenlabs_api_key,
    )
    request = TTSRequest(
        text="Quick test of the flash model.",
        voice_id=RACHEL_VOICE_ID,
        seed=42,
    )

    response = generate_tts(config, request)

    assert isinstance(response, TTSResponse)
    assert response.status == "completed"
    assert response.content_type == "audio/mpeg"

    # Verify audio is valid base64
    audio_bytes = base64.b64decode(response.audio)
    assert len(audio_bytes) > 0

    print(f"  TTS sync flash: {len(audio_bytes)} bytes audio")


# ==================== STS Tests ====================


@pytest.mark.e2e
async def test_sts_async(elevenlabs_api_key):
    """Async STS: generate TTS audio first, then convert voice via STS."""
    # First, generate source audio via TTS
    tts_config = AudioGenerationConfig(
        provider="elevenlabs",
        model="eleven_flash_v2_5",
        api_key=elevenlabs_api_key,
    )
    tts_request = TTSRequest(
        text="This audio will be converted to a different voice.",
        voice_id=RACHEL_VOICE_ID,
        output_format="mp3_44100_128",
    )
    tts_response = await generate_tts_async(tts_config, tts_request)
    source_audio_bytes = base64.b64decode(tts_response.audio)

    # Now convert via STS
    sts_config = AudioGenerationConfig(
        provider="elevenlabs",
        model="eleven_multilingual_sts_v2",
        api_key=elevenlabs_api_key,
    )
    sts_request = STSRequest(
        audio={"content": source_audio_bytes, "content_type": "audio/mpeg"},
        voice_id=RACHEL_VOICE_ID,
        output_format="mp3_44100_128",
    )

    response = await generate_sts_async(sts_config, sts_request)

    assert isinstance(response, STSResponse)
    assert response.status == "completed"
    assert response.content_type == "audio/mpeg"

    audio_bytes = base64.b64decode(response.audio)
    assert len(audio_bytes) > 0

    print(f"  STS async: {len(audio_bytes)} bytes audio")
