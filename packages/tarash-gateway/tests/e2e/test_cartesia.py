"""End-to-end tests for Cartesia TTS and STS (Voice Changer) provider.

Requires CARTESIA_API_KEY environment variable to be set.
Run with: uv run pytest tests/e2e/test_cartesia.py --e2e -v
"""

import base64
import os

import pytest

from tarash.tarash_gateway.api import (
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
def cartesia_api_key():
    """Get Cartesia API key from environment."""
    api_key = os.getenv("CARTESIA_API_KEY")
    if not api_key:
        pytest.skip("CARTESIA_API_KEY environment variable not set")
    return api_key


# Use a well-known default Cartesia voice ID
# "Barbershop Man" is a commonly available default voice
BARBERSHOP_MAN_VOICE_ID = "a0e99841-438c-4a64-b679-ae501e7d6091"


# ==================== TTS Tests ====================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_tts_async_sonic3_with_generation_config(cartesia_api_key):
    """Async TTS with sonic-3 model, generation_config (emotion + speed + volume), language, mp3 output."""
    config = AudioGenerationConfig(
        provider="cartesia",
        model="sonic-3",
        api_key=cartesia_api_key,
    )
    request = TTSRequest(
        text="Hello, this is a test of the Cartesia TTS integration with emotion control.",
        voice_id=BARBERSHOP_MAN_VOICE_ID,
        output_format="mp3_44100_128",
        language_code="en",
        extra_params={
            "generation_config": {"emotion": "happy", "speed": 1.1, "volume": 0.9},
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

    # Validate execution metadata
    assert response.execution_metadata is not None
    assert response.execution_metadata.total_attempts == 1


@pytest.mark.e2e
def test_tts_sync_sonic_turbo(cartesia_api_key):
    """Sync TTS with sonic-turbo model, minimal params, wav output."""
    config = AudioGenerationConfig(
        provider="cartesia",
        model="sonic-turbo",
        api_key=cartesia_api_key,
    )
    request = TTSRequest(
        text="Quick test of the sonic turbo model.",
        voice_id=BARBERSHOP_MAN_VOICE_ID,
        output_format="wav_44100",
    )

    response = generate_tts(config, request)

    assert isinstance(response, TTSResponse)
    assert response.status == "completed"
    assert response.content_type == "audio/wav"

    audio_bytes = base64.b64decode(response.audio)
    assert len(audio_bytes) > 0


# ==================== STS (Voice Changer) Tests ====================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_sts_async_voice_changer(cartesia_api_key):
    """Async STS: generate TTS audio first, then convert via Voice Changer."""
    # Step 1: Generate source audio via TTS
    tts_config = AudioGenerationConfig(
        provider="cartesia",
        model="sonic-turbo",
        api_key=cartesia_api_key,
    )
    tts_request = TTSRequest(
        text="This audio will be converted to a different voice.",
        voice_id=BARBERSHOP_MAN_VOICE_ID,
        output_format="mp3_44100_128",
    )
    tts_response = await generate_tts_async(tts_config, tts_request)
    source_audio_bytes = base64.b64decode(tts_response.audio)

    # Step 2: Convert via Voice Changer
    sts_config = AudioGenerationConfig(
        provider="cartesia",
        model="sonic-3",  # Voice changer uses same model param
        api_key=cartesia_api_key,
    )
    sts_request = STSRequest(
        audio={"content": source_audio_bytes, "content_type": "audio/mpeg"},
        voice_id=BARBERSHOP_MAN_VOICE_ID,
        output_format="mp3_44100_128",
    )
    response = await generate_sts_async(sts_config, sts_request)

    assert isinstance(response, STSResponse)
    assert response.status == "completed"
    assert response.content_type == "audio/mpeg"

    audio_bytes = base64.b64decode(response.audio)
    assert len(audio_bytes) > 0
