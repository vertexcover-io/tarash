"""End-to-end tests for Sarvam AI TTS provider.

Requires SARVAM_API_KEY environment variable to be set.
Run with: uv run pytest tests/e2e/test_sarvam.py --e2e -v
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
def sarvam_api_key():
    """Get Sarvam API key from environment."""
    api_key = os.getenv("SARVAM_API_KEY")
    if not api_key:
        pytest.skip("SARVAM_API_KEY environment variable not set")
    return api_key


# ==================== TTS Tests ====================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_tts_async_v3_with_all_features(sarvam_api_key):
    """Async TTS with bulbul:v3 model, mp3 output, pace, temperature, speaker=Shubh, language=hi-IN."""
    config = AudioGenerationConfig(
        provider="sarvam",
        model="bulbul:v3",
        api_key=sarvam_api_key,
    )
    request = TTSRequest(
        text="नमस्ते, यह सर्वम एआई टीटीएस इंटीग्रेशन का परीक्षण है।",
        voice_id="shubh",
        language_code="hi-IN",
        output_format=AudioOutputFormat(format="mp3", sample_rate=24000),
        voice_settings={"pace": 1.2, "temperature": 0.8},
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
def test_tts_sync_v2_with_voice_controls(sarvam_api_key):
    """Sync TTS with bulbul:v2, wav output, pitch, loudness, pace, speaker=anushka, language=en-IN."""
    config = AudioGenerationConfig(
        provider="sarvam",
        model="bulbul:v2",
        api_key=sarvam_api_key,
    )
    request = TTSRequest(
        text="Hello, this is a test of the Sarvam AI text to speech system.",
        voice_id="anushka",
        language_code="en-IN",
        output_format=AudioOutputFormat(format="wav", sample_rate=22050),
        voice_settings={"pace": 0.9, "pitch": 0.25, "loudness": 1.5},
    )

    response = generate_tts(config, request)

    assert isinstance(response, TTSResponse)
    assert response.status == "completed"
    assert response.content_type == "audio/wav"

    audio_bytes = base64.b64decode(response.audio)
    assert len(audio_bytes) > 0
