"""End-to-end tests for Fal TTS provider (MiniMax Speech models).

Requires FAL_KEY environment variable to be set.
Run with: uv run pytest tests/e2e/test_fal_tts.py --e2e -v
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
    TTSUpdate,
)


@pytest.fixture(scope="module")
def fal_api_key():
    """Get Fal API key from environment."""
    api_key = os.getenv("FAL_KEY")
    if not api_key:
        pytest.skip("FAL_KEY environment variable not set")
    return api_key


# ==================== TTS Tests ====================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_minimax_speech_28_hd_async_with_voice_settings(fal_api_key):
    """Async TTS with speech-2.8-hd, voice settings, language boost, progress tracking.

    This tests:
    - MiniMax Speech 2.8 HD model via Fal queue API
    - Progress callback tracking (queued → processing → completed)
    - Voice settings (speed, vol, emotion, pitch, english_normalization)
    - Language boost (English)
    - Interjection support in prompt
    - Pause syntax (<#0.5#>) in prompt
    - Duration extraction from response
    """
    progress_updates = []

    async def progress_callback(update: TTSUpdate):
        progress_updates.append(update)
        print(f"  Progress: {update.status}")

    config = AudioGenerationConfig(
        model="fal-ai/minimax/speech-2.8-hd",
        provider="fal",
        api_key=fal_api_key,
        timeout=120,
    )

    request = TTSRequest(
        text="Hello! (laughs) This is a test of the MiniMax Speech model. <#0.5#> Pretty cool, right?",
        voice_id="Wise_Woman",
        output_format=AudioOutputFormat(format="mp3", sample_rate=44100, bitrate=128),
        language_code="English",
        voice_settings={
            "speed": 1.1,
            "vol": 1.2,
            "emotion": "happy",
            "pitch": 3,
            "english_normalization": True,
        },
    )

    response = await generate_tts_async(config, request, on_progress=progress_callback)

    assert isinstance(response, TTSResponse)
    assert response.status == "completed"
    assert response.request_id is not None
    assert response.content_type == "audio/mpeg"
    assert response.is_mock is False

    # Validate audio is valid base64
    audio_bytes = base64.b64decode(response.audio)
    assert len(audio_bytes) > 0

    # Validate duration is present and reasonable (> 0 seconds)
    assert response.duration is not None
    assert response.duration > 0

    # Validate progress was tracked
    assert len(progress_updates) > 0
    statuses = [u.status for u in progress_updates]
    assert "completed" in statuses

    # Validate execution metadata
    assert response.execution_metadata is not None
    assert response.execution_metadata.total_attempts == 1

    print(f"✓ Generated: {response.request_id}")
    print(f"  Duration: {response.duration:.2f}s")
    print(f"  Audio size: {len(audio_bytes)} bytes")
    print(f"  Progress updates: {len(progress_updates)}")


@pytest.mark.e2e
def test_minimax_speech_28_turbo_sync_full_settings(fal_api_key):
    """Sync TTS with speech-2.8-turbo, flac output, normalization, stereo audio.

    This tests:
    - Different model variant (turbo vs HD)
    - Synchronous generation path
    - Different voice (Deep_Voice_Man)
    - Flac output format with custom sample rate and stereo channel
    - Voice settings: speed, vol, english_normalization
    - Normalization setting (target loudness, range, peak)
    - Language boost (French)
    """
    config = AudioGenerationConfig(
        model="fal-ai/minimax/speech-2.8-turbo",
        provider="fal",
        api_key=fal_api_key,
        timeout=120,
    )

    request = TTSRequest(
        text="Bonjour! Ceci est un test du modèle MiniMax Speech turbo.",
        voice_id="Deep_Voice_Man",
        output_format=AudioOutputFormat(format="flac", sample_rate=44100),
        language_code="French",
        voice_settings={
            "speed": 0.9,
            "vol": 1.5,
            "english_normalization": False,
        },
        extra_params={
            "audio_setting": {"channel": 2},
            "normalization_setting": {
                "enabled": True,
                "target_loudness": -16,
                "target_range": 6,
                "target_peak": -1.0,
            },
        },
    )

    response = generate_tts(config, request)

    assert isinstance(response, TTSResponse)
    assert response.status == "completed"
    assert response.request_id is not None
    assert response.content_type == "audio/flac"

    audio_bytes = base64.b64decode(response.audio)
    assert len(audio_bytes) > 0

    assert response.duration is not None
    assert response.duration > 0

    assert response.execution_metadata is not None
    assert response.execution_metadata.total_attempts == 1

    print(f"✓ Sync generated: {response.request_id}")
    print(f"  Duration: {response.duration:.2f}s")
    print(f"  Audio size: {len(audio_bytes)} bytes")
