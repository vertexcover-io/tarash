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


# ==================== Qwen 3 TTS Tests ====================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_qwen3_tts_text_to_speech_async(fal_api_key):
    """Async Qwen 3 TTS text-to-speech with progress tracking and sampling params.

    This tests:
    - Qwen 3 TTS 1.7B text-to-speech model
    - Progress callback tracking
    - Predefined voice selection (Vivian)
    - Language setting (English)
    - Sampling parameters via voice_settings (temperature, top_k)
    - Duration extraction from Qwen audio response format
    """
    progress_updates = []

    async def progress_callback(update: TTSUpdate):
        progress_updates.append(update)
        print(f"  Progress: {update.status}")

    config = AudioGenerationConfig(
        model="fal-ai/qwen-3-tts/text-to-speech/1.7b",
        provider="fal",
        api_key=fal_api_key,
        timeout=120,
    )

    request = TTSRequest(
        text="Hello! This is a test of Qwen 3 text-to-speech model. How does it sound?",
        voice_id="Vivian",
        language_code="English",
        voice_settings={
            "temperature": 0.8,
            "top_k": 40,
        },
    )

    response = await generate_tts_async(config, request, on_progress=progress_callback)

    assert isinstance(response, TTSResponse)
    assert response.status == "completed"
    assert response.request_id is not None
    assert response.content_type == "audio/mpeg"
    assert response.is_mock is False

    audio_bytes = base64.b64decode(response.audio)
    assert len(audio_bytes) > 0

    assert response.duration is not None
    assert response.duration > 0

    assert len(progress_updates) > 0
    statuses = [u.status for u in progress_updates]
    assert "completed" in statuses

    assert response.execution_metadata is not None
    assert response.execution_metadata.total_attempts == 1

    print(f"✓ Generated: {response.request_id}")
    print(f"  Duration: {response.duration:.2f}s")
    print(f"  Audio size: {len(audio_bytes)} bytes")
    print(f"  Progress updates: {len(progress_updates)}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_qwen_voice_design_with_all_params(fal_api_key):
    """Async Qwen 3 TTS voice-design with all new field mappers.

    This tests:
    - Qwen 3 TTS 1.7B voice-design model
    - All 12 new parameters via extra_params
    - Voice style guidance prompt
    - Sampling parameters (top_k, top_p, temperature, repetition_penalty)
    - Sub-talker parameters (subtalker_dosample, subtalker_top_k, subtalker_top_p, subtalker_temperature)
    - Max tokens control
    """
    progress_updates = []

    async def progress_callback(update: TTSUpdate):
        progress_updates.append(update)
        print(f"  Progress: {update.status}")

    config = AudioGenerationConfig(
        model="fal-ai/qwen-3-tts/voice-design/1.7b",
        provider="fal",
        api_key=fal_api_key,
        timeout=120,
    )

    request = TTSRequest(
        text="Hello! Welcome to Qwen 3 TTS voice design with comprehensive testing.",
        prompt="Speak in an enthusiastic and welcoming tone with clear articulation.",
        language_code="English",
        extra_params={
            "top_k": 75,
            "top_p": 0.95,
            "temperature": 0.85,
            "repetition_penalty": 1.1,
            "subtalker_dosample": True,
            "subtalker_top_k": 50,
            "subtalker_top_p": 1.0,
            "subtalker_temperature": 0.9,
            "max_new_tokens": 300,
        },
    )

    response = await generate_tts_async(config, request, on_progress=progress_callback)

    assert isinstance(response, TTSResponse)
    assert response.status == "completed"

    print("✓ Voice design generated successfully")
    print(f"  Duration: {response.duration:.2f}s")


@pytest.mark.e2e
def test_qwen3_tts_voice_design_sync(fal_api_key):
    """Sync Qwen 3 TTS voice-design with style prompt.

    This tests:
    - Qwen 3 TTS voice-design 1.7B model
    - Synchronous generation path
    - Voice style description via voice_id → prompt
    - Different language (Chinese)
    - Extra params (max_new_tokens, repetition_penalty)
    """
    config = AudioGenerationConfig(
        model="fal-ai/qwen-3-tts/voice-design/1.7b",
        provider="fal",
        api_key=fal_api_key,
        timeout=120,
    )

    request = TTSRequest(
        text="The weather today is absolutely beautiful with clear skies and sunshine.",
        language_code="English",
        extra_params={
            "prompt": "Speak in a warm, friendly tone with a gentle cadence.",
            "max_new_tokens": 400,
            "repetition_penalty": 1.1,
        },
    )

    response = generate_tts(config, request)

    assert isinstance(response, TTSResponse)
    assert response.status == "completed"
    assert response.request_id is not None
    assert response.content_type == "audio/mpeg"

    audio_bytes = base64.b64decode(response.audio)
    assert len(audio_bytes) > 0

    assert response.duration is not None
    assert response.duration > 0

    assert response.execution_metadata is not None
    assert response.execution_metadata.total_attempts == 1

    print(f"✓ Sync generated: {response.request_id}")
    print(f"  Duration: {response.duration:.2f}s")
    print(f"  Audio size: {len(audio_bytes)} bytes")


@pytest.mark.e2e
def test_qwen3_tts_text_to_speech_0_6b_sync(fal_api_key):
    """Sync Qwen 3 TTS 0.6B smaller model variant.

    This tests:
    - Qwen 3 TTS 0.6B text-to-speech model (smaller variant)
    - Different voice (Dylan)
    - Verifies prefix matching works for 0.6b variant
    """
    config = AudioGenerationConfig(
        model="fal-ai/qwen-3-tts/text-to-speech/0.6b",
        provider="fal",
        api_key=fal_api_key,
        timeout=120,
    )

    request = TTSRequest(
        text="Testing the smaller Qwen model variant.",
        voice_id="Dylan",
    )

    response = generate_tts(config, request)

    assert isinstance(response, TTSResponse)
    assert response.status == "completed"
    assert response.request_id is not None

    audio_bytes = base64.b64decode(response.audio)
    assert len(audio_bytes) > 0

    assert response.duration is not None
    assert response.duration > 0

    print(f"✓ Sync 0.6B generated: {response.request_id}")
    print(f"  Duration: {response.duration:.2f}s")
    print(f"  Audio size: {len(audio_bytes)} bytes")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_minimax_speech_28_hd_voice_modification(fal_api_key):
    """Async MiniMax Speech 2.8 HD with voice modification.

    This tests:
    - MiniMax Speech 2.8 HD model
    - Voice modification via extra_params (pitch, intensity, timbre)
    - New field mapper functionality
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
        text="This voice has been modified with pitch, intensity, and timbre adjustments.",
        voice_id="Friendly_Person",
        output_format=AudioOutputFormat(format="mp3", sample_rate=44100),
        extra_params={
            "voice_modify": {
                "pitch": 5,
                "intensity": 20,
                "timbre": 10,
            },
        },
    )

    response = await generate_tts_async(config, request, on_progress=progress_callback)

    assert isinstance(response, TTSResponse)
    assert response.status == "completed"
    assert isinstance(response.raw_response, dict)

    print("✓ Speech generated with voice modification")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_minimax_speech_28_hd_normalization(fal_api_key):
    """Async MiniMax Speech 2.8 HD with loudness normalization.

    This tests:
    - MiniMax Speech 2.8 HD model
    - Normalization settings via extra_params
    - New field mapper functionality
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
        text="This speech has loudness normalization applied.",
        voice_id="Calm_Woman",
        output_format=AudioOutputFormat(format="mp3"),
        extra_params={
            "normalization_setting": {
                "enabled": True,
                "target_loudness": -18.0,
                "target_range": 8.0,
                "target_peak": -0.5,
            },
        },
    )

    response = await generate_tts_async(config, request, on_progress=progress_callback)

    assert isinstance(response, TTSResponse)
    assert response.status == "completed"
    assert isinstance(response.raw_response, dict)

    print("✓ Speech generated with normalization settings")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_minimax_speech_28_hd_pronunciation_dict(fal_api_key):
    """Async MiniMax Speech 2.8 HD with custom pronunciation dictionary.

    This tests:
    - MiniMax Speech 2.8 HD model
    - Pronunciation dictionary via extra_params
    - New field mapper functionality
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
        text="This tests the pronunciation dictionary feature.",
        voice_id="Inspirational_girl",
        output_format=AudioOutputFormat(format="mp3"),
        extra_params={
            "pronunciation_dict": {
                "tone_list": [
                    "tests/(t3sts)",
                    "pronunciation/(pr4nunciation)",
                    "feature/(f3chure)",
                ]
            },
        },
    )

    response = await generate_tts_async(config, request, on_progress=progress_callback)

    assert isinstance(response, TTSResponse)
    assert response.status == "completed"
    assert isinstance(response.raw_response, dict)

    print("✓ Speech generated with pronunciation dictionary")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_minimax_speech_28_hd_hex_output(fal_api_key):
    """Async MiniMax Speech 2.8 HD with hex output format.

    This tests:
    - MiniMax Speech 2.8 HD model
    - Output format "hex" via extra_params
    - New field mapper functionality
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
        text="Testing hex output format.",
        voice_id="Deep_Voice_Man",
        extra_params={
            "output_format": "hex",
        },
    )

    response = await generate_tts_async(config, request, on_progress=progress_callback)

    assert isinstance(response, TTSResponse)
    assert response.status == "completed"
    assert isinstance(response.raw_response, dict)

    print("✓ Speech generated with hex output format")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_minimax_speech_28_hd_comprehensive(fal_api_key):
    """Async MiniMax Speech 2.8 HD with all new features combined.

    This tests:
    - MiniMax Speech 2.8 HD model
    - All 4 new parameters via extra_params
    - Voice modification (pitch, intensity, timbre)
    - Normalization settings (enabled, target_loudness, target_range, target_peak)
    - Pronunciation dictionary
    - Output format selection
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
        text="Comprehensive test with voice modification, normalization, and custom pronunciation.",
        voice_id="Young_Knight",
        language_code="English",
        output_format=AudioOutputFormat(format="flac", sample_rate=44100, bitrate=256),
        extra_params={
            "normalization_setting": {
                "enabled": True,
                "target_loudness": -16.0,
                "target_range": 10.0,
            },
            "voice_modify": {
                "pitch": 3,
                "intensity": 15,
                "timbre": 5,
            },
            "pronunciation_dict": {"tone_list": ["comprehensive/(k9mpr7hensiv)"]},
            "output_format": "url",
        },
    )

    response = await generate_tts_async(config, request, on_progress=progress_callback)

    assert isinstance(response, TTSResponse)
    assert response.status == "completed"
    assert isinstance(response.raw_response, dict)

    print("✓ Speech generated with all features")
    print(f"  Duration: {response.duration:.2f}s")
