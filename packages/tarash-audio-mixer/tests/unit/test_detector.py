"""Tests for speech detection using Silero VAD."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tarash.tarash_audio_mixer.exceptions import DetectionError
from tarash.tarash_audio_mixer.models import AudioMixerConfig, SpeechSegment


@pytest.fixture
def default_config():
    """Default audio mixer config."""
    return AudioMixerConfig()


@pytest.fixture
def custom_threshold_config():
    """Config with custom VAD threshold."""
    return AudioMixerConfig(vad_threshold=0.7)


@pytest.fixture
def audio_path():
    """Dummy audio path for testing."""
    return Path("/tmp/test_audio.wav")


def test_detect_returns_speech_segments(default_config, audio_path):
    """Mock silero_vad imports, verify SpeechSegment list returned."""
    mock_model = MagicMock()
    mock_timestamps = [
        {"start": 0.5, "end": 2.3},
        {"start": 4.0, "end": 6.1},
    ]

    with patch(
        "tarash.tarash_audio_mixer.detector._silero_available",
        return_value=True,
    ):
        with patch.dict(
            "sys.modules", {"silero_vad": MagicMock(), "torch": MagicMock()}
        ):
            with patch(
                "tarash.tarash_audio_mixer.detector.detect_speech_segments"
            ) as _:
                # Re-import to use patched modules; instead, patch at function level
                pass

            # Directly patch the imports inside the function
            import tarash.tarash_audio_mixer.detector as det_module

            mock_torch = MagicMock()
            mock_silero = MagicMock()
            mock_silero.load_silero_vad.return_value = mock_model
            mock_silero.read_audio.return_value = MagicMock()
            mock_silero.get_speech_timestamps.return_value = mock_timestamps

            with patch.dict(
                "sys.modules",
                {"torch": mock_torch, "silero_vad": mock_silero},
            ):
                segments = det_module.detect_speech_segments(audio_path, default_config)

    assert len(segments) == 2
    assert segments[0] == SpeechSegment(start=0.5, end=2.3)
    assert segments[1] == SpeechSegment(start=4.0, end=6.1)


def test_detect_no_speech_returns_empty_list(default_config, audio_path):
    """Mock returns empty timestamps, verify empty list."""
    mock_model = MagicMock()

    with patch(
        "tarash.tarash_audio_mixer.detector._silero_available",
        return_value=True,
    ):
        mock_torch = MagicMock()
        mock_silero = MagicMock()
        mock_silero.load_silero_vad.return_value = mock_model
        mock_silero.read_audio.return_value = MagicMock()
        mock_silero.get_speech_timestamps.return_value = []

        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "silero_vad": mock_silero},
        ):
            from tarash.tarash_audio_mixer.detector import detect_speech_segments

            segments = detect_speech_segments(audio_path, default_config)

    assert segments == []


def test_detect_raises_when_silero_unavailable(default_config, audio_path):
    """Mock _silero_available to return False, verify DetectionError with install hint."""
    with patch(
        "tarash.tarash_audio_mixer.detector._silero_available",
        return_value=False,
    ):
        from tarash.tarash_audio_mixer.detector import detect_speech_segments

        with pytest.raises(DetectionError, match="silero-vad is not installed"):
            detect_speech_segments(audio_path, default_config)


def test_detect_raises_on_model_load_failure(default_config, audio_path):
    """Mock model loading to raise, verify DetectionError."""
    with patch(
        "tarash.tarash_audio_mixer.detector._silero_available",
        return_value=True,
    ):
        mock_torch = MagicMock()
        mock_silero = MagicMock()
        mock_silero.load_silero_vad.side_effect = RuntimeError("Model load failed")

        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "silero_vad": mock_silero},
        ):
            from tarash.tarash_audio_mixer.detector import detect_speech_segments

            with pytest.raises(DetectionError, match="Failed to load Silero VAD model"):
                detect_speech_segments(audio_path, default_config)


def test_detect_raises_on_processing_failure(default_config, audio_path):
    """Mock get_speech_timestamps to raise, verify DetectionError."""
    mock_model = MagicMock()

    with patch(
        "tarash.tarash_audio_mixer.detector._silero_available",
        return_value=True,
    ):
        mock_torch = MagicMock()
        mock_silero = MagicMock()
        mock_silero.load_silero_vad.return_value = mock_model
        mock_silero.read_audio.return_value = MagicMock()
        mock_silero.get_speech_timestamps.side_effect = RuntimeError("Processing error")

        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "silero_vad": mock_silero},
        ):
            from tarash.tarash_audio_mixer.detector import detect_speech_segments

            with pytest.raises(DetectionError, match="Silero VAD processing failed"):
                detect_speech_segments(audio_path, default_config)


async def test_detect_async_delegates_to_thread(default_config, audio_path):
    """Mock asyncio.to_thread, verify it's called with sync detection."""
    expected = [SpeechSegment(start=1.0, end=3.0)]

    with patch(
        "tarash.tarash_audio_mixer.detector._silero_available",
        return_value=True,
    ):
        with patch(
            "tarash.tarash_audio_mixer.detector.asyncio.to_thread",
            new_callable=AsyncMock,
            return_value=expected,
        ) as mock_to_thread:
            from tarash.tarash_audio_mixer.detector import (
                detect_speech_segments,
                detect_speech_segments_async,
            )

            segments = await detect_speech_segments_async(audio_path, default_config)

    assert segments == expected
    mock_to_thread.assert_called_once_with(
        detect_speech_segments, audio_path, default_config
    )


async def test_detect_async_raises_when_silero_unavailable(default_config, audio_path):
    """Async version also raises DetectionError when silero unavailable."""
    with patch(
        "tarash.tarash_audio_mixer.detector._silero_available",
        return_value=False,
    ):
        from tarash.tarash_audio_mixer.detector import detect_speech_segments_async

        with pytest.raises(DetectionError, match="silero-vad is not installed"):
            await detect_speech_segments_async(audio_path, default_config)


def test_detect_uses_config_vad_threshold(custom_threshold_config, audio_path):
    """Verify threshold from config is passed to get_speech_timestamps."""
    mock_model = MagicMock()

    with patch(
        "tarash.tarash_audio_mixer.detector._silero_available",
        return_value=True,
    ):
        mock_torch = MagicMock()
        mock_silero = MagicMock()
        mock_silero.load_silero_vad.return_value = mock_model
        mock_silero.read_audio.return_value = MagicMock()
        mock_silero.get_speech_timestamps.return_value = []

        with patch.dict(
            "sys.modules",
            {"torch": mock_torch, "silero_vad": mock_silero},
        ):
            from tarash.tarash_audio_mixer.detector import detect_speech_segments

            detect_speech_segments(audio_path, custom_threshold_config)

            mock_silero.get_speech_timestamps.assert_called_once()
            call_kwargs = mock_silero.get_speech_timestamps.call_args
            assert call_kwargs.kwargs["threshold"] == 0.7
