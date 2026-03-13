"""Tests for public API orchestration (Phase 5)."""

from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

from tarash.tarash_audio_mixer.api import _resolve_output_path
from tarash.tarash_audio_mixer.exceptions import InvalidInputError
from tarash.tarash_audio_mixer.models import (
    AudioMixerConfig,
    AudioMixerRequest,
    AudioMixerResponse,
    SpeechSegment,
)
from tarash.tarash_audio_mixer.processor import AudioInfo


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def config():
    """Default AudioMixerConfig."""
    return AudioMixerConfig()


@pytest.fixture
def speech_segments():
    """Sample speech segments returned by the detector."""
    return [
        SpeechSegment(start=1.0, end=3.0),
        SpeechSegment(start=5.0, end=7.0),
    ]


@pytest.fixture
def fg_info():
    """Foreground AudioInfo."""
    return AudioInfo(duration=10.0, sample_rate=44100, channels=2)


@pytest.fixture
def bg_info():
    """Background AudioInfo."""
    return AudioInfo(duration=30.0, sample_rate=44100, channels=2)


@pytest.fixture
def output_info():
    """Output AudioInfo after mixing."""
    return AudioInfo(duration=10.0, sample_rate=44100, channels=2)


@pytest.fixture
def tmp_audio_files(tmp_path):
    """Create temporary foreground and background audio files."""
    fg = tmp_path / "speech.wav"
    fg.write_text("fake fg")
    bg = tmp_path / "music.mp3"
    bg.write_text("fake bg")
    return fg, bg


# ---------------------------------------------------------------------------
# _resolve_output_path tests
# ---------------------------------------------------------------------------


def test_resolve_output_path_explicit():
    """Provided output_path is returned as-is."""
    fg = Path("/audio/speech.wav")
    explicit = Path("/output/mixed.wav")
    result = _resolve_output_path(fg, explicit, None)
    assert result == explicit


def test_resolve_output_path_default():
    """Generates {stem}_mixed.{ext} when no output_path or format given (REQ-018)."""
    fg = Path("/audio/speech.wav")
    result = _resolve_output_path(fg, None, None)
    assert result == Path("/audio/speech_mixed.wav")


def test_resolve_output_path_custom_format():
    """Uses output_format extension when provided (REQ-016)."""
    fg = Path("/audio/speech.wav")
    result = _resolve_output_path(fg, None, "mp3")
    assert result == Path("/audio/speech_mixed.mp3")


def test_resolve_output_path_default_format_matches_foreground():
    """Suffix matches foreground when no output_format given (REQ-017)."""
    fg = Path("/audio/podcast.flac")
    result = _resolve_output_path(fg, None, None)
    assert result == Path("/audio/podcast_mixed.flac")
    assert result.suffix == fg.suffix


# ---------------------------------------------------------------------------
# mix_audio validation tests
# ---------------------------------------------------------------------------


def test_mix_audio_validates_foreground_exists(config):
    """Non-existent foreground raises InvalidInputError."""
    from tarash.tarash_audio_mixer.api import mix_audio

    request = AudioMixerRequest(
        foreground_path=Path("/nonexistent/speech.wav"),
        background_path=Path("/nonexistent/music.mp3"),
    )
    with pytest.raises(InvalidInputError, match="Foreground file does not exist"):
        mix_audio(config, request)


def test_mix_audio_validates_background_exists(config, tmp_path):
    """Non-existent background raises InvalidInputError."""
    from tarash.tarash_audio_mixer.api import mix_audio

    fg = tmp_path / "speech.wav"
    fg.write_text("fake")
    request = AudioMixerRequest(
        foreground_path=fg,
        background_path=Path("/nonexistent/music.mp3"),
    )
    with pytest.raises(InvalidInputError, match="Background file does not exist"):
        mix_audio(config, request)


# ---------------------------------------------------------------------------
# mix_audio full orchestration tests
# ---------------------------------------------------------------------------


def _patch_mix_internals(speech_segments, fg_info, bg_info, output_info, loops_used=0):
    """Return a dict of patch context managers for mix_audio internals."""
    return {
        "probe": patch(
            "tarash.tarash_audio_mixer.api.probe_audio_info",
            side_effect=[fg_info, bg_info, output_info],
        ),
        "detect": patch(
            "tarash.tarash_audio_mixer.api.detector.detect_speech_segments",
            return_value=speech_segments,
        ),
        "run": patch(
            "tarash.tarash_audio_mixer.api.run_mix",
            return_value=loops_used,
        ),
    }


def test_mix_audio_returns_response(
    config, tmp_audio_files, speech_segments, fg_info, bg_info, output_info
):
    """Mock all internals, verify AudioMixerResponse fields (REQ-019)."""
    from tarash.tarash_audio_mixer.api import mix_audio

    fg, bg = tmp_audio_files
    request = AudioMixerRequest(foreground_path=fg, background_path=bg)

    patches = _patch_mix_internals(speech_segments, fg_info, bg_info, output_info)

    with patches["probe"], patches["detect"], patches["run"]:
        response = mix_audio(config, request)

    assert isinstance(response, AudioMixerResponse)
    assert response.output_path == fg.parent / f"{fg.stem}_mixed{fg.suffix}"
    assert response.foreground_duration == fg_info.duration
    assert response.background_duration == bg_info.duration
    assert response.output_duration == output_info.duration


def test_mix_audio_response_has_speech_segments(
    config, tmp_audio_files, speech_segments, fg_info, bg_info, output_info
):
    """Segments from detector are included in response."""
    from tarash.tarash_audio_mixer.api import mix_audio

    fg, bg = tmp_audio_files
    request = AudioMixerRequest(foreground_path=fg, background_path=bg)

    patches = _patch_mix_internals(speech_segments, fg_info, bg_info, output_info)

    with patches["probe"], patches["detect"], patches["run"]:
        response = mix_audio(config, request)

    assert response.speech_segments == speech_segments
    assert len(response.speech_segments) == 2


def test_mix_audio_response_has_loops_used(
    config, tmp_audio_files, speech_segments, fg_info, bg_info, output_info
):
    """loops_used from processor is included in response."""
    from tarash.tarash_audio_mixer.api import mix_audio

    fg, bg = tmp_audio_files
    request = AudioMixerRequest(foreground_path=fg, background_path=bg)

    patches = _patch_mix_internals(
        speech_segments, fg_info, bg_info, output_info, loops_used=3
    )

    with patches["probe"], patches["detect"], patches["run"]:
        response = mix_audio(config, request)

    assert response.loops_used == 3


# ---------------------------------------------------------------------------
# mix_audio_async tests
# ---------------------------------------------------------------------------


async def test_mix_audio_async_produces_same_result(
    config, tmp_audio_files, speech_segments, fg_info, bg_info, output_info
):
    """Async variant returns equivalent response (REQ-020)."""
    from tarash.tarash_audio_mixer.api import mix_audio_async

    fg, bg = tmp_audio_files
    request = AudioMixerRequest(foreground_path=fg, background_path=bg)

    with (
        patch(
            "tarash.tarash_audio_mixer.api.probe_audio_info_async",
            side_effect=[fg_info, bg_info, output_info],
        ),
        patch(
            "tarash.tarash_audio_mixer.api.detector.detect_speech_segments_async",
            new_callable=AsyncMock,
            return_value=speech_segments,
        ),
        patch(
            "tarash.tarash_audio_mixer.api.run_mix_async",
            new_callable=AsyncMock,
            return_value=0,
        ),
    ):
        response = await mix_audio_async(config, request)

    assert isinstance(response, AudioMixerResponse)
    assert response.output_path == fg.parent / f"{fg.stem}_mixed{fg.suffix}"
    assert response.foreground_duration == fg_info.duration
    assert response.background_duration == bg_info.duration
    assert response.output_duration == output_info.duration
    assert response.speech_segments == speech_segments
    assert response.loops_used == 0


# ---------------------------------------------------------------------------
# detect_speech tests
# ---------------------------------------------------------------------------


def test_detect_speech_returns_segments(config, tmp_audio_files, speech_segments):
    """Standalone detection returns list[SpeechSegment] (REQ-021)."""
    from tarash.tarash_audio_mixer.api import detect_speech

    fg, _ = tmp_audio_files

    with patch(
        "tarash.tarash_audio_mixer.api.detector.detect_speech_segments",
        return_value=speech_segments,
    ):
        result = detect_speech(config, fg)

    assert result == speech_segments
    assert all(isinstance(s, SpeechSegment) for s in result)


def test_detect_speech_validates_path(config):
    """Non-existent path raises InvalidInputError."""
    from tarash.tarash_audio_mixer.api import detect_speech

    with pytest.raises(InvalidInputError, match="Foreground file does not exist"):
        detect_speech(config, Path("/nonexistent/audio.wav"))


async def test_detect_speech_async_returns_segments(
    config, tmp_audio_files, speech_segments
):
    """Async detection works (REQ-021)."""
    from tarash.tarash_audio_mixer.api import detect_speech_async

    fg, _ = tmp_audio_files

    with patch(
        "tarash.tarash_audio_mixer.api.detector.detect_speech_segments_async",
        new_callable=AsyncMock,
        return_value=speech_segments,
    ):
        result = await detect_speech_async(config, fg)

    assert result == speech_segments
    assert all(isinstance(s, SpeechSegment) for s in result)
