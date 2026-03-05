"""Tests for the public API."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tarash.tarash_silence_remover.api import (
    _get_detector,
    _resolve_output_path,
    detect_silence,
    detect_silence_async,
    remove_silence,
    remove_silence_async,
)
from tarash.tarash_silence_remover.exceptions import InvalidInputError
from tarash.tarash_silence_remover.models import (
    MediaInfo,
    SilenceRemovalConfig,
    SilenceRemovalRequest,
    SilenceRemovalResponse,
    SpeechSegment,
)


# ==================== _resolve_output_path ====================


def test_resolve_output_path_default():
    """Default output adds _cleaned suffix."""
    result = _resolve_output_path(
        input_path=Path("/tmp/video.mp4"),
        output_path=None,
    )
    assert result == Path("/tmp/video_cleaned.mp4")


def test_resolve_output_path_explicit():
    """Explicit output path is returned as-is."""
    result = _resolve_output_path(
        input_path=Path("/tmp/video.mp4"),
        output_path=Path("/tmp/out.mp4"),
    )
    assert result == Path("/tmp/out.mp4")


# ==================== _get_detector ====================


def test_get_detector_ffmpeg():
    """FFmpeg config returns FFmpegDetector."""
    from tarash.tarash_silence_remover.detectors.ffmpeg import FFmpegDetector

    config = SilenceRemovalConfig(detector="ffmpeg")
    detector = _get_detector(config)
    assert isinstance(detector, FFmpegDetector)


def test_get_detector_silero():
    """Silero config returns SileroDetector."""
    from tarash.tarash_silence_remover.detectors.silero import SileroDetector

    config = SilenceRemovalConfig(detector="silero")
    detector = _get_detector(config)
    assert isinstance(detector, SileroDetector)


# ==================== detect_silence ====================


def test_detect_silence_delegates_to_detector():
    """detect_silence calls the detector and returns segments."""
    config = SilenceRemovalConfig(detector="ffmpeg")
    expected = [SpeechSegment(start=0.0, end=5.0)]

    with patch("tarash.tarash_silence_remover.api._get_detector") as mock_get:
        mock_detector = MagicMock()
        mock_detector.detect_speech_segments.return_value = expected
        mock_get.return_value = mock_detector

        result = detect_silence(config, Path("/tmp/test.wav"))

    assert result == expected


# ==================== remove_silence ====================


def test_remove_silence_invalid_input():
    """Raises InvalidInputError when input file doesn't exist."""
    config = SilenceRemovalConfig(detector="ffmpeg")
    request = SilenceRemovalRequest(input_path=Path("/nonexistent/file.mp4"))

    with pytest.raises(InvalidInputError, match="does not exist"):
        remove_silence(config, request)


def test_remove_silence_full_pipeline(tmp_path):
    """Full pipeline: detect -> merge -> process -> response."""
    input_file = tmp_path / "input.mp4"
    input_file.touch()  # Create empty file (detection is mocked)

    config = SilenceRemovalConfig(detector="ffmpeg")
    request = SilenceRemovalRequest(input_path=input_file)

    mock_segments = [
        SpeechSegment(start=0.0, end=3.0),
        SpeechSegment(start=5.0, end=10.0),
    ]

    mock_media_info = MediaInfo(duration=10.0)

    with (
        patch("tarash.tarash_silence_remover.api._get_detector") as mock_get_det,
        patch(
            "tarash.tarash_silence_remover.api.probe_media_info",
            return_value=mock_media_info,
        ),
        patch(
            "tarash.tarash_silence_remover.api.get_duration",
            return_value=7.8,
        ),
        patch("tarash.tarash_silence_remover.api.process_segments") as mock_process,
    ):
        mock_detector = MagicMock()
        mock_detector.detect_speech_segments.return_value = mock_segments
        mock_get_det.return_value = mock_detector

        response = remove_silence(config, request)

    assert isinstance(response, SilenceRemovalResponse)
    assert response.output_path == tmp_path / "input_cleaned.mp4"
    assert response.detector_used == "ffmpeg"
    assert len(response.segments_kept) == 2
    assert response.original_duration == 10.0
    assert response.output_duration == 7.8
    mock_process.assert_called_once()
    # Verify media_info was passed through to process_segments
    _, kwargs = mock_process.call_args
    assert kwargs.get("media_info") == mock_media_info


# ==================== detect_silence_async ====================


async def test_detect_silence_async_delegates_to_detector():
    """detect_silence_async calls the async detector method and returns segments."""
    config = SilenceRemovalConfig(detector="ffmpeg")
    expected = [SpeechSegment(start=0.0, end=5.0)]

    with patch("tarash.tarash_silence_remover.api._get_detector") as mock_get:
        mock_detector = MagicMock()
        mock_detector.detect_speech_segments_async = AsyncMock(return_value=expected)
        mock_get.return_value = mock_detector

        result = await detect_silence_async(config, Path("/tmp/test.wav"))

    assert result == expected
    mock_detector.detect_speech_segments_async.assert_called_once()


# ==================== remove_silence_async ====================


async def test_remove_silence_async_invalid_input():
    """Raises InvalidInputError when input file doesn't exist."""
    config = SilenceRemovalConfig(detector="ffmpeg")
    request = SilenceRemovalRequest(input_path=Path("/nonexistent/file.mp4"))

    with pytest.raises(InvalidInputError, match="does not exist"):
        await remove_silence_async(config, request)


async def test_remove_silence_async_full_pipeline(tmp_path):
    """Full async pipeline: detect -> merge -> process -> response."""
    input_file = tmp_path / "input.mp4"
    input_file.touch()

    config = SilenceRemovalConfig(detector="ffmpeg")
    request = SilenceRemovalRequest(input_path=input_file)

    mock_segments = [
        SpeechSegment(start=0.0, end=3.0),
        SpeechSegment(start=5.0, end=10.0),
    ]

    mock_media_info = MediaInfo(duration=10.0)

    with (
        patch("tarash.tarash_silence_remover.api._get_detector") as mock_get_det,
        patch(
            "tarash.tarash_silence_remover.api.probe_media_info_async",
            new_callable=AsyncMock,
            return_value=mock_media_info,
        ),
        patch(
            "tarash.tarash_silence_remover.api.get_duration_async",
            new_callable=AsyncMock,
            return_value=7.8,
        ),
        patch(
            "tarash.tarash_silence_remover.api.process_segments_async",
            new_callable=AsyncMock,
        ) as mock_process,
    ):
        mock_detector = MagicMock()
        mock_detector.detect_speech_segments_async = AsyncMock(
            return_value=mock_segments
        )
        mock_get_det.return_value = mock_detector

        response = await remove_silence_async(config, request)

    assert isinstance(response, SilenceRemovalResponse)
    assert response.output_path == tmp_path / "input_cleaned.mp4"
    assert response.detector_used == "ffmpeg"
    assert len(response.segments_kept) == 2
    assert response.original_duration == 10.0
    assert response.output_duration == 7.8
    mock_process.assert_called_once()
    # Verify media_info was passed through to process_segments_async
    _, kwargs = mock_process.call_args
    assert kwargs.get("media_info") == mock_media_info
