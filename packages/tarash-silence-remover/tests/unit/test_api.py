"""Tests for the public API."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from tarash.tarash_silence_remover.api import (
    _estimate_output_duration,
    _get_detector,
    _resolve_output_path,
    detect_silence,
    detect_silence_async,
    preview_silence,
    preview_silence_async,
    remove_silence,
    remove_silence_async,
)
from tarash.tarash_silence_remover.exceptions import InvalidInputError
from tarash.tarash_silence_remover.models import (
    MediaInfo,
    SilenceRemovalConfig,
    SilenceRemovalPreview,
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


# ==================== remove_silence with on_progress ====================


def test_remove_silence_fires_probing_and_detecting_callbacks(tmp_path):
    """remove_silence fires probing and detecting updates before those steps."""
    input_file = tmp_path / "input.mp4"
    input_file.touch()

    config = SilenceRemovalConfig(detector="ffmpeg")
    request = SilenceRemovalRequest(input_path=input_file)

    mock_segments = [SpeechSegment(start=0.0, end=3.0)]
    mock_media_info = MediaInfo(duration=10.0)

    received = []

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
        patch("tarash.tarash_silence_remover.api.process_segments"),
    ):
        mock_detector = MagicMock()
        mock_detector.detect_speech_segments.return_value = mock_segments
        mock_get_det.return_value = mock_detector

        remove_silence(config, request, on_progress=received.append)

    # Should have at least probing and detecting updates
    phases = [u.phase for u in received]
    assert "probing" in phases
    assert "detecting" in phases
    # Probing should come before detecting
    assert phases.index("probing") < phases.index("detecting")


def test_remove_silence_passes_callback_to_process_segments(tmp_path):
    """remove_silence passes on_progress through to process_segments."""
    input_file = tmp_path / "input.mp4"
    input_file.touch()

    config = SilenceRemovalConfig(detector="ffmpeg")
    request = SilenceRemovalRequest(input_path=input_file)

    mock_segments = [SpeechSegment(start=0.0, end=3.0)]
    mock_media_info = MediaInfo(duration=10.0)

    def my_callback(update):
        pass

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

        remove_silence(config, request, on_progress=my_callback)

    _, kwargs = mock_process.call_args
    assert kwargs.get("on_progress") is my_callback


def test_remove_silence_no_callback_backward_compatible(tmp_path):
    """remove_silence works with no callback (default None)."""
    input_file = tmp_path / "input.mp4"
    input_file.touch()

    config = SilenceRemovalConfig(detector="ffmpeg")
    request = SilenceRemovalRequest(input_path=input_file)

    mock_segments = [SpeechSegment(start=0.0, end=3.0)]
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
    _, kwargs = mock_process.call_args
    assert kwargs.get("on_progress") is None


def test_remove_silence_callback_error_does_not_crash(tmp_path):
    """remove_silence continues when callback raises during probing."""
    input_file = tmp_path / "input.mp4"
    input_file.touch()

    config = SilenceRemovalConfig(detector="ffmpeg")
    request = SilenceRemovalRequest(input_path=input_file)

    mock_segments = [SpeechSegment(start=0.0, end=3.0)]
    mock_media_info = MediaInfo(duration=10.0)

    def bad_callback(update):
        raise RuntimeError("callback broke")

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
        patch("tarash.tarash_silence_remover.api.process_segments"),
    ):
        mock_detector = MagicMock()
        mock_detector.detect_speech_segments.return_value = mock_segments
        mock_get_det.return_value = mock_detector

        # Should not raise
        response = remove_silence(config, request, on_progress=bad_callback)

    assert isinstance(response, SilenceRemovalResponse)


# ==================== remove_silence_async with on_progress ====================


async def test_remove_silence_async_fires_probing_and_detecting(tmp_path):
    """remove_silence_async fires probing and detecting updates."""
    input_file = tmp_path / "input.mp4"
    input_file.touch()

    config = SilenceRemovalConfig(detector="ffmpeg")
    request = SilenceRemovalRequest(input_path=input_file)

    mock_segments = [SpeechSegment(start=0.0, end=3.0)]
    mock_media_info = MediaInfo(duration=10.0)

    received = []

    async def on_progress(update):
        received.append(update)

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
        ),
    ):
        mock_detector = MagicMock()
        mock_detector.detect_speech_segments_async = AsyncMock(
            return_value=mock_segments
        )
        mock_get_det.return_value = mock_detector

        await remove_silence_async(config, request, on_progress=on_progress)

    phases = [u.phase for u in received]
    assert "probing" in phases
    assert "detecting" in phases


async def test_remove_silence_async_passes_callback_through(tmp_path):
    """remove_silence_async passes on_progress to process_segments_async."""
    input_file = tmp_path / "input.mp4"
    input_file.touch()

    config = SilenceRemovalConfig(detector="ffmpeg")
    request = SilenceRemovalRequest(input_path=input_file)

    mock_segments = [SpeechSegment(start=0.0, end=3.0)]
    mock_media_info = MediaInfo(duration=10.0)

    async def my_callback(update):
        pass

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

        await remove_silence_async(config, request, on_progress=my_callback)

    _, kwargs = mock_process.call_args
    assert kwargs.get("on_progress") is my_callback


# ==================== _estimate_output_duration ====================


def test_estimate_output_duration_multiple_segments():
    """Estimates speech + silence gaps correctly with multiple segments."""
    merged = [
        SpeechSegment(start=0.0, end=3.0),
        SpeechSegment(start=5.0, end=8.0),
    ]
    config = SilenceRemovalConfig(
        min_silence_duration=0.5,
        target_silence_duration=0.3,
    )
    duration, gaps = _estimate_output_duration(merged, config)
    # speech = 3.0 + 3.0 = 6.0, gap between segments = 2.0 > 0.5, so 1 gap
    # estimated = 6.0 + 1 * 0.3 = 6.3
    assert duration == pytest.approx(6.3)
    assert gaps == 1


def test_estimate_output_duration_no_segments():
    """Empty segment list returns zero duration and zero gaps."""
    config = SilenceRemovalConfig()
    duration, gaps = _estimate_output_duration([], config)
    assert duration == 0.0
    assert gaps == 0


def test_estimate_output_duration_single_segment():
    """Single segment has no gaps."""
    merged = [SpeechSegment(start=1.0, end=5.0)]
    config = SilenceRemovalConfig()
    duration, gaps = _estimate_output_duration(merged, config)
    assert duration == pytest.approx(4.0)
    assert gaps == 0


def test_estimate_output_duration_gap_below_min_silence():
    """Gaps smaller than min_silence_duration are not counted."""
    merged = [
        SpeechSegment(start=0.0, end=3.0),
        SpeechSegment(start=3.2, end=6.0),  # gap = 0.2, below default 0.5
    ]
    config = SilenceRemovalConfig(min_silence_duration=0.5)
    duration, gaps = _estimate_output_duration(merged, config)
    # speech = 3.0 + 2.8 = 5.8, no qualifying gaps (0.2 < 0.5)
    assert duration == pytest.approx(5.8)
    assert gaps == 0


def test_estimate_output_duration_zero_target_silence():
    """target_silence_duration=0 means gaps are not inserted."""
    merged = [
        SpeechSegment(start=0.0, end=3.0),
        SpeechSegment(start=5.0, end=8.0),
    ]
    config = SilenceRemovalConfig(
        min_silence_duration=0.5,
        target_silence_duration=0.0,
    )
    duration, gaps = _estimate_output_duration(merged, config)
    # speech = 6.0, target_silence_duration=0 so no gap insertion
    assert duration == pytest.approx(6.0)
    assert gaps == 0


# ==================== preview_silence ====================


def test_preview_silence_invalid_input():
    """Raises InvalidInputError when input file doesn't exist."""
    config = SilenceRemovalConfig(detector="ffmpeg")
    with pytest.raises(InvalidInputError, match="does not exist"):
        preview_silence(config, Path("/nonexistent/file.mp4"))


def test_preview_silence_full_pipeline(tmp_path):
    """Full pipeline: probe -> detect -> pad -> merge -> estimate -> preview."""
    input_file = tmp_path / "input.mp4"
    input_file.touch()

    config = SilenceRemovalConfig(
        detector="ffmpeg",
        min_silence_duration=0.5,
        target_silence_duration=0.3,
        padding=0.1,
    )

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
    ):
        mock_detector = MagicMock()
        mock_detector.detect_speech_segments.return_value = mock_segments
        mock_get_det.return_value = mock_detector

        result = preview_silence(config, input_file)

    assert isinstance(result, SilenceRemovalPreview)
    assert result.original_duration == 10.0
    assert result.detector_used == "ffmpeg"
    assert len(result.segments_to_keep) >= 1
    assert result.estimated_output_duration > 0
    assert result.silence_gaps_to_insert >= 0


def test_preview_silence_does_not_call_ffmpeg_processing(tmp_path):
    """preview_silence must not call process_segments or get_duration."""
    input_file = tmp_path / "input.mp4"
    input_file.touch()

    config = SilenceRemovalConfig(detector="ffmpeg")
    mock_segments = [SpeechSegment(start=0.0, end=5.0)]
    mock_media_info = MediaInfo(duration=10.0)

    with (
        patch("tarash.tarash_silence_remover.api._get_detector") as mock_get_det,
        patch(
            "tarash.tarash_silence_remover.api.probe_media_info",
            return_value=mock_media_info,
        ),
        patch("tarash.tarash_silence_remover.api.process_segments") as mock_process,
        patch("tarash.tarash_silence_remover.api.get_duration") as mock_get_dur,
    ):
        mock_detector = MagicMock()
        mock_detector.detect_speech_segments.return_value = mock_segments
        mock_get_det.return_value = mock_detector

        preview_silence(config, input_file)

    mock_process.assert_not_called()
    mock_get_dur.assert_not_called()


def test_preview_silence_all_speech_no_silence(tmp_path):
    """All-speech input returns preview with duration equal to total speech."""
    input_file = tmp_path / "input.mp4"
    input_file.touch()

    config = SilenceRemovalConfig(detector="ffmpeg")
    mock_media_info = MediaInfo(duration=10.0)
    # Single segment covering the entire file -- no silence at all
    mock_segments = [SpeechSegment(start=0.0, end=10.0)]

    with (
        patch("tarash.tarash_silence_remover.api._get_detector") as mock_get_det,
        patch(
            "tarash.tarash_silence_remover.api.probe_media_info",
            return_value=mock_media_info,
        ),
    ):
        mock_detector = MagicMock()
        mock_detector.detect_speech_segments.return_value = mock_segments
        mock_get_det.return_value = mock_detector

        result = preview_silence(config, input_file)

    assert isinstance(result, SilenceRemovalPreview)
    assert result.estimated_output_duration == pytest.approx(10.0)
    assert len(result.segments_to_keep) == 1
    assert result.silence_gaps_to_insert == 0
    assert result.reduction_percent == pytest.approx(0.0)


def test_preview_silence_no_speech_segments(tmp_path):
    """All-silence input returns zero estimated output duration."""
    input_file = tmp_path / "input.mp4"
    input_file.touch()

    config = SilenceRemovalConfig(detector="ffmpeg")
    mock_media_info = MediaInfo(duration=10.0)

    with (
        patch("tarash.tarash_silence_remover.api._get_detector") as mock_get_det,
        patch(
            "tarash.tarash_silence_remover.api.probe_media_info",
            return_value=mock_media_info,
        ),
    ):
        mock_detector = MagicMock()
        mock_detector.detect_speech_segments.return_value = []
        mock_get_det.return_value = mock_detector

        result = preview_silence(config, input_file)

    assert result.estimated_output_duration == 0.0
    assert result.segments_to_keep == []
    assert result.silence_gaps_to_insert == 0
    assert result.reduction_percent == 100.0


# ==================== preview_silence_async ====================


async def test_preview_silence_async_invalid_input():
    """Raises InvalidInputError when input file doesn't exist."""
    config = SilenceRemovalConfig(detector="ffmpeg")
    with pytest.raises(InvalidInputError, match="does not exist"):
        await preview_silence_async(config, Path("/nonexistent/file.mp4"))


async def test_preview_silence_async_full_pipeline(tmp_path):
    """Full async pipeline: probe -> detect -> pad -> merge -> estimate -> preview."""
    input_file = tmp_path / "input.mp4"
    input_file.touch()

    config = SilenceRemovalConfig(
        detector="ffmpeg",
        min_silence_duration=0.5,
        target_silence_duration=0.3,
        padding=0.1,
    )

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
    ):
        mock_detector = MagicMock()
        mock_detector.detect_speech_segments_async = AsyncMock(
            return_value=mock_segments
        )
        mock_get_det.return_value = mock_detector

        result = await preview_silence_async(config, input_file)

    assert isinstance(result, SilenceRemovalPreview)
    assert result.original_duration == 10.0
    assert result.detector_used == "ffmpeg"
    assert len(result.segments_to_keep) >= 1
    assert result.estimated_output_duration > 0


async def test_preview_silence_async_does_not_call_ffmpeg_processing(tmp_path):
    """preview_silence_async must not call process_segments_async or get_duration_async."""
    input_file = tmp_path / "input.mp4"
    input_file.touch()

    config = SilenceRemovalConfig(detector="ffmpeg")
    mock_segments = [SpeechSegment(start=0.0, end=5.0)]
    mock_media_info = MediaInfo(duration=10.0)

    with (
        patch("tarash.tarash_silence_remover.api._get_detector") as mock_get_det,
        patch(
            "tarash.tarash_silence_remover.api.probe_media_info_async",
            new_callable=AsyncMock,
            return_value=mock_media_info,
        ),
        patch(
            "tarash.tarash_silence_remover.api.process_segments_async",
            new_callable=AsyncMock,
        ) as mock_process,
        patch(
            "tarash.tarash_silence_remover.api.get_duration_async",
            new_callable=AsyncMock,
        ) as mock_get_dur,
    ):
        mock_detector = MagicMock()
        mock_detector.detect_speech_segments_async = AsyncMock(
            return_value=mock_segments
        )
        mock_get_det.return_value = mock_detector

        await preview_silence_async(config, input_file)

    mock_process.assert_not_called()
    mock_get_dur.assert_not_called()


# ==================== preview functions exported ====================


def test_preview_functions_exported_from_package():
    """preview_silence and preview_silence_async are importable from package."""
    from tarash.tarash_silence_remover import (
        preview_silence as ps,
        preview_silence_async as psa,
    )

    assert ps is not None
    assert psa is not None


def test_preview_functions_in_package_all():
    """preview_silence and preview_silence_async are listed in __all__."""
    import tarash.tarash_silence_remover as pkg

    assert "preview_silence" in pkg.__all__
    assert "preview_silence_async" in pkg.__all__
