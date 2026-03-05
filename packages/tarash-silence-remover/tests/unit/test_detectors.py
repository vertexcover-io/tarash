"""Tests for silence detectors."""

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from conftest import make_async_proc
from tarash.tarash_silence_remover.detectors.ffmpeg import (
    FFmpegDetector,
    parse_silencedetect_output,
)
from tarash.tarash_silence_remover.exceptions import FFmpegNotFoundError
from tarash.tarash_silence_remover.models import SilenceRemovalConfig, SpeechSegment


# ==================== FFmpeg Output Parsing ====================


SAMPLE_FFMPEG_OUTPUT = """\
[silencedetect @ 0x55b3a] silence_start: 1.5
[silencedetect @ 0x55b3a] silence_end: 3.2 | silence_duration: 1.7
[silencedetect @ 0x55b3a] silence_start: 5.0
[silencedetect @ 0x55b3a] silence_end: 6.8 | silence_duration: 1.8
"""


def test_parse_silencedetect_output_basic():
    """Parse FFmpeg silencedetect stderr into silence intervals."""
    silences = parse_silencedetect_output(SAMPLE_FFMPEG_OUTPUT)
    assert len(silences) == 2
    assert silences[0] == (1.5, 3.2)
    assert silences[1] == (5.0, 6.8)


def test_parse_silencedetect_output_empty():
    """No silence detected returns empty list."""
    silences = parse_silencedetect_output("")
    assert silences == []


def test_parse_silencedetect_output_unterminated_silence():
    """Silence that starts but doesn't end (e.g., silence at end of file)."""
    output = "[silencedetect @ 0x55b3a] silence_start: 8.0\n"
    silences = parse_silencedetect_output(output)
    # Unterminated silence_start without silence_end should be ignored
    assert silences == []


# ==================== FFmpeg Detector - Invert to Speech ====================


def test_invert_silences_to_speech_segments():
    """Invert silence intervals to get speech segments given total duration."""
    from tarash.tarash_silence_remover.detectors.ffmpeg import invert_silences_to_speech

    # File is 10s, silence from 3-5s and 8-9s
    # Speech should be: 0-3, 5-8, 9-10
    silences = [(3.0, 5.0), (8.0, 9.0)]
    segments = invert_silences_to_speech(silences, total_duration=10.0)
    assert segments == [
        SpeechSegment(start=0.0, end=3.0),
        SpeechSegment(start=5.0, end=8.0),
        SpeechSegment(start=9.0, end=10.0),
    ]


def test_invert_silences_no_silence():
    """No silence means entire file is speech."""
    from tarash.tarash_silence_remover.detectors.ffmpeg import invert_silences_to_speech

    segments = invert_silences_to_speech([], total_duration=5.0)
    assert segments == [SpeechSegment(start=0.0, end=5.0)]


def test_invert_silences_entire_file_silent():
    """Entire file is silence means no speech segments."""
    from tarash.tarash_silence_remover.detectors.ffmpeg import invert_silences_to_speech

    silences = [(0.0, 10.0)]
    segments = invert_silences_to_speech(silences, total_duration=10.0)
    assert segments == []


def test_invert_silences_starts_with_silence():
    """File starts with silence."""
    from tarash.tarash_silence_remover.detectors.ffmpeg import invert_silences_to_speech

    silences = [(0.0, 2.0)]
    segments = invert_silences_to_speech(silences, total_duration=5.0)
    assert segments == [SpeechSegment(start=2.0, end=5.0)]


# ==================== FFmpeg Detector - Integration (mocked subprocess) ====================


@pytest.fixture
def ffmpeg_detector():
    return FFmpegDetector()


@pytest.fixture
def default_config():
    return SilenceRemovalConfig(detector="ffmpeg")


def test_ffmpeg_detector_detect_speech_segments(ffmpeg_detector, default_config):
    """FFmpeg detector calls subprocess and parses output."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stderr = SAMPLE_FFMPEG_OUTPUT

    # Mock ffprobe for duration
    mock_probe = MagicMock()
    mock_probe.returncode = 0
    mock_probe.stdout = "10.0\n"

    with patch("subprocess.run", side_effect=[mock_probe, mock_result]):
        segments = ffmpeg_detector.detect_speech_segments(
            Path("/tmp/test.mp4"), default_config
        )

    # Silence at 1.5-3.2 and 5.0-6.8, total 10s
    # Speech: 0-1.5, 3.2-5.0, 6.8-10.0
    assert len(segments) == 3
    assert segments[0] == SpeechSegment(start=0.0, end=1.5)
    assert segments[1] == SpeechSegment(start=3.2, end=5.0)
    assert segments[2] == SpeechSegment(start=6.8, end=10.0)


def test_ffmpeg_detector_with_preprobed_duration(ffmpeg_detector, default_config):
    """FFmpeg detector skips ffprobe when duration is pre-provided."""
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stderr = SAMPLE_FFMPEG_OUTPUT

    with patch("subprocess.run", return_value=mock_result) as mock_run:
        segments = ffmpeg_detector.detect_speech_segments(
            Path("/tmp/test.mp4"), default_config, duration=10.0
        )

    # Only one subprocess call (silencedetect), no ffprobe call
    assert mock_run.call_count == 1
    assert len(segments) == 3


async def test_ffmpeg_detector_async_with_preprobed_duration(
    ffmpeg_detector, default_config
):
    """FFmpeg detector async skips ffprobe when duration is pre-provided."""
    detect_proc = make_async_proc(returncode=0, stderr=SAMPLE_FFMPEG_OUTPUT.encode())

    with patch(
        "asyncio.create_subprocess_exec",
        return_value=detect_proc,
    ) as mock_exec:
        segments = await ffmpeg_detector.detect_speech_segments_async(
            Path("/tmp/test.mp4"), default_config, duration=10.0
        )

    # Only one subprocess call (silencedetect), no ffprobe call
    assert mock_exec.call_count == 1
    assert len(segments) == 3


def test_ffmpeg_detector_not_found(ffmpeg_detector, default_config):
    """Raises FFmpegNotFoundError when ffmpeg binary not found."""
    with patch("subprocess.run", side_effect=FileNotFoundError("ffmpeg not found")):
        with pytest.raises(FFmpegNotFoundError):
            ffmpeg_detector.detect_speech_segments(
                Path("/tmp/test.mp4"), default_config
            )


# ==================== Silero VAD Detector ====================


def test_silero_detector_falls_back_when_silero_missing():
    """When silero-vad is not available, Silero detector falls back to FFmpeg."""
    from tarash.tarash_silence_remover.detectors.silero import SileroDetector

    detector = SileroDetector()

    with patch(
        "tarash.tarash_silence_remover.detectors.silero._silero_available",
        return_value=False,
    ):
        mock_ffmpeg = MagicMock()
        mock_ffmpeg.detect_speech_segments.return_value = [
            SpeechSegment(start=0.0, end=5.0)
        ]
        with patch(
            "tarash.tarash_silence_remover.detectors.ffmpeg.FFmpegDetector",
            return_value=mock_ffmpeg,
        ):
            config = SilenceRemovalConfig(detector="silero")
            segments = detector.detect_speech_segments(Path("/tmp/test.wav"), config)

    assert segments == [SpeechSegment(start=0.0, end=5.0)]
    mock_ffmpeg.detect_speech_segments.assert_called_once()


# ==================== FFmpeg Detector - Async ====================


async def test_ffmpeg_detector_detect_speech_segments_async(
    ffmpeg_detector, default_config
):
    """FFmpeg detector async calls subprocess and parses output."""
    # First call: ffprobe for duration, second call: ffmpeg silencedetect
    probe_proc = make_async_proc(returncode=0, stdout=b"10.0\n")
    detect_proc = make_async_proc(returncode=0, stderr=SAMPLE_FFMPEG_OUTPUT.encode())

    with patch(
        "asyncio.create_subprocess_exec",
        side_effect=[probe_proc, detect_proc],
    ):
        segments = await ffmpeg_detector.detect_speech_segments_async(
            Path("/tmp/test.mp4"), default_config
        )

    # Silence at 1.5-3.2 and 5.0-6.8, total 10s
    # Speech: 0-1.5, 3.2-5.0, 6.8-10.0
    assert len(segments) == 3
    assert segments[0] == SpeechSegment(start=0.0, end=1.5)
    assert segments[1] == SpeechSegment(start=3.2, end=5.0)
    assert segments[2] == SpeechSegment(start=6.8, end=10.0)


async def test_ffmpeg_detector_async_not_found(ffmpeg_detector, default_config):
    """Raises FFmpegNotFoundError when ffprobe binary not found in async path."""
    with patch(
        "asyncio.create_subprocess_exec",
        side_effect=FileNotFoundError("ffprobe not found"),
    ):
        with pytest.raises(FFmpegNotFoundError):
            await ffmpeg_detector.detect_speech_segments_async(
                Path("/tmp/test.mp4"), default_config
            )


# ==================== Silero Detector - Async ====================


async def test_silero_detector_async_falls_back_when_silero_missing():
    """When silero-vad is unavailable, async path falls back to FFmpeg async detector."""
    from tarash.tarash_silence_remover.detectors.silero import SileroDetector

    detector = SileroDetector()
    expected = [SpeechSegment(start=0.0, end=5.0)]

    with patch(
        "tarash.tarash_silence_remover.detectors.silero._silero_available",
        return_value=False,
    ):
        mock_ffmpeg = MagicMock()
        mock_ffmpeg.detect_speech_segments_async = AsyncMock(return_value=expected)
        with patch(
            "tarash.tarash_silence_remover.detectors.ffmpeg.FFmpegDetector",
            return_value=mock_ffmpeg,
        ):
            config = SilenceRemovalConfig(detector="silero")
            segments = await detector.detect_speech_segments_async(
                Path("/tmp/test.wav"), config
            )

    assert segments == expected
    mock_ffmpeg.detect_speech_segments_async.assert_called_once()


async def test_silero_detector_async_uses_to_thread():
    """When silero-vad is available, async path uses asyncio.to_thread."""
    from tarash.tarash_silence_remover.detectors.silero import SileroDetector

    detector = SileroDetector()
    expected = [SpeechSegment(start=0.0, end=5.0)]

    with patch(
        "tarash.tarash_silence_remover.detectors.silero._silero_available",
        return_value=True,
    ):
        with patch(
            "asyncio.to_thread",
            new_callable=AsyncMock,
            return_value=expected,
        ) as mock_to_thread:
            config = SilenceRemovalConfig(detector="silero")
            segments = await detector.detect_speech_segments_async(
                Path("/tmp/test.wav"), config
            )

    assert segments == expected
    mock_to_thread.assert_called_once_with(
        detector._detect_with_silero, Path("/tmp/test.wav"), config
    )
