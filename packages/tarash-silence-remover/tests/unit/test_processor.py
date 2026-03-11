"""Tests for the processor (segment merging, silence shortening, FFmpeg command building)."""

from pathlib import Path
from unittest.mock import patch

import pytest

from conftest import make_async_proc
from tarash.tarash_silence_remover.exceptions import (
    FFmpegNotFoundError,
    InvalidInputError,
)
from tarash.tarash_silence_remover.models import SpeechSegment
from tarash.tarash_silence_remover.processor import (
    _parse_duration,
    _parse_media_info,
    apply_padding,
    derive_ffprobe_path,
    get_duration_async,
    merge_overlapping_segments,
    probe_media_info,
    probe_media_info_async,
)


# ==================== apply_padding ====================


def test_apply_padding_basic():
    """Padding extends segments on both sides."""
    segments = [SpeechSegment(start=1.0, end=3.0)]
    padded = apply_padding(segments, padding=0.1, total_duration=10.0)
    assert padded == [SpeechSegment(start=0.9, end=3.1)]


def test_apply_padding_clamps_to_zero():
    """Padding doesn't go below 0."""
    segments = [SpeechSegment(start=0.05, end=2.0)]
    padded = apply_padding(segments, padding=0.1, total_duration=10.0)
    assert padded[0].start == 0.0


def test_apply_padding_clamps_to_duration():
    """Padding doesn't exceed total duration."""
    segments = [SpeechSegment(start=9.0, end=9.95)]
    padded = apply_padding(segments, padding=0.1, total_duration=10.0)
    assert padded[0].end == 10.0


# ==================== merge_overlapping_segments ====================


def test_merge_overlapping_basic():
    """Overlapping segments are merged into one."""
    segments = [
        SpeechSegment(start=0.0, end=2.0),
        SpeechSegment(start=1.5, end=4.0),
    ]
    merged = merge_overlapping_segments(segments)
    assert merged == [SpeechSegment(start=0.0, end=4.0)]


def test_merge_adjacent_segments():
    """Adjacent (touching) segments are merged."""
    segments = [
        SpeechSegment(start=0.0, end=2.0),
        SpeechSegment(start=2.0, end=4.0),
    ]
    merged = merge_overlapping_segments(segments)
    assert merged == [SpeechSegment(start=0.0, end=4.0)]


def test_merge_non_overlapping_stays_separate():
    """Non-overlapping segments stay separate."""
    segments = [
        SpeechSegment(start=0.0, end=1.0),
        SpeechSegment(start=3.0, end=5.0),
    ]
    merged = merge_overlapping_segments(segments)
    assert len(merged) == 2


def test_merge_empty():
    """Empty input returns empty output."""
    assert merge_overlapping_segments([]) == []


def test_merge_single():
    """Single segment returned as-is."""
    segments = [SpeechSegment(start=1.0, end=3.0)]
    merged = merge_overlapping_segments(segments)
    assert merged == segments


# ==================== derive_ffprobe_path ====================


def test_derive_ffprobe_path_basic():
    """Derives ffprobe from ffmpeg binary name."""
    assert derive_ffprobe_path("ffmpeg") == "ffprobe"


def test_derive_ffprobe_path_with_directory():
    """Derives ffprobe from full path to ffmpeg."""
    assert derive_ffprobe_path("/usr/local/bin/ffmpeg") == "/usr/local/bin/ffprobe"


def test_derive_ffprobe_path_with_ffmpeg_in_directory():
    """Does not replace 'ffmpeg' in directory components."""
    assert (
        derive_ffprobe_path("/usr/local/ffmpeg-build/ffmpeg")
        == "/usr/local/ffmpeg-build/ffprobe"
    )


def test_derive_ffprobe_path_non_ffmpeg_binary():
    """Returns default 'ffprobe' for non-ffmpeg binary names."""
    assert derive_ffprobe_path("/usr/bin/avconv") == "ffprobe"


# ==================== get_duration_async ====================


async def test_get_duration_async_success():
    """Returns parsed float duration."""
    proc = make_async_proc(returncode=0, stdout=b"12.345\n")
    with patch("asyncio.create_subprocess_exec", return_value=proc):
        result = await get_duration_async("ffmpeg", Path("/tmp/test.mp4"))
    assert result == 12.345


async def test_get_duration_async_ffprobe_not_found():
    """Raises FFmpegNotFoundError when ffprobe binary missing."""
    with patch(
        "asyncio.create_subprocess_exec",
        side_effect=FileNotFoundError("ffprobe not found"),
    ):
        with pytest.raises(FFmpegNotFoundError):
            await get_duration_async("ffmpeg", Path("/tmp/test.mp4"))


async def test_get_duration_async_invalid_input():
    """Raises InvalidInputError on non-zero exit."""
    proc = make_async_proc(returncode=1, stderr=b"No such file")
    with patch("asyncio.create_subprocess_exec", return_value=proc):
        with pytest.raises(InvalidInputError, match="Cannot read file"):
            await get_duration_async("ffmpeg", Path("/tmp/test.mp4"))


# ==================== probe_media_info ====================


def test_probe_media_info_with_video():
    """Returns MediaInfo with video properties for video files."""
    import json
    from unittest.mock import MagicMock

    output = json.dumps(
        {
            "streams": [{"width": 1920, "height": 1080, "r_frame_rate": "30/1"}],
            "format": {"duration": "12.5"},
        }
    )
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = output

    with patch("subprocess.run", return_value=mock_result):
        info = probe_media_info("ffmpeg", Path("/tmp/test.mp4"))

    assert info.duration == 12.5
    assert info.video_width == 1920
    assert info.video_height == 1080
    assert info.video_fps == "30/1"
    assert info.has_video is True


def test_probe_media_info_audio_only():
    """Returns MediaInfo without video properties for audio files."""
    import json
    from unittest.mock import MagicMock

    output = json.dumps(
        {
            "streams": [],
            "format": {"duration": "8.0"},
        }
    )
    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = output

    with patch("subprocess.run", return_value=mock_result):
        info = probe_media_info("ffmpeg", Path("/tmp/test.mp3"))

    assert info.duration == 8.0
    assert info.video_width is None
    assert info.has_video is False


def test_probe_media_info_ffprobe_not_found():
    """Raises FFmpegNotFoundError when ffprobe binary missing."""
    with patch("subprocess.run", side_effect=FileNotFoundError("ffprobe not found")):
        with pytest.raises(FFmpegNotFoundError):
            probe_media_info("ffmpeg", Path("/tmp/test.mp4"))


def test_probe_media_info_invalid_input():
    """Raises InvalidInputError on non-zero exit."""
    from unittest.mock import MagicMock

    mock_result = MagicMock()
    mock_result.returncode = 1
    mock_result.stderr = "No such file"

    with patch("subprocess.run", return_value=mock_result):
        with pytest.raises(InvalidInputError, match="Cannot read file"):
            probe_media_info("ffmpeg", Path("/tmp/test.mp4"))


# ==================== probe_media_info_async ====================


async def test_probe_media_info_async_with_video():
    """Returns MediaInfo with video properties for video files."""
    import json

    output = json.dumps(
        {
            "streams": [{"width": 1920, "height": 1080, "r_frame_rate": "30/1"}],
            "format": {"duration": "12.5"},
        }
    ).encode()
    proc = make_async_proc(returncode=0, stdout=output)
    with patch("asyncio.create_subprocess_exec", return_value=proc):
        info = await probe_media_info_async("ffmpeg", Path("/tmp/test.mp4"))

    assert info.duration == 12.5
    assert info.video_width == 1920
    assert info.has_video is True


async def test_probe_media_info_async_audio_only():
    """Returns MediaInfo without video properties for audio files."""
    import json

    output = json.dumps(
        {
            "streams": [],
            "format": {"duration": "8.0"},
        }
    ).encode()
    proc = make_async_proc(returncode=0, stdout=output)
    with patch("asyncio.create_subprocess_exec", return_value=proc):
        info = await probe_media_info_async("ffmpeg", Path("/tmp/test.mp3"))

    assert info.duration == 8.0
    assert info.has_video is False


async def test_probe_media_info_async_ffprobe_not_found():
    """Raises FFmpegNotFoundError when ffprobe binary missing."""
    with patch(
        "asyncio.create_subprocess_exec",
        side_effect=FileNotFoundError("ffprobe not found"),
    ):
        with pytest.raises(FFmpegNotFoundError):
            await probe_media_info_async("ffmpeg", Path("/tmp/test.mp4"))


async def test_probe_media_info_async_invalid_input():
    """Raises InvalidInputError on non-zero exit."""
    proc = make_async_proc(returncode=1, stderr=b"No such file")
    with patch("asyncio.create_subprocess_exec", return_value=proc):
        with pytest.raises(InvalidInputError, match="Cannot read file"):
            await probe_media_info_async("ffmpeg", Path("/tmp/test.mp4"))


# ==================== _parse_duration ====================


def test_parse_duration_success():
    """Parses float duration from stdout."""
    assert _parse_duration(0, "12.345\n", "", Path("/tmp/test.mp4")) == 12.345


def test_parse_duration_error():
    """Raises InvalidInputError on non-zero returncode."""
    with pytest.raises(InvalidInputError, match="Cannot read file"):
        _parse_duration(1, "", "No such file", Path("/tmp/test.mp4"))


# ==================== _parse_media_info ====================


def test_parse_media_info_with_video():
    """Parses MediaInfo with video stream."""
    import json

    stdout = json.dumps(
        {
            "streams": [{"width": 1920, "height": 1080, "r_frame_rate": "30/1"}],
            "format": {"duration": "12.5"},
        }
    )
    info = _parse_media_info(0, stdout, "", Path("/tmp/test.mp4"))
    assert info.duration == 12.5
    assert info.video_width == 1920
    assert info.has_video is True


def test_parse_media_info_audio_only():
    """Parses MediaInfo without video stream."""
    import json

    stdout = json.dumps(
        {
            "streams": [],
            "format": {"duration": "8.0"},
        }
    )
    info = _parse_media_info(0, stdout, "", Path("/tmp/test.mp3"))
    assert info.duration == 8.0
    assert info.has_video is False


def test_parse_media_info_error():
    """Raises InvalidInputError on non-zero returncode."""
    with pytest.raises(InvalidInputError, match="Cannot read file"):
        _parse_media_info(1, "", "No such file", Path("/tmp/test.mp4"))


# ==================== _notify_progress / _notify_progress_async ====================


def test_notify_progress_fires_callback_with_correct_update():
    """_notify_progress calls the callback with a well-formed ProcessingUpdate."""
    from tarash.tarash_silence_remover.processor import _notify_progress

    received = []
    _notify_progress(received.append, "extracting", 2, 4, "Extracting segment 2/4")

    assert len(received) == 1
    update = received[0]
    assert update.phase == "extracting"
    assert update.current_step == 2
    assert update.total_steps == 4
    assert update.progress_percent == 50
    assert update.message == "Extracting segment 2/4"


def test_notify_progress_none_callback_is_noop():
    """_notify_progress with None callback does nothing."""
    from tarash.tarash_silence_remover.processor import _notify_progress

    # Should not raise
    _notify_progress(None, "extracting", 1, 1, "test")


def test_notify_progress_catches_callback_exception(caplog):
    """_notify_progress catches callback errors and logs a warning."""
    import logging

    from tarash.tarash_silence_remover.processor import _notify_progress

    def bad_callback(update):
        raise RuntimeError("callback broke")

    with caplog.at_level(logging.WARNING):
        _notify_progress(bad_callback, "extracting", 1, 1, "test")

    assert "Progress callback error" in caplog.text


async def test_notify_progress_async_fires_callback():
    """_notify_progress_async calls the async callback correctly."""
    from tarash.tarash_silence_remover.processor import _notify_progress_async

    received = []

    async def cb(update):
        received.append(update)

    await _notify_progress_async(cb, "concatenating", 5, 5, "Concatenating segments")

    assert len(received) == 1
    assert received[0].phase == "concatenating"
    assert received[0].progress_percent == 100


async def test_notify_progress_async_none_callback_is_noop():
    """_notify_progress_async with None callback does nothing."""
    from tarash.tarash_silence_remover.processor import _notify_progress_async

    await _notify_progress_async(None, "extracting", 1, 1, "test")


async def test_notify_progress_async_catches_callback_exception(caplog):
    """_notify_progress_async catches callback errors and logs a warning."""
    import logging

    from tarash.tarash_silence_remover.processor import _notify_progress_async

    async def bad_callback(update):
        raise RuntimeError("async callback broke")

    with caplog.at_level(logging.WARNING):
        await _notify_progress_async(bad_callback, "extracting", 1, 1, "test")

    assert "Progress callback error" in caplog.text


# ==================== SegmentJobs ====================


def test_segment_jobs_total_commands():
    """SegmentJobs.total_commands returns extract + silence + 1 (concat)."""
    from tarash.tarash_silence_remover.processor import SegmentJobs

    jobs = SegmentJobs(
        extract_cmds=[["cmd1"], ["cmd2"], ["cmd3"]],
        silence_cmds=[["s1"], ["s2"]],
        concat_text="file 'a'\nfile 'b'",
    )
    assert jobs.total_commands == 6  # 3 + 2 + 1


def test_segment_jobs_no_silence_commands():
    """SegmentJobs with no silence commands: total = extract + 1."""
    from tarash.tarash_silence_remover.processor import SegmentJobs

    jobs = SegmentJobs(
        extract_cmds=[["cmd1"]],
        silence_cmds=[],
        concat_text="file 'a'",
    )
    assert jobs.total_commands == 2  # 1 + 0 + 1


# ==================== process_segments with on_progress ====================


def test_process_segments_fires_progress_callbacks_in_order():
    """process_segments fires callbacks in order: extracting -> generating_silence -> concatenating."""
    from unittest.mock import MagicMock

    from tarash.tarash_silence_remover.models import MediaInfo, SilenceRemovalConfig
    from tarash.tarash_silence_remover.processor import process_segments

    config = SilenceRemovalConfig(
        detector="ffmpeg",
        min_silence_duration=0.5,
        target_silence_duration=0.3,
    )
    segments = [
        SpeechSegment(start=0.0, end=2.0),
        SpeechSegment(start=4.0, end=6.0),
    ]
    media_info = MediaInfo(duration=10.0)

    received = []

    def on_progress(update):
        received.append(update)

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        process_segments(
            Path("/tmp/input.mp4"),
            Path("/tmp/output.mp4"),
            segments,
            config,
            media_info=media_info,
            on_progress=on_progress,
        )

    # Should have: 2 extract + 1 silence + 1 concat = 4 updates
    assert len(received) == 4
    assert received[0].phase == "extracting"
    assert received[1].phase == "extracting"
    assert received[2].phase == "generating_silence"
    assert received[3].phase == "concatenating"

    # progress_percent should be monotonically increasing
    percents = [u.progress_percent for u in received]
    assert percents == sorted(percents)

    # current_step should increment and total_steps should be consistent
    steps = [u.current_step for u in received]
    assert steps == [1, 2, 3, 4]
    assert all(u.total_steps == 4 for u in received)


def test_process_segments_no_callback_works():
    """process_segments works with no callback (backward compatible)."""
    from unittest.mock import MagicMock

    from tarash.tarash_silence_remover.models import MediaInfo, SilenceRemovalConfig
    from tarash.tarash_silence_remover.processor import process_segments

    config = SilenceRemovalConfig(detector="ffmpeg")
    segments = [SpeechSegment(start=0.0, end=2.0)]
    media_info = MediaInfo(duration=10.0)

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        # Should not raise — on_progress defaults to None
        process_segments(
            Path("/tmp/input.mp4"),
            Path("/tmp/output.mp4"),
            segments,
            config,
            media_info=media_info,
        )


def test_process_segments_callback_error_does_not_crash():
    """process_segments continues even when callback raises."""
    from unittest.mock import MagicMock

    from tarash.tarash_silence_remover.models import MediaInfo, SilenceRemovalConfig
    from tarash.tarash_silence_remover.processor import process_segments

    config = SilenceRemovalConfig(detector="ffmpeg")
    segments = [SpeechSegment(start=0.0, end=2.0)]
    media_info = MediaInfo(duration=10.0)

    def bad_callback(update):
        raise RuntimeError("callback error")

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        # Should not raise despite callback error
        process_segments(
            Path("/tmp/input.mp4"),
            Path("/tmp/output.mp4"),
            segments,
            config,
            media_info=media_info,
            on_progress=bad_callback,
        )


def test_process_segments_single_segment_no_silence():
    """Single segment produces only extracting + concatenating updates."""
    from unittest.mock import MagicMock

    from tarash.tarash_silence_remover.models import MediaInfo, SilenceRemovalConfig
    from tarash.tarash_silence_remover.processor import process_segments

    config = SilenceRemovalConfig(detector="ffmpeg")
    segments = [SpeechSegment(start=0.0, end=5.0)]
    media_info = MediaInfo(duration=10.0)

    received = []

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        process_segments(
            Path("/tmp/input.mp4"),
            Path("/tmp/output.mp4"),
            segments,
            config,
            media_info=media_info,
            on_progress=received.append,
        )

    assert len(received) == 2  # 1 extract + 1 concat
    assert received[0].phase == "extracting"
    assert received[1].phase == "concatenating"


def test_process_segments_message_contains_segment_indices():
    """Progress messages include segment indices like 'Extracting segment 1/2'."""
    from unittest.mock import MagicMock

    from tarash.tarash_silence_remover.models import MediaInfo, SilenceRemovalConfig
    from tarash.tarash_silence_remover.processor import process_segments

    config = SilenceRemovalConfig(
        detector="ffmpeg",
        min_silence_duration=0.5,
        target_silence_duration=0.3,
    )
    segments = [
        SpeechSegment(start=0.0, end=2.0),
        SpeechSegment(start=4.0, end=6.0),
    ]
    media_info = MediaInfo(duration=10.0)

    received = []

    mock_result = MagicMock()
    mock_result.returncode = 0
    mock_result.stdout = ""
    mock_result.stderr = ""

    with patch("subprocess.run", return_value=mock_result):
        process_segments(
            Path("/tmp/input.mp4"),
            Path("/tmp/output.mp4"),
            segments,
            config,
            media_info=media_info,
            on_progress=received.append,
        )

    assert "1/2" in received[0].message
    assert "2/2" in received[1].message
    assert "1/1" in received[2].message
    assert "Concatenating" in received[3].message


async def test_process_segments_async_fires_callbacks_in_order():
    """Async variant fires callbacks in correct order."""
    from tarash.tarash_silence_remover.models import MediaInfo, SilenceRemovalConfig
    from tarash.tarash_silence_remover.processor import process_segments_async

    config = SilenceRemovalConfig(
        detector="ffmpeg",
        min_silence_duration=0.5,
        target_silence_duration=0.3,
    )
    segments = [
        SpeechSegment(start=0.0, end=2.0),
        SpeechSegment(start=4.0, end=6.0),
    ]
    media_info = MediaInfo(duration=10.0)

    received = []

    async def on_progress(update):
        received.append(update)

    proc = make_async_proc(returncode=0, stdout=b"", stderr=b"")
    with patch("asyncio.create_subprocess_exec", return_value=proc):
        await process_segments_async(
            Path("/tmp/input.mp4"),
            Path("/tmp/output.mp4"),
            segments,
            config,
            media_info=media_info,
            on_progress=on_progress,
        )

    assert len(received) == 4
    assert received[0].phase == "extracting"
    assert received[1].phase == "extracting"
    assert received[2].phase == "generating_silence"
    assert received[3].phase == "concatenating"
