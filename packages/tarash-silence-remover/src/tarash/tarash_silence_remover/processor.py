"""FFmpeg-based processor for cutting and concatenating media segments."""

import asyncio
import json
import logging
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from tarash.tarash_silence_remover.exceptions import (
    FFmpegNotFoundError,
    InvalidInputError,
    ProcessingError,
)
from tarash.tarash_silence_remover.logging import log_info
from tarash.tarash_silence_remover.models import (
    AsyncProgressCallback,
    MediaInfo,
    ProcessingPhase,
    ProcessingUpdate,
    SilenceRemovalConfig,
    SpeechSegment,
    SyncProgressCallback,
)

_LOGGER_NAME = "tarash.tarash_silence_remover.processor"
_logger = logging.getLogger(_LOGGER_NAME)


@dataclass(frozen=True)
class SegmentJobs:
    """Structured result from _prepare_segment_jobs."""

    extract_cmds: list[list[str]]
    silence_cmds: list[list[str]]
    concat_text: str

    @property
    def total_commands(self) -> int:
        """Total FFmpeg commands including concat."""
        return len(self.extract_cmds) + len(self.silence_cmds) + 1  # +1 for concat


def _notify_progress(
    on_progress: SyncProgressCallback | None,
    phase: ProcessingPhase,
    current_step: int,
    total_steps: int,
    message: str,
    *,
    progress_percent: int | None = None,
) -> None:
    """Fire sync progress callback, catching and logging errors."""
    if on_progress is None:
        return
    try:
        pct = (
            progress_percent
            if progress_percent is not None
            else int((current_step / total_steps) * 100)
        )
        update = ProcessingUpdate(
            phase=phase,
            progress_percent=pct,
            current_step=current_step,
            total_steps=total_steps,
            message=message,
        )
        on_progress(update)
    except Exception:
        _logger.warning("Progress callback error", exc_info=True)


async def _notify_progress_async(
    on_progress: AsyncProgressCallback | None,
    phase: ProcessingPhase,
    current_step: int,
    total_steps: int,
    message: str,
    *,
    progress_percent: int | None = None,
) -> None:
    """Fire async progress callback, catching and logging errors."""
    if on_progress is None:
        return
    try:
        pct = (
            progress_percent
            if progress_percent is not None
            else int((current_step / total_steps) * 100)
        )
        update = ProcessingUpdate(
            phase=phase,
            progress_percent=pct,
            current_step=current_step,
            total_steps=total_steps,
            message=message,
        )
        await on_progress(update)
    except Exception:
        _logger.warning("Progress callback error", exc_info=True)


def derive_ffprobe_path(ffmpeg_path: str) -> str:
    """Derive the ffprobe binary path from the ffmpeg binary path."""
    p = Path(ffmpeg_path)
    if p.name == "ffmpeg":
        return str(p.with_name("ffprobe"))
    return "ffprobe"


def apply_padding(
    segments: list[SpeechSegment],
    padding: float,
    total_duration: float,
) -> list[SpeechSegment]:
    """Extend each segment by padding on both sides, clamped to file bounds.

    Args:
        segments: Speech segments to pad.
        padding: Seconds to add before/after each segment.
        total_duration: Total file duration (upper clamp).

    Returns:
        New list of padded SpeechSegment.
    """
    return [
        SpeechSegment(
            start=max(0.0, seg.start - padding),
            end=min(total_duration, seg.end + padding),
        )
        for seg in segments
    ]


def merge_overlapping_segments(
    segments: list[SpeechSegment],
) -> list[SpeechSegment]:
    """Merge overlapping or adjacent segments.

    Args:
        segments: Potentially overlapping segments (must be sorted by start).

    Returns:
        Merged list with no overlaps.
    """
    if not segments:
        return []

    sorted_segs = sorted(segments, key=lambda s: s.start)
    merged: list[SpeechSegment] = [sorted_segs[0]]

    for seg in sorted_segs[1:]:
        last = merged[-1]
        if seg.start <= last.end:
            # Overlapping or adjacent — extend
            merged[-1] = SpeechSegment(start=last.start, end=max(last.end, seg.end))
        else:
            merged.append(seg)

    return merged


# ---------------------------------------------------------------------------
# Command builders (shared between sync/async)
# ---------------------------------------------------------------------------


def _ffprobe_duration_cmd(ffprobe_path: str, file_path: Path) -> list[str]:
    """Build ffprobe command for getting duration."""
    return [
        ffprobe_path,
        "-v",
        "error",
        "-show_entries",
        "format=duration",
        "-of",
        "default=noprint_wrappers=1:nokey=1",
        str(file_path),
    ]


def _parse_duration(
    returncode: int, stdout: str, stderr: str, file_path: Path
) -> float:
    """Parse ffprobe duration output, raising on errors."""
    if returncode != 0:
        raise InvalidInputError(f"Cannot read file '{file_path}': {stderr.strip()}")
    return float(stdout.strip())


def _ffprobe_info_cmd(ffprobe_path: str, file_path: Path) -> list[str]:
    """Build ffprobe command for media info."""
    return [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "format=duration",
        "-show_entries",
        "stream=width,height,r_frame_rate",
        "-of",
        "json",
        str(file_path),
    ]


def _parse_media_info(
    returncode: int, stdout: str, stderr: str, file_path: Path
) -> MediaInfo:
    """Parse ffprobe JSON output into MediaInfo, raising on errors."""
    if returncode != 0:
        raise InvalidInputError(f"Cannot read file '{file_path}': {stderr.strip()}")

    data = json.loads(stdout)
    duration = float(data["format"]["duration"])
    streams = data.get("streams", [])

    if streams:
        s = streams[0]
        return MediaInfo(
            duration=duration,
            video_width=int(s["width"]),
            video_height=int(s["height"]),
            video_fps=s.get("r_frame_rate", "30/1"),
        )

    return MediaInfo(duration=duration)


def _segment_extract_cmd(
    ffmpeg_path: str,
    input_path: Path,
    seg: SpeechSegment,
    part_path: Path,
    has_video: bool,
) -> list[str]:
    """Build FFmpeg command to extract one segment."""
    duration = seg.end - seg.start
    cmd = [
        ffmpeg_path,
        "-y",
        "-ss",
        str(seg.start),
        "-i",
        str(input_path),
        "-t",
        str(duration),
    ]
    if not has_video:
        cmd += ["-c", "copy"]
    else:
        cmd += ["-c:v", "libx264", "-crf", "0", "-c:a", "aac"]
    cmd += ["-avoid_negative_ts", "make_zero", str(part_path)]
    return cmd


def _silence_cmd(
    ffmpeg_path: str,
    output_path: Path,
    duration: float,
    media_info: MediaInfo | None,
) -> list[str]:
    """Build FFmpeg command to generate silence."""
    if media_info is not None and media_info.has_video:
        res = f"{media_info.video_width}x{media_info.video_height}"
        fps = media_info.video_fps
        return [
            ffmpeg_path,
            "-y",
            "-f",
            "lavfi",
            "-i",
            f"anullsrc=r=44100:cl=stereo:d={duration}",
            "-f",
            "lavfi",
            "-i",
            f"color=c=black:s={res}:r={fps}:d={duration}",
            "-shortest",
            "-c:v",
            "libx264",
            "-crf",
            "0",
            "-c:a",
            "aac",
            str(output_path),
        ]
    return [
        ffmpeg_path,
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"anullsrc=r=44100:cl=stereo:d={duration}",
        "-t",
        str(duration),
        str(output_path),
    ]


def _concat_cmd(ffmpeg_path: str, concat_list: Path, output_path: Path) -> list[str]:
    """Build FFmpeg concat command."""
    return [
        ffmpeg_path,
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_list),
        "-c",
        "copy",
        str(output_path),
    ]


# ---------------------------------------------------------------------------
# Thin subprocess runners
# ---------------------------------------------------------------------------


def _run_sync(cmd: list[str]) -> tuple[int, str, str]:
    """Run command synchronously, return (returncode, stdout, stderr).

    Raises:
        FFmpegNotFoundError: If the binary is not found.
    """
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    except FileNotFoundError as e:
        raise FFmpegNotFoundError(
            f"Binary not found at '{cmd[0]}'. Ensure FFmpeg is installed."
        ) from e
    return result.returncode, result.stdout, result.stderr


async def _run_async(cmd: list[str]) -> tuple[int, str, str]:
    """Run command asynchronously, return (returncode, stdout, stderr).

    Raises:
        FFmpegNotFoundError: If the binary is not found.
    """
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout_bytes, stderr_bytes = await proc.communicate()
    except FileNotFoundError as e:
        raise FFmpegNotFoundError(
            f"Binary not found at '{cmd[0]}'. Ensure FFmpeg is installed."
        ) from e
    return proc.returncode, stdout_bytes.decode(), stderr_bytes.decode()


# ---------------------------------------------------------------------------
# Public API: duration
# ---------------------------------------------------------------------------


def get_duration(ffmpeg_path: str, file_path: Path) -> float:
    """Get media file duration using ffprobe.

    Args:
        ffmpeg_path: Path to ffmpeg binary.
        file_path: Path to media file.

    Returns:
        Duration in seconds.

    Raises:
        FFmpegNotFoundError: If ffprobe is not found.
        InvalidInputError: If file cannot be probed.
    """
    cmd = _ffprobe_duration_cmd(derive_ffprobe_path(ffmpeg_path), file_path)
    rc, stdout, stderr = _run_sync(cmd)
    return _parse_duration(rc, stdout, stderr, file_path)


async def get_duration_async(ffmpeg_path: str, file_path: Path) -> float:
    """Get media file duration using ffprobe (async).

    Args:
        ffmpeg_path: Path to ffmpeg binary.
        file_path: Path to media file.

    Returns:
        Duration in seconds.

    Raises:
        FFmpegNotFoundError: If ffprobe is not found.
        InvalidInputError: If file cannot be probed.
    """
    cmd = _ffprobe_duration_cmd(derive_ffprobe_path(ffmpeg_path), file_path)
    rc, stdout, stderr = await _run_async(cmd)
    return _parse_duration(rc, stdout, stderr, file_path)


# ---------------------------------------------------------------------------
# Public API: media info
# ---------------------------------------------------------------------------


def probe_media_info(ffmpeg_path: str, file_path: Path) -> MediaInfo:
    """Probe media file for duration and video stream info in a single ffprobe call.

    Args:
        ffmpeg_path: Path to ffmpeg binary.
        file_path: Path to media file.

    Returns:
        MediaInfo with duration and optional video properties.

    Raises:
        FFmpegNotFoundError: If ffprobe is not found.
        InvalidInputError: If file cannot be probed.
    """
    cmd = _ffprobe_info_cmd(derive_ffprobe_path(ffmpeg_path), file_path)
    rc, stdout, stderr = _run_sync(cmd)
    return _parse_media_info(rc, stdout, stderr, file_path)


async def probe_media_info_async(ffmpeg_path: str, file_path: Path) -> MediaInfo:
    """Probe media file for duration and video stream info (async).

    Args:
        ffmpeg_path: Path to ffmpeg binary.
        file_path: Path to media file.

    Returns:
        MediaInfo with duration and optional video properties.

    Raises:
        FFmpegNotFoundError: If ffprobe is not found.
        InvalidInputError: If file cannot be probed.
    """
    cmd = _ffprobe_info_cmd(derive_ffprobe_path(ffmpeg_path), file_path)
    rc, stdout, stderr = await _run_async(cmd)
    return _parse_media_info(rc, stdout, stderr, file_path)


# ---------------------------------------------------------------------------
# Shared segment preparation
# ---------------------------------------------------------------------------


def _prepare_segment_jobs(
    input_path: Path,
    output_path: Path,
    segments: list[SpeechSegment],
    config: SilenceRemovalConfig,
    media_info: MediaInfo,
    tmp: Path,
) -> SegmentJobs:
    """Prepare segment extraction commands and concat file content.

    Returns:
        SegmentJobs with separate extract/silence commands and concat text.
    """
    part_paths: list[Path] = []
    extract_cmds: list[list[str]] = []

    for i, seg in enumerate(segments):
        part_path = tmp / f"part_{i:04d}{output_path.suffix}"
        part_paths.append(part_path)
        extract_cmds.append(
            _segment_extract_cmd(
                config.ffmpeg_path, input_path, seg, part_path, media_info.has_video
            )
        )

    # Build concat file lines and collect silence commands
    lines: list[str] = []
    silence_cmds: list[list[str]] = []
    for i, part_path in enumerate(part_paths):
        lines.append(f"file '{part_path}'")

        if i < len(part_paths) - 1:
            gap = segments[i + 1].start - segments[i].end
            if gap > config.min_silence_duration and config.target_silence_duration > 0:
                silence_path = tmp / f"silence_{i:04d}{output_path.suffix}"
                silence_cmds.append(
                    _silence_cmd(
                        config.ffmpeg_path,
                        silence_path,
                        config.target_silence_duration,
                        media_info,
                    )
                )
                lines.append(f"file '{silence_path}'")

    return SegmentJobs(
        extract_cmds=extract_cmds,
        silence_cmds=silence_cmds,
        concat_text="\n".join(lines),
    )


# ---------------------------------------------------------------------------
# Public API: process segments
# ---------------------------------------------------------------------------


def process_segments(
    input_path: Path,
    output_path: Path,
    segments: list[SpeechSegment],
    config: SilenceRemovalConfig,
    media_info: MediaInfo | None = None,
    on_progress: SyncProgressCallback | None = None,
) -> None:
    """Cut and concatenate segments using FFmpeg.

    Extracts each speech segment from the input file, inserts shortened
    silence gaps between them, and concatenates into the output file.

    Args:
        input_path: Path to source media file.
        output_path: Path for output file.
        segments: Speech segments to keep.
        config: Silence removal configuration.
        media_info: Pre-probed media info. If None, probes the file.
        on_progress: Optional sync callback for progress updates.

    Raises:
        ProcessingError: If FFmpeg processing fails.
    """
    if not segments:
        raise ProcessingError(
            "No speech segments to process — file may be entirely silent."
        )

    if media_info is None:
        media_info = probe_media_info(config.ffmpeg_path, input_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        jobs = _prepare_segment_jobs(
            input_path,
            output_path,
            segments,
            config,
            media_info,
            tmp,
        )

        total = jobs.total_commands
        step = 0

        for i, cmd in enumerate(jobs.extract_cmds):
            step += 1
            _notify_progress(
                on_progress,
                "extracting",
                step,
                total,
                f"Extracting segment {i + 1}/{len(jobs.extract_cmds)}",
            )
            rc, _, stderr = _run_sync(cmd)
            if rc != 0:
                raise ProcessingError(f"FFmpeg processing failed: {stderr.strip()}")

        for j, cmd in enumerate(jobs.silence_cmds):
            step += 1
            _notify_progress(
                on_progress,
                "generating_silence",
                step,
                total,
                f"Generating silence gap {j + 1}/{len(jobs.silence_cmds)}",
            )
            rc, _, stderr = _run_sync(cmd)
            if rc != 0:
                raise ProcessingError(f"FFmpeg processing failed: {stderr.strip()}")

        concat_list = tmp / "concat.txt"
        concat_list.write_text(jobs.concat_text)

        step += 1
        _notify_progress(
            on_progress,
            "concatenating",
            step,
            total,
            "Concatenating segments",
        )
        rc, _, stderr = _run_sync(
            _concat_cmd(config.ffmpeg_path, concat_list, output_path)
        )
        if rc != 0:
            raise ProcessingError(f"FFmpeg concatenation failed: {stderr.strip()}")

    log_info(
        "Segment processing complete",
        context={"segment_count": len(segments), "output_path": str(output_path)},
        logger_name=_LOGGER_NAME,
    )


async def process_segments_async(
    input_path: Path,
    output_path: Path,
    segments: list[SpeechSegment],
    config: SilenceRemovalConfig,
    media_info: MediaInfo | None = None,
    on_progress: AsyncProgressCallback | None = None,
) -> None:
    """Async version of process_segments using asyncio.subprocess.

    Args:
        input_path: Path to source media file.
        output_path: Path for output file.
        segments: Speech segments to keep.
        config: Silence removal configuration.
        media_info: Pre-probed media info. If None, probes the file.
        on_progress: Optional async callback for progress updates.

    Raises:
        ProcessingError: If FFmpeg processing fails.
    """
    if not segments:
        raise ProcessingError(
            "No speech segments to process — file may be entirely silent."
        )

    if media_info is None:
        media_info = await probe_media_info_async(config.ffmpeg_path, input_path)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)
        jobs = _prepare_segment_jobs(
            input_path,
            output_path,
            segments,
            config,
            media_info,
            tmp,
        )

        total = jobs.total_commands
        step = 0

        for i, cmd in enumerate(jobs.extract_cmds):
            step += 1
            await _notify_progress_async(
                on_progress,
                "extracting",
                step,
                total,
                f"Extracting segment {i + 1}/{len(jobs.extract_cmds)}",
            )
            rc, _, stderr = await _run_async(cmd)
            if rc != 0:
                raise ProcessingError(f"FFmpeg processing failed: {stderr.strip()}")

        for j, cmd in enumerate(jobs.silence_cmds):
            step += 1
            await _notify_progress_async(
                on_progress,
                "generating_silence",
                step,
                total,
                f"Generating silence gap {j + 1}/{len(jobs.silence_cmds)}",
            )
            rc, _, stderr = await _run_async(cmd)
            if rc != 0:
                raise ProcessingError(f"FFmpeg processing failed: {stderr.strip()}")

        concat_list = tmp / "concat.txt"
        concat_list.write_text(jobs.concat_text)

        step += 1
        await _notify_progress_async(
            on_progress,
            "concatenating",
            step,
            total,
            "Concatenating segments",
        )
        rc, _, stderr = await _run_async(
            _concat_cmd(config.ffmpeg_path, concat_list, output_path)
        )
        if rc != 0:
            raise ProcessingError(f"FFmpeg concatenation failed: {stderr.strip()}")

    log_info(
        "Segment processing complete",
        context={"segment_count": len(segments), "output_path": str(output_path)},
        logger_name=_LOGGER_NAME,
    )
