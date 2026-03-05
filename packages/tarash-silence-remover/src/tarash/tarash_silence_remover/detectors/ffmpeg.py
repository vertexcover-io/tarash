"""FFmpeg-based silence detector using the silencedetect audio filter."""

import asyncio
import re
import subprocess
from pathlib import Path

from tarash.tarash_silence_remover.exceptions import (
    DetectionError,
    FFmpegNotFoundError,
    InvalidInputError,
)
from tarash.tarash_silence_remover.logging import log_info
from tarash.tarash_silence_remover.models import SilenceRemovalConfig, SpeechSegment
from tarash.tarash_silence_remover.processor import (
    get_duration,
    get_duration_async,
)

_LOGGER_NAME = "tarash.tarash_silence_remover.detectors.ffmpeg"

_SILENCE_START_RE = re.compile(r"silence_start:\s*([\d.]+)")
_SILENCE_END_RE = re.compile(r"silence_end:\s*([\d.]+)")


def parse_silencedetect_output(stderr: str) -> list[tuple[float, float]]:
    """Parse FFmpeg silencedetect filter output into silence intervals.

    Args:
        stderr: Raw stderr output from FFmpeg silencedetect.

    Returns:
        List of (start, end) tuples for each silence interval.
    """
    starts: list[float] = []
    ends: list[float] = []

    for line in stderr.splitlines():
        start_match = _SILENCE_START_RE.search(line)
        if start_match:
            starts.append(float(start_match.group(1)))

        end_match = _SILENCE_END_RE.search(line)
        if end_match:
            ends.append(float(end_match.group(1)))

    # Pair up starts and ends; ignore unterminated silences
    return list(zip(starts, ends, strict=False))


def invert_silences_to_speech(
    silences: list[tuple[float, float]],
    total_duration: float,
) -> list[SpeechSegment]:
    """Invert silence intervals to get speech segments.

    Args:
        silences: List of (start, end) silence intervals.
        total_duration: Total duration of the file in seconds.

    Returns:
        List of SpeechSegment covering non-silent regions.
    """
    if not silences:
        return [SpeechSegment(start=0.0, end=total_duration)]

    segments: list[SpeechSegment] = []
    cursor = 0.0

    for silence_start, silence_end in sorted(silences):
        if cursor < silence_start:
            segments.append(SpeechSegment(start=cursor, end=silence_start))
        cursor = silence_end

    if cursor < total_duration:
        segments.append(SpeechSegment(start=cursor, end=total_duration))

    return segments


def _get_duration(ffmpeg_path: str, file_path: Path) -> float:
    """Get file duration using ffprobe.

    Wraps processor.get_duration, re-raising InvalidInputError as DetectionError.
    """
    try:
        return get_duration(ffmpeg_path, file_path)
    except InvalidInputError as e:
        raise DetectionError(f"ffprobe failed: {e.message}") from e


async def _get_duration_async(ffmpeg_path: str, file_path: Path) -> float:
    """Get file duration using ffprobe (async).

    Wraps processor.get_duration_async, re-raising InvalidInputError as DetectionError.
    """
    try:
        return await get_duration_async(ffmpeg_path, file_path)
    except InvalidInputError as e:
        raise DetectionError(f"ffprobe failed: {e.message}") from e


class FFmpegDetector:
    """Silence detector using FFmpeg's silencedetect audio filter.

    Uses `ffmpeg -af silencedetect` to find silence intervals, then
    inverts them to produce speech segments.
    """

    def detect_speech_segments(
        self,
        audio_path: Path,
        config: SilenceRemovalConfig,
        duration: float | None = None,
    ) -> list[SpeechSegment]:
        """Detect speech segments using FFmpeg silencedetect.

        Args:
            audio_path: Path to the input file.
            config: Silence removal configuration.
            duration: Pre-probed file duration. If None, probes via ffprobe.

        Returns:
            List of SpeechSegment with start/end timestamps for speech regions.

        Raises:
            FFmpegNotFoundError: If FFmpeg binary is not found.
            DetectionError: If FFmpeg processing fails.
        """
        total_duration = (
            duration
            if duration is not None
            else _get_duration(config.ffmpeg_path, audio_path)
        )

        silence_filter = (
            f"silencedetect=noise={config.silence_threshold_db}dB"
            f":d={config.min_silence_duration}"
        )

        try:
            result = subprocess.run(
                [
                    config.ffmpeg_path,
                    "-i",
                    str(audio_path),
                    "-af",
                    silence_filter,
                    "-f",
                    "null",
                    "-",
                ],
                capture_output=True,
                text=True,
                check=False,
            )
        except FileNotFoundError as e:
            raise FFmpegNotFoundError(
                f"FFmpeg not found at '{config.ffmpeg_path}'. Ensure FFmpeg is installed."
            ) from e

        if result.returncode != 0 and "silencedetect" not in result.stderr:
            raise DetectionError(
                f"FFmpeg silencedetect failed: {result.stderr.strip()}"
            )

        silences = parse_silencedetect_output(result.stderr)
        log_info(
            "FFmpeg silence detection complete",
            context={"silence_interval_count": len(silences)},
            logger_name=_LOGGER_NAME,
        )

        return invert_silences_to_speech(silences, total_duration)

    async def detect_speech_segments_async(
        self,
        audio_path: Path,
        config: SilenceRemovalConfig,
        duration: float | None = None,
    ) -> list[SpeechSegment]:
        """Detect speech segments using FFmpeg silencedetect (async).

        Args:
            audio_path: Path to the input file.
            config: Silence removal configuration.
            duration: Pre-probed file duration. If None, probes via ffprobe.

        Returns:
            List of SpeechSegment with start/end timestamps for speech regions.

        Raises:
            FFmpegNotFoundError: If FFmpeg binary is not found.
            DetectionError: If FFmpeg processing fails.
        """
        total_duration = (
            duration
            if duration is not None
            else await _get_duration_async(config.ffmpeg_path, audio_path)
        )

        silence_filter = (
            f"silencedetect=noise={config.silence_threshold_db}dB"
            f":d={config.min_silence_duration}"
        )

        try:
            proc = await asyncio.create_subprocess_exec(
                config.ffmpeg_path,
                "-i",
                str(audio_path),
                "-af",
                silence_filter,
                "-f",
                "null",
                "-",
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            _, stderr = await proc.communicate()
        except FileNotFoundError as e:
            raise FFmpegNotFoundError(
                f"FFmpeg not found at '{config.ffmpeg_path}'. Ensure FFmpeg is installed."
            ) from e

        stderr_text = stderr.decode()
        if proc.returncode != 0 and "silencedetect" not in stderr_text:
            raise DetectionError(f"FFmpeg silencedetect failed: {stderr_text.strip()}")

        silences = parse_silencedetect_output(stderr_text)
        log_info(
            "FFmpeg silence detection complete",
            context={"silence_interval_count": len(silences)},
            logger_name=_LOGGER_NAME,
        )

        return invert_silences_to_speech(silences, total_duration)
