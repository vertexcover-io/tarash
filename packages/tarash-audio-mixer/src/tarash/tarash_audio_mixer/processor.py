"""Audio mixing processor — envelope generation and FFmpeg command building."""

from __future__ import annotations

import asyncio
import json
import math
import subprocess
from pathlib import Path
from typing import NamedTuple

from .exceptions import FFmpegNotFoundError, InvalidInputError, ProcessingError
from .models import AudioMixerConfig, SpeechSegment


# ---------------------------------------------------------------------------
# Internal types
# ---------------------------------------------------------------------------


class AudioInfo(NamedTuple):
    """Probed audio stream properties."""

    duration: float
    sample_rate: int
    channels: int


class DuckRegion(NamedTuple):
    """Internal representation of a ducking region with ramp boundaries."""

    attack_start: float
    attack_end: float
    full_duck_start: float
    full_duck_end: float
    release_start: float
    release_end: float


def compute_duck_regions(
    segments: list[SpeechSegment],
    config: AudioMixerConfig,
    total_duration: float,
) -> list[DuckRegion]:
    """Compute duck regions from speech segments.

    For each segment, applies speech padding and computes attack/release ramp
    boundaries. All times are clamped to [0, total_duration].
    """
    attack_s = config.attack_ms / 1000.0
    release_s = config.release_ms / 1000.0
    regions: list[DuckRegion] = []

    for seg in segments:
        duck_start = max(0.0, seg.start - config.speech_padding)
        duck_end = min(total_duration, seg.end + config.speech_padding)

        attack_start = max(0.0, duck_start - attack_s)
        attack_end = duck_start

        release_start = duck_end
        release_end = min(total_duration, duck_end + release_s)

        regions.append(
            DuckRegion(
                attack_start=attack_start,
                attack_end=attack_end,
                full_duck_start=attack_end,
                full_duck_end=release_start,
                release_start=release_start,
                release_end=release_end,
            )
        )

    return regions


def merge_duck_regions(regions: list[DuckRegion]) -> list[DuckRegion]:
    """Merge overlapping duck regions.

    Sorts by attack_start, then merges any regions whose boundaries overlap
    (i.e. one region's release overlaps another's attack).
    """
    if not regions:
        return []

    sorted_regions = sorted(regions, key=lambda r: r.attack_start)
    merged: list[DuckRegion] = [sorted_regions[0]]

    for current in sorted_regions[1:]:
        prev = merged[-1]
        # Overlap: current's attack starts before previous release ends
        if current.attack_start <= prev.release_end:
            merged[-1] = DuckRegion(
                attack_start=prev.attack_start,
                attack_end=prev.attack_end,
                full_duck_start=prev.full_duck_start,
                full_duck_end=current.full_duck_end,
                release_start=current.release_start,
                release_end=current.release_end,
            )
        else:
            merged.append(current)

    return merged


def build_volume_expression(
    regions: list[DuckRegion],
    config: AudioMixerConfig,
    total_duration: float,
) -> str:
    """Build an FFmpeg volume filter expression from merged duck regions.

    Converts regions into nested if(between(t,...)) clauses with linear
    interpolation for attack/release ramps.
    """
    base_gain = 10 ** (config.base_music_volume_db / 20.0)
    duck_gain = 10 ** ((config.base_music_volume_db + config.duck_level_db) / 20.0)

    if not regions:
        return str(base_gain)

    clauses: list[str] = []

    for region in regions:
        # Attack ramp: linear interpolation from base_gain to duck_gain
        if region.attack_start < region.attack_end:
            attack_dur = region.attack_end - region.attack_start
            # Linear interp: base_gain + (duck_gain - base_gain) * (t - attack_start) / attack_dur
            clauses.append(
                f"if(between(t,{region.attack_start},{region.attack_end}),"
                f"{base_gain}+({duck_gain}-{base_gain})"
                f"*(t-{region.attack_start})/{attack_dur})"
            )

        # Full duck: constant duck_gain
        if region.full_duck_start < region.full_duck_end:
            clauses.append(
                f"if(between(t,{region.full_duck_start},{region.full_duck_end}),"
                f"{duck_gain})"
            )

        # Release ramp: linear interpolation from duck_gain to base_gain
        if region.release_start < region.release_end:
            release_dur = region.release_end - region.release_start
            # Linear interp: duck_gain + (base_gain - duck_gain) * (t - release_start) / release_dur
            clauses.append(
                f"if(between(t,{region.release_start},{region.release_end}),"
                f"{duck_gain}+({base_gain}-{duck_gain})"
                f"*(t-{region.release_start})/{release_dur})"
            )

    if not clauses:
        return str(base_gain)

    # Build nested expression: each clause is tried in order, default is base_gain
    # Format: clause1 + clause2 + ... with default via subtraction trick
    # Actually, use nested if approach for clarity
    # if(cond1, val1, if(cond2, val2, ... base_gain))
    expr = str(base_gain)
    for clause in reversed(clauses):
        # clause is "if(between(...), value)"  — we need to insert the else branch
        # Remove trailing ")" and add else branch
        expr = clause[:-1] + "," + expr + ")"

    return expr


# ---------------------------------------------------------------------------
# FFmpeg subprocess runners
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
# Probing functions
# ---------------------------------------------------------------------------


def derive_ffprobe_path(ffmpeg_path: str) -> str:
    """Derive the ffprobe binary path from the ffmpeg binary path."""
    p = Path(ffmpeg_path)
    if p.name == "ffmpeg":
        return str(p.with_name("ffprobe"))
    return "ffprobe"


def _ffprobe_audio_cmd(ffprobe_path: str, file_path: Path) -> list[str]:
    """Build ffprobe command for audio stream info."""
    return [
        ffprobe_path,
        "-v",
        "error",
        "-select_streams",
        "a:0",
        "-show_entries",
        "stream=sample_rate,channels",
        "-show_entries",
        "format=duration",
        "-of",
        "json",
        str(file_path),
    ]


def _parse_audio_info(
    returncode: int, stdout: str, stderr: str, file_path: Path
) -> AudioInfo:
    """Parse ffprobe JSON output into AudioInfo."""
    if returncode != 0:
        raise InvalidInputError(f"Cannot read file '{file_path}': {stderr.strip()}")

    data = json.loads(stdout)
    duration = float(data["format"]["duration"])
    streams = data.get("streams", [])

    if not streams:
        raise InvalidInputError(f"No audio stream found in '{file_path}'")

    stream = streams[0]
    return AudioInfo(
        duration=duration,
        sample_rate=int(stream["sample_rate"]),
        channels=int(stream["channels"]),
    )


def probe_audio_info(ffmpeg_path: str, file_path: Path) -> AudioInfo:
    """Probe audio file for duration, sample_rate, channels.

    Args:
        ffmpeg_path: Path to ffmpeg binary.
        file_path: Path to audio file.

    Returns:
        AudioInfo with duration, sample_rate, channels.

    Raises:
        FFmpegNotFoundError: If ffprobe is not found.
        InvalidInputError: If file cannot be probed.
    """
    cmd = _ffprobe_audio_cmd(derive_ffprobe_path(ffmpeg_path), file_path)
    rc, stdout, stderr = _run_sync(cmd)
    return _parse_audio_info(rc, stdout, stderr, file_path)


async def probe_audio_info_async(ffmpeg_path: str, file_path: Path) -> AudioInfo:
    """Probe audio file for duration, sample_rate, channels (async).

    Args:
        ffmpeg_path: Path to ffmpeg binary.
        file_path: Path to audio file.

    Returns:
        AudioInfo with duration, sample_rate, channels.

    Raises:
        FFmpegNotFoundError: If ffprobe is not found.
        InvalidInputError: If file cannot be probed.
    """
    cmd = _ffprobe_audio_cmd(derive_ffprobe_path(ffmpeg_path), file_path)
    rc, stdout, stderr = await _run_async(cmd)
    return _parse_audio_info(rc, stdout, stderr, file_path)


# ---------------------------------------------------------------------------
# Background looping
# ---------------------------------------------------------------------------


def build_loop_filter(
    bg_duration: float, fg_duration: float, crossfade: float
) -> tuple[str, int]:
    """Build loop filter fragment for background audio.

    Args:
        bg_duration: Background audio duration in seconds.
        fg_duration: Foreground audio duration in seconds.
        crossfade: Crossfade seconds at loop boundaries.

    Returns:
        Tuple of (filter string fragment, loops_used count).
    """
    loops_needed = math.ceil(fg_duration / bg_duration)

    # EDGE-005: skip crossfade if background too short
    if bg_duration < crossfade * 2:
        crossfade = 0.0

    if crossfade > 0:
        # Use aloop with crossfade: loop N-1 times (aloop adds to original)
        loop_count = loops_needed - 1
        filter_str = (
            f"aloop=loop={loop_count}:size=2147483647,"
            f"acrossfade=d={crossfade}:c1=tri:c2=tri"
        )
    else:
        loop_count = loops_needed - 1
        filter_str = f"aloop=loop={loop_count}:size=2147483647"

    return filter_str, loops_needed


# ---------------------------------------------------------------------------
# Filter complex builder
# ---------------------------------------------------------------------------


def build_filter_complex(
    config: AudioMixerConfig,
    volume_expr: str,
    fg_info: AudioInfo,
    bg_info: AudioInfo,
) -> str:
    """Build FFmpeg filter_complex string for mixing.

    Input [0] = foreground, Input [1] = background.

    Args:
        config: Audio mixer configuration.
        volume_expr: Volume expression from build_volume_expression.
        fg_info: Foreground audio info.
        bg_info: Background audio info.

    Returns:
        Filter complex string for FFmpeg.
    """
    bg_filters: list[str] = []
    loops_used = 0

    # Background duration/looping/padding
    if bg_info.duration < fg_info.duration:
        if config.loop_background:
            loop_filter, loops_used = build_loop_filter(
                bg_info.duration, fg_info.duration, config.loop_crossfade
            )
            bg_filters.append(loop_filter)
            # Trim to foreground duration after looping
            bg_filters.append(f"atrim=end={fg_info.duration}")
        else:
            # Pad with silence then trim (REQ-012, EDGE-004)
            bg_filters.append(f"apad=whole_dur={fg_info.duration}")
            bg_filters.append(f"atrim=end={fg_info.duration}")
    elif bg_info.duration > fg_info.duration:
        # Trim background to foreground duration (EDGE-010)
        bg_filters.append(f"atrim=end={fg_info.duration}")

    # Resample if sample rates differ (REQ-014, EDGE-006)
    if bg_info.sample_rate != fg_info.sample_rate:
        bg_filters.append(f"aresample={fg_info.sample_rate}")

    # Channel convert if different (REQ-015, EDGE-007)
    if bg_info.channels != fg_info.channels:
        if fg_info.channels == 1:
            layout = "mono"
        elif fg_info.channels == 2:
            layout = "stereo"
        else:
            layout = f"{fg_info.channels}c"
        bg_filters.append(f"aformat=channel_layouts={layout}")

    # Apply volume expression for ducking
    bg_filters.append(f"volume='{volume_expr}':eval=frame")

    # Build background chain
    bg_chain = "[1:a]" + ",".join(bg_filters) + "[bg]"

    # Build foreground chain
    fg_gain_linear = 10 ** (config.foreground_gain_db / 20.0)
    if config.foreground_gain_db != 0.0:
        # REQ-008: apply foreground gain
        fg_chain = f"[0:a]volume={fg_gain_linear}[fg]"
    else:
        # REQ-009: pass through
        fg_chain = "[0:a]acopy[fg]"

    # Mix with amix
    mix = "[fg][bg]amix=inputs=2:duration=first:dropout_transition=0[out]"

    return ";".join([bg_chain, fg_chain, mix])


# ---------------------------------------------------------------------------
# Mix command builder
# ---------------------------------------------------------------------------


def build_mix_command(
    config: AudioMixerConfig,
    fg_path: Path,
    bg_path: Path,
    output_path: Path,
    filter_complex: str,
) -> list[str]:
    """Build full FFmpeg command for mixing.

    Args:
        config: Audio mixer configuration.
        fg_path: Path to foreground audio.
        bg_path: Path to background audio.
        output_path: Path for output file.
        filter_complex: Filter complex string.

    Returns:
        FFmpeg command as list of strings.
    """
    cmd = [
        config.ffmpeg_path,
        "-y",
        "-i",
        str(fg_path),
        "-i",
        str(bg_path),
        "-filter_complex",
        filter_complex,
        "-map",
        "[out]",
        str(output_path),
    ]
    return cmd


# ---------------------------------------------------------------------------
# Mix execution
# ---------------------------------------------------------------------------


def run_mix(
    config: AudioMixerConfig,
    fg_path: Path,
    bg_path: Path,
    output_path: Path,
    volume_expr: str,
    fg_info: AudioInfo,
    bg_info: AudioInfo,
) -> int:
    """Run the audio mix synchronously.

    Args:
        config: Audio mixer configuration.
        fg_path: Path to foreground audio.
        bg_path: Path to background audio.
        output_path: Path for output file.
        volume_expr: Volume expression from build_volume_expression.
        fg_info: Foreground audio info.
        bg_info: Background audio info.

    Returns:
        Number of loops used (0 if background was not looped).

    Raises:
        ProcessingError: If FFmpeg returns non-zero exit code.
    """
    filter_complex = build_filter_complex(config, volume_expr, fg_info, bg_info)
    cmd = build_mix_command(config, fg_path, bg_path, output_path, filter_complex)
    rc, _, stderr = _run_sync(cmd)
    if rc != 0:
        raise ProcessingError(f"FFmpeg mixing failed: {stderr.strip()}")

    # Determine loops used
    if bg_info.duration < fg_info.duration and config.loop_background:
        _, loops_used = build_loop_filter(
            bg_info.duration, fg_info.duration, config.loop_crossfade
        )
        return loops_used
    return 0


async def run_mix_async(
    config: AudioMixerConfig,
    fg_path: Path,
    bg_path: Path,
    output_path: Path,
    volume_expr: str,
    fg_info: AudioInfo,
    bg_info: AudioInfo,
) -> int:
    """Run the audio mix asynchronously.

    Args:
        config: Audio mixer configuration.
        fg_path: Path to foreground audio.
        bg_path: Path to background audio.
        output_path: Path for output file.
        volume_expr: Volume expression from build_volume_expression.
        fg_info: Foreground audio info.
        bg_info: Background audio info.

    Returns:
        Number of loops used (0 if background was not looped).

    Raises:
        ProcessingError: If FFmpeg returns non-zero exit code.
    """
    filter_complex = build_filter_complex(config, volume_expr, fg_info, bg_info)
    cmd = build_mix_command(config, fg_path, bg_path, output_path, filter_complex)
    rc, _, stderr = await _run_async(cmd)
    if rc != 0:
        raise ProcessingError(f"FFmpeg mixing failed: {stderr.strip()}")

    # Determine loops used
    if bg_info.duration < fg_info.duration and config.loop_background:
        _, loops_used = build_loop_filter(
            bg_info.duration, fg_info.duration, config.loop_crossfade
        )
        return loops_used
    return 0
