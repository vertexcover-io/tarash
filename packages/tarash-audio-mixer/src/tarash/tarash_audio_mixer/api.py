"""Public API for tarash-audio-mixer."""

from pathlib import Path

from tarash.tarash_audio_mixer import detector
from tarash.tarash_audio_mixer.exceptions import InvalidInputError
from tarash.tarash_audio_mixer.logging import log_info
from tarash.tarash_audio_mixer.models import (
    AudioMixerConfig,
    AudioMixerRequest,
    AudioMixerResponse,
    SpeechSegment,
)
from tarash.tarash_audio_mixer.processor import (
    build_volume_expression,
    compute_duck_regions,
    merge_duck_regions,
    probe_audio_info,
    probe_audio_info_async,
    run_mix,
    run_mix_async,
)

_LOGGER_NAME = "tarash.tarash_audio_mixer.api"


def _resolve_output_path(
    fg_path: Path,
    output_path: Path | None,
    output_format: str | None,
) -> Path:
    """Resolve the output path for the mixed audio file.

    Args:
        fg_path: Foreground audio file path.
        output_path: Explicit output path, or None for default.
        output_format: Output format extension, or None to match foreground.

    Returns:
        Resolved output path.
    """
    if output_path is not None:
        return output_path
    if output_format is not None:
        return fg_path.parent / f"{fg_path.stem}_mixed.{output_format}"
    return fg_path.parent / f"{fg_path.stem}_mixed{fg_path.suffix}"


def detect_speech(
    config: AudioMixerConfig,
    foreground_path: Path,
) -> list[SpeechSegment]:
    """Detect speech segments in a foreground audio file.

    Args:
        config: Audio mixer configuration.
        foreground_path: Path to the foreground audio file.

    Returns:
        List of detected speech segments.

    Raises:
        InvalidInputError: If foreground_path does not exist.
        DetectionError: If speech detection fails.
    """
    if not foreground_path.exists():
        raise InvalidInputError(f"Foreground file does not exist: {foreground_path}")
    return detector.detect_speech_segments(foreground_path, config)


async def detect_speech_async(
    config: AudioMixerConfig,
    foreground_path: Path,
) -> list[SpeechSegment]:
    """Detect speech segments in a foreground audio file (async).

    Args:
        config: Audio mixer configuration.
        foreground_path: Path to the foreground audio file.

    Returns:
        List of detected speech segments.

    Raises:
        InvalidInputError: If foreground_path does not exist.
        DetectionError: If speech detection fails.
    """
    if not foreground_path.exists():
        raise InvalidInputError(f"Foreground file does not exist: {foreground_path}")
    return await detector.detect_speech_segments_async(foreground_path, config)


def mix_audio(
    config: AudioMixerConfig,
    request: AudioMixerRequest,
) -> AudioMixerResponse:
    """Mix foreground speech with background music using ducking (sync).

    Args:
        config: Audio mixer configuration.
        request: Audio mixing parameters.

    Returns:
        AudioMixerResponse with output path and metrics.

    Raises:
        InvalidInputError: If input files don't exist.
        FFmpegNotFoundError: If FFmpeg is not found.
        DetectionError: If speech detection fails.
        ProcessingError: If FFmpeg processing fails.
    """
    if not request.foreground_path.exists():
        raise InvalidInputError(
            f"Foreground file does not exist: {request.foreground_path}"
        )
    if not request.background_path.exists():
        raise InvalidInputError(
            f"Background file does not exist: {request.background_path}"
        )

    output_path = _resolve_output_path(
        request.foreground_path, request.output_path, config.output_format
    )

    fg_info = probe_audio_info(config.ffmpeg_path, request.foreground_path)
    bg_info = probe_audio_info(config.ffmpeg_path, request.background_path)

    segments = detector.detect_speech_segments(request.foreground_path, config)

    duck_regions = compute_duck_regions(segments, config, fg_info.duration)
    merged_regions = merge_duck_regions(duck_regions)
    volume_expr = build_volume_expression(merged_regions, config, fg_info.duration)

    loops_used = run_mix(
        config,
        request.foreground_path,
        request.background_path,
        output_path,
        volume_expr,
        fg_info,
        bg_info,
    )

    output_info = probe_audio_info(config.ffmpeg_path, output_path)

    log_info(
        "Audio mixing complete",
        context={
            "foreground_duration": round(fg_info.duration, 5),
            "background_duration": round(bg_info.duration, 5),
            "output_duration": round(output_info.duration, 5),
            "speech_segments": len(segments),
            "loops_used": loops_used,
        },
        logger_name=_LOGGER_NAME,
    )

    return AudioMixerResponse(
        output_path=output_path,
        foreground_duration=fg_info.duration,
        background_duration=bg_info.duration,
        output_duration=output_info.duration,
        speech_segments=segments,
        loops_used=loops_used,
    )


async def mix_audio_async(
    config: AudioMixerConfig,
    request: AudioMixerRequest,
) -> AudioMixerResponse:
    """Mix foreground speech with background music using ducking (async).

    Args:
        config: Audio mixer configuration.
        request: Audio mixing parameters.

    Returns:
        AudioMixerResponse with output path and metrics.

    Raises:
        InvalidInputError: If input files don't exist.
        FFmpegNotFoundError: If FFmpeg is not found.
        DetectionError: If speech detection fails.
        ProcessingError: If FFmpeg processing fails.
    """
    if not request.foreground_path.exists():
        raise InvalidInputError(
            f"Foreground file does not exist: {request.foreground_path}"
        )
    if not request.background_path.exists():
        raise InvalidInputError(
            f"Background file does not exist: {request.background_path}"
        )

    output_path = _resolve_output_path(
        request.foreground_path, request.output_path, config.output_format
    )

    fg_info = await probe_audio_info_async(config.ffmpeg_path, request.foreground_path)
    bg_info = await probe_audio_info_async(config.ffmpeg_path, request.background_path)

    segments = await detector.detect_speech_segments_async(
        request.foreground_path, config
    )

    duck_regions = compute_duck_regions(segments, config, fg_info.duration)
    merged_regions = merge_duck_regions(duck_regions)
    volume_expr = build_volume_expression(merged_regions, config, fg_info.duration)

    loops_used = await run_mix_async(
        config,
        request.foreground_path,
        request.background_path,
        output_path,
        volume_expr,
        fg_info,
        bg_info,
    )

    output_info = await probe_audio_info_async(config.ffmpeg_path, output_path)

    log_info(
        "Audio mixing complete",
        context={
            "foreground_duration": round(fg_info.duration, 5),
            "background_duration": round(bg_info.duration, 5),
            "output_duration": round(output_info.duration, 5),
            "speech_segments": len(segments),
            "loops_used": loops_used,
        },
        logger_name=_LOGGER_NAME,
    )

    return AudioMixerResponse(
        output_path=output_path,
        foreground_duration=fg_info.duration,
        background_duration=bg_info.duration,
        output_duration=output_info.duration,
        speech_segments=segments,
        loops_used=loops_used,
    )
