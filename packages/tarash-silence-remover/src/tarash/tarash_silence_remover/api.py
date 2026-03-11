"""Public API for tarash-silence-remover."""

from pathlib import Path

from tarash.tarash_silence_remover.detectors.ffmpeg import FFmpegDetector
from tarash.tarash_silence_remover.detectors.silero import SileroDetector
from tarash.tarash_silence_remover.exceptions import InvalidInputError
from tarash.tarash_silence_remover.logging import log_info
from tarash.tarash_silence_remover.models import (
    AsyncProgressCallback,
    SilenceRemovalConfig,
    SilenceRemovalRequest,
    SilenceRemovalResponse,
    SpeechSegment,
    SyncProgressCallback,
)
from tarash.tarash_silence_remover.processor import (
    _notify_progress,
    _notify_progress_async,
    apply_padding,
    get_duration,
    get_duration_async,
    merge_overlapping_segments,
    probe_media_info,
    probe_media_info_async,
    process_segments,
    process_segments_async,
)

_LOGGER_NAME = "tarash.tarash_silence_remover.api"


def _resolve_output_path(input_path: Path, output_path: Path | None) -> Path:
    """Resolve the output path, defaulting to '<input>_cleaned.<ext>'.

    Args:
        input_path: Original input file path.
        output_path: Explicit output path, or None for default.

    Returns:
        Resolved output path.
    """
    if output_path is not None:
        return output_path
    return input_path.with_stem(f"{input_path.stem}_cleaned")


def _get_detector(config: SilenceRemovalConfig) -> FFmpegDetector | SileroDetector:
    """Get the appropriate detector based on config.

    Args:
        config: Silence removal configuration.

    Returns:
        A SilenceDetector instance.
    """
    if config.detector == "ffmpeg":
        return FFmpegDetector()
    return SileroDetector()


def detect_silence(
    config: SilenceRemovalConfig,
    input_path: Path,
) -> list[SpeechSegment]:
    """Detect speech segments without processing.

    Useful for inspecting what would be kept/removed before committing.

    Args:
        config: Silence removal configuration.
        input_path: Path to the input file.

    Returns:
        List of detected speech segments.
    """
    detector = _get_detector(config)
    return detector.detect_speech_segments(input_path, config)


async def detect_silence_async(
    config: SilenceRemovalConfig,
    input_path: Path,
) -> list[SpeechSegment]:
    """Detect speech segments without processing (async).

    Args:
        config: Silence removal configuration.
        input_path: Path to the input file.

    Returns:
        List of detected speech segments.
    """
    detector = _get_detector(config)
    return await detector.detect_speech_segments_async(input_path, config)


def remove_silence(
    config: SilenceRemovalConfig,
    request: SilenceRemovalRequest,
    on_progress: SyncProgressCallback | None = None,
) -> SilenceRemovalResponse:
    """Remove silence from a video or audio file (sync).

    Args:
        config: Silence removal configuration.
        request: Silence removal parameters.
        on_progress: Optional sync callback for progress updates.

    Returns:
        SilenceRemovalResponse with output path and metrics.

    Raises:
        InvalidInputError: If input file doesn't exist.
        FFmpegNotFoundError: If FFmpeg is not found.
        DetectionError: If silence detection fails.
        ProcessingError: If FFmpeg processing fails.
    """
    if not request.input_path.exists():
        raise InvalidInputError(f"Input file does not exist: {request.input_path}")

    output_path = _resolve_output_path(request.input_path, request.output_path)

    _notify_progress(
        on_progress, "probing", 1, 1, "Probing media file", progress_percent=0
    )
    media_info = probe_media_info(config.ffmpeg_path, request.input_path)
    original_duration = media_info.duration

    # Detect speech segments
    _notify_progress(
        on_progress, "detecting", 1, 1, "Detecting speech segments", progress_percent=5
    )
    detector = _get_detector(config)
    raw_segments = detector.detect_speech_segments(
        request.input_path, config, duration=original_duration
    )

    # Apply padding and merge overlapping segments
    padded = apply_padding(raw_segments, config.padding, original_duration)
    merged = merge_overlapping_segments(padded)

    # Process (cut and concatenate)
    process_segments(
        request.input_path,
        output_path,
        merged,
        config,
        media_info=media_info,
        on_progress=on_progress,
    )

    # Measure output duration
    output_duration = get_duration(config.ffmpeg_path, output_path)

    log_info(
        "Silence removal complete",
        context={
            "original_duration": round(original_duration, 5),
            "output_duration": round(output_duration, 5),
            "removed_duration": round(original_duration - output_duration, 5),
        },
        logger_name=_LOGGER_NAME,
    )

    return SilenceRemovalResponse(
        output_path=output_path,
        original_duration=original_duration,
        output_duration=output_duration,
        segments_kept=merged,
        detector_used=config.detector,
    )


async def remove_silence_async(
    config: SilenceRemovalConfig,
    request: SilenceRemovalRequest,
    on_progress: AsyncProgressCallback | None = None,
) -> SilenceRemovalResponse:
    """Remove silence from a video or audio file (async).

    Args:
        config: Silence removal configuration.
        request: Silence removal parameters.
        on_progress: Optional async callback for progress updates.

    Returns:
        SilenceRemovalResponse with output path and metrics.

    Raises:
        InvalidInputError: If input file doesn't exist.
        FFmpegNotFoundError: If FFmpeg is not found.
        DetectionError: If silence detection fails.
        ProcessingError: If FFmpeg processing fails.
    """
    if not request.input_path.exists():
        raise InvalidInputError(f"Input file does not exist: {request.input_path}")

    output_path = _resolve_output_path(request.input_path, request.output_path)

    await _notify_progress_async(
        on_progress, "probing", 1, 1, "Probing media file", progress_percent=0
    )
    media_info = await probe_media_info_async(config.ffmpeg_path, request.input_path)
    original_duration = media_info.duration

    # Detect speech segments
    await _notify_progress_async(
        on_progress, "detecting", 1, 1, "Detecting speech segments", progress_percent=5
    )
    detector = _get_detector(config)
    raw_segments = await detector.detect_speech_segments_async(
        request.input_path, config, duration=original_duration
    )

    # Apply padding and merge
    padded = apply_padding(raw_segments, config.padding, original_duration)
    merged = merge_overlapping_segments(padded)

    # Process async
    await process_segments_async(
        request.input_path,
        output_path,
        merged,
        config,
        media_info=media_info,
        on_progress=on_progress,
    )

    # Measure output
    output_duration = await get_duration_async(config.ffmpeg_path, output_path)

    log_info(
        "Silence removal complete",
        context={
            "original_duration": round(original_duration, 5),
            "output_duration": round(output_duration, 5),
            "removed_duration": round(original_duration - output_duration, 5),
        },
        logger_name=_LOGGER_NAME,
    )

    return SilenceRemovalResponse(
        output_path=output_path,
        original_duration=original_duration,
        output_duration=output_duration,
        segments_kept=merged,
        detector_used=config.detector,
    )
