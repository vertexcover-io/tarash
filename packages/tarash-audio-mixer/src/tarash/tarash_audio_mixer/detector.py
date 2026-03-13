"""Speech detection using Silero VAD."""

import asyncio
import logging
from pathlib import Path

from tarash.tarash_audio_mixer.exceptions import DetectionError
from tarash.tarash_audio_mixer.models import AudioMixerConfig, SpeechSegment

_LOGGER_NAME = "tarash.tarash_audio_mixer.detector"
_logger = logging.getLogger(_LOGGER_NAME)

_SILERO_SAMPLE_RATE = 16000


def _silero_available() -> bool:
    """Check if silero-vad is importable."""
    try:
        import silero_vad  # noqa: F401

        return True
    except ImportError:
        return False


def detect_speech_segments(
    audio_path: Path,
    config: AudioMixerConfig,
) -> list[SpeechSegment]:
    """Detect speech segments using Silero VAD.

    Args:
        audio_path: Path to the input audio file.
        config: Audio mixer configuration.

    Returns:
        List of SpeechSegment with start/end timestamps for speech regions.
        Returns empty list if no speech is detected.

    Raises:
        DetectionError: If silero-vad is not installed or detection fails.
    """
    if not _silero_available():
        raise DetectionError(
            "silero-vad is not installed. "
            "Install it with: pip install tarash-audio-mixer[silero]"
        )

    try:
        import torch
        from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
    except ImportError as e:
        raise DetectionError(f"Failed to import silero-vad: {e}") from e

    torch.set_num_threads(1)

    try:
        model = load_silero_vad()
        _logger.info("Silero VAD model loaded")
    except Exception as e:
        raise DetectionError(f"Failed to load Silero VAD model: {e}") from e

    try:
        wav = read_audio(str(audio_path), sampling_rate=_SILERO_SAMPLE_RATE)
        model.reset_states()

        speech_timestamps = get_speech_timestamps(
            wav,
            model,
            threshold=config.vad_threshold,
            sampling_rate=_SILERO_SAMPLE_RATE,
            return_seconds=True,
        )
    except Exception as e:
        raise DetectionError(f"Silero VAD processing failed: {e}") from e

    segments = [
        SpeechSegment(
            start=round(entry["start"], 5),
            end=round(entry["end"], 5),
        )
        for entry in speech_timestamps
    ]

    _logger.info(
        "Silero VAD detection complete: %d segments from %s",
        len(segments),
        audio_path,
    )

    return segments


async def detect_speech_segments_async(
    audio_path: Path,
    config: AudioMixerConfig,
) -> list[SpeechSegment]:
    """Detect speech segments using Silero VAD (async).

    Uses asyncio.to_thread for CPU-bound Silero inference.

    Args:
        audio_path: Path to the input audio file.
        config: Audio mixer configuration.

    Returns:
        List of SpeechSegment with start/end timestamps for speech regions.

    Raises:
        DetectionError: If silero-vad is not installed or detection fails.
    """
    if not _silero_available():
        raise DetectionError(
            "silero-vad is not installed. "
            "Install it with: pip install tarash-audio-mixer[silero]"
        )

    return await asyncio.to_thread(detect_speech_segments, audio_path, config)
