"""Silero VAD-based silence detector."""

import asyncio
from pathlib import Path

from tarash.tarash_silence_remover.exceptions import DetectionError
from tarash.tarash_silence_remover.logging import log_info, log_warning
from tarash.tarash_silence_remover.models import SilenceRemovalConfig, SpeechSegment

_LOGGER_NAME = "tarash.tarash_silence_remover.detectors.silero"

_SILERO_SAMPLE_RATE = 16000


def _silero_available() -> bool:
    """Check if silero-vad is importable."""
    try:
        import silero_vad  # noqa: F401

        return True
    except ImportError:
        return False


class SileroDetector:
    """Silence detector using Silero VAD.

    Falls back to FFmpegDetector if silero-vad is not installed.
    """

    def detect_speech_segments(
        self,
        audio_path: Path,
        config: SilenceRemovalConfig,
        duration: float | None = None,
    ) -> list[SpeechSegment]:
        """Detect speech segments using Silero VAD.

        Args:
            audio_path: Path to the input file.
            config: Silence removal configuration.
            duration: Pre-probed file duration (passed through on FFmpeg fallback).

        Returns:
            List of SpeechSegment with start/end timestamps for speech regions.

        Raises:
            DetectionError: If detection fails.
        """
        if not _silero_available():
            log_warning(
                "silero-vad not installed, falling back to FFmpeg detector",
                context={"install_hint": "pip install tarash-silence-remover[silero]"},
                logger_name=_LOGGER_NAME,
            )
            from tarash.tarash_silence_remover.detectors.ffmpeg import FFmpegDetector

            return FFmpegDetector().detect_speech_segments(
                audio_path, config, duration=duration
            )

        # duration is only used by the FFmpeg fallback; Silero reads audio directly
        return self._detect_with_silero(audio_path, config)

    async def detect_speech_segments_async(
        self,
        audio_path: Path,
        config: SilenceRemovalConfig,
        duration: float | None = None,
    ) -> list[SpeechSegment]:
        """Detect speech segments using Silero VAD (async).

        Falls back to FFmpegDetector async path if torch is unavailable.
        Uses asyncio.to_thread for CPU-bound Silero inference.

        Args:
            audio_path: Path to the input file.
            config: Silence removal configuration.
            duration: Pre-probed file duration (passed through on FFmpeg fallback).

        Returns:
            List of SpeechSegment with start/end timestamps for speech regions.

        Raises:
            DetectionError: If detection fails.
        """
        if not _silero_available():
            log_warning(
                "silero-vad not installed, falling back to FFmpeg detector",
                context={"install_hint": "pip install tarash-silence-remover[silero]"},
                logger_name=_LOGGER_NAME,
            )
            from tarash.tarash_silence_remover.detectors.ffmpeg import FFmpegDetector

            return await FFmpegDetector().detect_speech_segments_async(
                audio_path, config, duration=duration
            )

        # duration is only used by the FFmpeg fallback; Silero reads audio directly
        return await asyncio.to_thread(self._detect_with_silero, audio_path, config)

    def _detect_with_silero(
        self,
        audio_path: Path,
        config: SilenceRemovalConfig,
    ) -> list[SpeechSegment]:
        """Run Silero VAD on the audio file.

        Silero VAD is a tiny model (~2MB) designed for CPU inference.
        GPU adds unnecessary overhead for such a small model.

        Args:
            audio_path: Path to the input file.
            config: Silence removal configuration.

        Returns:
            List of SpeechSegment from VAD results.
        """
        try:
            import torch
            from silero_vad import get_speech_timestamps, load_silero_vad, read_audio
        except ImportError as e:
            raise DetectionError(f"Failed to import silero-vad: {e}") from e

        torch.set_num_threads(1)

        try:
            model = load_silero_vad()
            log_info("Silero VAD model loaded", logger_name=_LOGGER_NAME)
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
                min_silence_duration_ms=int(config.min_silence_duration * 1000),
                speech_pad_ms=int(config.padding * 1000),
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
        log_info(
            "Silero VAD detection complete",
            context={"segment_count": len(segments), "audio_path": str(audio_path)},
            logger_name=_LOGGER_NAME,
        )

        return segments
