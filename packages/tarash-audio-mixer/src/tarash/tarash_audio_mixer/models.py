"""Core data models for audio mixing."""

from pathlib import Path
from typing import ClassVar

from pydantic import BaseModel, ConfigDict, Field


class AudioMixerConfig(BaseModel):
    """Configuration for audio mixing with ducking."""

    duck_level_db: float = Field(
        default=-12.0,
        le=0,
        description="dB reduction relative to base during speech.",
    )
    attack_ms: float = Field(
        default=200.0,
        ge=0,
        description="Attack ramp in milliseconds.",
    )
    release_ms: float = Field(
        default=300.0,
        ge=0,
        description="Release ramp in milliseconds.",
    )
    speech_padding: float = Field(
        default=0.3,
        ge=0,
        description="Seconds of padding around speech for duck regions.",
    )
    base_music_volume_db: float = Field(
        default=-6.0,
        description="Background volume in dB for non-speech sections.",
    )
    foreground_gain_db: float = Field(
        default=0.0,
        description="Gain applied to foreground audio in dB.",
    )
    loop_background: bool = Field(
        default=True,
        description="Loop background if shorter than foreground.",
    )
    loop_crossfade: float = Field(
        default=2.0,
        ge=0,
        description="Crossfade seconds at loop boundaries.",
    )
    vad_threshold: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Silero VAD speech probability threshold (0.0-1.0).",
    )
    output_format: str | None = Field(
        default=None,
        description="Output container format.",
    )
    ffmpeg_path: str = Field(
        default="ffmpeg",
        description="Path to FFmpeg binary.",
    )
    device: str | None = Field(
        default=None,
        description="Torch device for Silero VAD: 'cpu', 'cuda', etc. "
        "None = auto-detect.",
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)


class AudioMixerRequest(BaseModel):
    """Parameters for an audio mixing operation."""

    foreground_path: Path = Field(
        description="Path to speech/foreground audio file.",
    )
    background_path: Path = Field(
        description="Path to music/background audio file.",
    )
    output_path: Path | None = Field(
        default=None,
        description="Path for output file.",
    )


class SpeechSegment(BaseModel):
    """A detected speech segment with timestamps."""

    start: float = Field(description="Start time in seconds.")
    end: float = Field(description="End time in seconds.")

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)


class AudioMixerResponse(BaseModel):
    """Result of an audio mixing operation."""

    output_path: Path = Field(description="Path to the output file.")
    foreground_duration: float = Field(
        description="Foreground audio duration in seconds."
    )
    background_duration: float = Field(
        description="Background audio duration in seconds."
    )
    output_duration: float = Field(description="Output file duration in seconds.")
    speech_segments: list[SpeechSegment] = Field(
        description="Detected speech segments."
    )
    loops_used: int = Field(
        description="How many times background was looped (0 if not looped)."
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)
