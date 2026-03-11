"""Core data models for silence removal."""

from collections.abc import Awaitable, Callable
from pathlib import Path
from typing import ClassVar, Literal

from pydantic import BaseModel, ConfigDict, Field

ProcessingPhase = Literal[
    "probing", "detecting", "extracting", "generating_silence", "concatenating"
]


class ProcessingUpdate(BaseModel):
    """Progress update fired during silence removal processing."""

    phase: ProcessingPhase
    progress_percent: int = Field(ge=0, le=100, description="Overall progress 0-100.")
    current_step: int = Field(ge=1, description="Current step number (1-based).")
    total_steps: int = Field(ge=1, description="Total number of steps.")
    message: str = Field(description="Human-readable status message for CLI display.")

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)


SyncProgressCallback = Callable[[ProcessingUpdate], None]
AsyncProgressCallback = Callable[[ProcessingUpdate], Awaitable[None]]
ProgressCallback = SyncProgressCallback | AsyncProgressCallback


class SilenceRemovalConfig(BaseModel):
    """Configuration for silence removal."""

    detector: Literal["silero", "ffmpeg"] = Field(
        default="silero",
        description="Detection backend: 'silero' (VAD, more accurate) or 'ffmpeg' (lightweight).",
    )
    min_silence_duration: float = Field(
        default=0.5,
        ge=0,
        description="Minimum silence duration (seconds) to trigger removal.",
    )
    target_silence_duration: float = Field(
        default=0.3,
        ge=0,
        description="Duration (seconds) to shorten silence to. Set to 0 for full removal.",
    )
    padding: float = Field(
        default=0.1,
        ge=0,
        description="Padding (seconds) to keep before/after speech segments.",
    )
    silence_threshold_db: float = Field(
        default=-30.0,
        le=0,
        description="Volume threshold in dB for FFmpeg silence detection.",
    )
    vad_threshold: float = Field(
        default=0.5,
        ge=0,
        le=1,
        description="Silero VAD speech probability threshold (0.0-1.0).",
    )
    device: str | None = Field(
        default=None,
        description="Torch device for Silero VAD: 'cpu', 'cuda', 'cuda:0', etc. "
        "None = auto-detect (cuda if available, else cpu).",
    )
    ffmpeg_path: str = Field(
        default="ffmpeg",
        description="Path to FFmpeg binary.",
    )

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)


class SilenceRemovalRequest(BaseModel):
    """Parameters for a silence removal operation."""

    input_path: Path = Field(
        description="Path to input video or audio file.",
    )
    output_path: Path | None = Field(
        default=None,
        description="Path for output file. Defaults to '<input>_cleaned.<ext>'.",
    )


class SpeechSegment(BaseModel):
    """A detected speech segment with timestamps."""

    start: float = Field(description="Start time in seconds.")
    end: float = Field(description="End time in seconds.")

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)


class MediaInfo(BaseModel):
    """Probed media file information."""

    duration: float = Field(description="File duration in seconds.")
    video_width: int | None = Field(default=None)
    video_height: int | None = Field(default=None)
    video_fps: str | None = Field(default=None, description="e.g. '30/1'")

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    @property
    def has_video(self) -> bool:
        return self.video_width is not None


class SilenceRemovalResponse(BaseModel):
    """Result of a silence removal operation."""

    output_path: Path = Field(description="Path to the output file.")
    original_duration: float = Field(description="Original file duration in seconds.")
    output_duration: float = Field(description="Output file duration in seconds.")
    segments_kept: list[SpeechSegment] = Field(
        description="Speech segments that were kept."
    )
    detector_used: str = Field(description="Which detector backend was used.")

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    @property
    def removed_duration(self) -> float:
        """Total silence removed in seconds."""
        return self.original_duration - self.output_duration
