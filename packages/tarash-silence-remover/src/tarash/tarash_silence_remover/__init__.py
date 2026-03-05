"""Tarash Silence Remover - Remove silent segments from video and audio files."""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("tarash-silence-remover")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0"

from .api import (
    detect_silence,
    detect_silence_async,
    remove_silence,
    remove_silence_async,
)
from .exceptions import (
    DetectionError,
    FFmpegNotFoundError,
    InvalidInputError,
    ProcessingError,
    SilenceRemoverException,
)
from .models import (
    MediaInfo,
    SilenceRemovalConfig,
    SilenceRemovalRequest,
    SilenceRemovalResponse,
    SpeechSegment,
)

__all__ = [
    # API
    "remove_silence",
    "remove_silence_async",
    "detect_silence",
    "detect_silence_async",
    # Models
    "MediaInfo",
    "SilenceRemovalConfig",
    "SilenceRemovalRequest",
    "SilenceRemovalResponse",
    "SpeechSegment",
    # Exceptions
    "SilenceRemoverException",
    "FFmpegNotFoundError",
    "InvalidInputError",
    "ProcessingError",
    "DetectionError",
]
