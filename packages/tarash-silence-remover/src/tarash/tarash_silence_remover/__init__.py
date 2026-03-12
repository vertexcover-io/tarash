"""Tarash Silence Remover - Remove silent segments from video and audio files."""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("tarash-silence-remover")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0"

from .api import (
    detect_silence,
    detect_silence_async,
    preview_silence,
    preview_silence_async,
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
    AsyncProgressCallback,
    DetectorBackend,
    MediaInfo,
    ProcessingPhase,
    ProcessingUpdate,
    ProgressCallback,
    SilenceRemovalConfig,
    SilenceRemovalPreview,
    SilenceRemovalRequest,
    SilenceRemovalResponse,
    SpeechSegment,
    SyncProgressCallback,
)

__all__ = [
    # API
    "remove_silence",
    "remove_silence_async",
    "detect_silence",
    "detect_silence_async",
    "preview_silence",
    "preview_silence_async",
    # Models
    "DetectorBackend",
    "MediaInfo",
    "SilenceRemovalConfig",
    "SilenceRemovalPreview",
    "SilenceRemovalRequest",
    "SilenceRemovalResponse",
    "SpeechSegment",
    # Progress reporting
    "ProcessingUpdate",
    "ProcessingPhase",
    "ProgressCallback",
    "SyncProgressCallback",
    "AsyncProgressCallback",
    # Exceptions
    "SilenceRemoverException",
    "FFmpegNotFoundError",
    "InvalidInputError",
    "ProcessingError",
    "DetectionError",
]
