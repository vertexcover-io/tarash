"""Tarash Audio Mixer - Mix foreground speech with background music using intelligent ducking."""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("tarash-audio-mixer")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0"

from .api import (
    detect_speech,
    detect_speech_async,
    mix_audio,
    mix_audio_async,
)
from .exceptions import (
    AudioMixerException,
    DetectionError,
    FFmpegNotFoundError,
    InvalidInputError,
    ProcessingError,
)
from .models import (
    AudioMixerConfig,
    AudioMixerRequest,
    AudioMixerResponse,
    SpeechSegment,
)

__all__ = [
    # API
    "mix_audio",
    "mix_audio_async",
    "detect_speech",
    "detect_speech_async",
    # Models
    "AudioMixerConfig",
    "AudioMixerRequest",
    "AudioMixerResponse",
    "SpeechSegment",
    # Exceptions
    "AudioMixerException",
    "FFmpegNotFoundError",
    "InvalidInputError",
    "ProcessingError",
    "DetectionError",
]
