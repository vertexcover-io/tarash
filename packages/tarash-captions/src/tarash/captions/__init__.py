"""Tarash Captions - AI-powered video caption generation."""

from __future__ import annotations

from tarash.captions.api import generate_caption
from tarash.captions.models import CaptionConfig, CaptionRequest, CaptionResponse

__all__ = ["CaptionConfig", "CaptionRequest", "CaptionResponse", "generate_caption"]

import importlib.metadata

try:
    __version__ = importlib.metadata.version("tarash-captions")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0"
