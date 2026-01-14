"""Tarash Scene Detector - Video scene detection utilities."""

import importlib.metadata

try:
    __version__ = importlib.metadata.version("tarash-scene-detector")
except importlib.metadata.PackageNotFoundError:
    __version__ = "0.1.0"


def hello() -> str:
    return "Hello from scenedetect!"
