"""Tests for the exception hierarchy."""

from tarash.tarash_silence_remover.exceptions import (
    DetectionError,
    FFmpegNotFoundError,
    InvalidInputError,
    ProcessingError,
    SilenceRemoverException,
)


def test_base_exception_has_message():
    ex = SilenceRemoverException("something failed")
    assert ex.message == "something failed"
    assert str(ex) == "something failed"


def test_base_exception_is_exception():
    assert issubclass(SilenceRemoverException, Exception)


def test_ffmpeg_not_found_inherits_base():
    ex = FFmpegNotFoundError("ffmpeg not found at /usr/bin/ffmpeg")
    assert isinstance(ex, SilenceRemoverException)
    assert ex.message == "ffmpeg not found at /usr/bin/ffmpeg"


def test_invalid_input_inherits_base():
    ex = InvalidInputError("file not found: test.mp4")
    assert isinstance(ex, SilenceRemoverException)


def test_processing_error_inherits_base():
    ex = ProcessingError("ffmpeg exited with code 1")
    assert isinstance(ex, SilenceRemoverException)


def test_detection_error_inherits_base():
    ex = DetectionError("silero vad failed")
    assert isinstance(ex, SilenceRemoverException)
