"""Tests for the exception hierarchy."""

from tarash.tarash_audio_mixer.exceptions import (
    AudioMixerException,
    DetectionError,
    FFmpegNotFoundError,
    InvalidInputError,
    ProcessingError,
)


def test_base_exception_message():
    ex = AudioMixerException("something failed")
    assert ex.message == "something failed"


def test_ffmpeg_not_found_inherits_base():
    ex = FFmpegNotFoundError("ffmpeg not found at /usr/bin/ffmpeg")
    assert isinstance(ex, AudioMixerException)
    assert ex.message == "ffmpeg not found at /usr/bin/ffmpeg"


def test_invalid_input_inherits_base():
    ex = InvalidInputError("file not found: test.mp4")
    assert isinstance(ex, AudioMixerException)


def test_processing_error_inherits_base():
    ex = ProcessingError("ffmpeg exited with code 1")
    assert isinstance(ex, AudioMixerException)


def test_detection_error_inherits_base():
    ex = DetectionError("silero vad failed")
    assert isinstance(ex, AudioMixerException)


def test_exception_str_matches_message():
    ex = AudioMixerException("test error")
    assert str(ex) == "test error"
