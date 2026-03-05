"""Tests for data models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from tarash.tarash_silence_remover.models import (
    SilenceRemovalConfig,
    SilenceRemovalRequest,
    SilenceRemovalResponse,
    SpeechSegment,
)


# ==================== SilenceRemovalConfig ====================


def test_config_defaults():
    config = SilenceRemovalConfig()
    assert config.detector == "silero"
    assert config.min_silence_duration == 0.5
    assert config.target_silence_duration == 0.3
    assert config.padding == 0.1
    assert config.silence_threshold_db == -30.0
    assert config.vad_threshold == 0.5
    assert config.ffmpeg_path == "ffmpeg"


def test_config_custom_values():
    config = SilenceRemovalConfig(
        detector="ffmpeg",
        min_silence_duration=1.0,
        target_silence_duration=0.0,
        padding=0.2,
        silence_threshold_db=-40.0,
        vad_threshold=0.7,
        ffmpeg_path="/usr/local/bin/ffmpeg",
    )
    assert config.detector == "ffmpeg"
    assert config.min_silence_duration == 1.0
    assert config.target_silence_duration == 0.0
    assert config.padding == 0.2


def test_config_is_frozen():
    config = SilenceRemovalConfig()
    with pytest.raises(ValidationError):
        config.detector = "ffmpeg"


def test_config_invalid_detector():
    with pytest.raises(ValidationError):
        SilenceRemovalConfig(detector="invalid")


def test_config_negative_padding_rejected():
    with pytest.raises(ValidationError):
        SilenceRemovalConfig(padding=-0.1)


def test_config_vad_threshold_out_of_range_rejected():
    with pytest.raises(ValidationError):
        SilenceRemovalConfig(vad_threshold=1.5)
    with pytest.raises(ValidationError):
        SilenceRemovalConfig(vad_threshold=-0.1)


def test_config_positive_silence_threshold_rejected():
    with pytest.raises(ValidationError):
        SilenceRemovalConfig(silence_threshold_db=5.0)


# ==================== SilenceRemovalRequest ====================


def test_request_with_input_only():
    req = SilenceRemovalRequest(input_path=Path("/tmp/video.mp4"))
    assert req.input_path == Path("/tmp/video.mp4")
    assert req.output_path is None


def test_request_with_output():
    req = SilenceRemovalRequest(
        input_path=Path("/tmp/video.mp4"),
        output_path=Path("/tmp/output.mp4"),
    )
    assert req.output_path == Path("/tmp/output.mp4")


# ==================== SpeechSegment ====================


def test_speech_segment():
    seg = SpeechSegment(start=1.5, end=3.2)
    assert seg.start == 1.5
    assert seg.end == 3.2


def test_speech_segment_is_frozen():
    seg = SpeechSegment(start=1.0, end=2.0)
    with pytest.raises(ValidationError):
        seg.start = 5.0


# ==================== SilenceRemovalResponse ====================


def test_response_fields():
    resp = SilenceRemovalResponse(
        output_path=Path("/tmp/output.mp4"),
        original_duration=10.0,
        output_duration=7.5,
        segments_kept=[
            SpeechSegment(start=0.0, end=3.0),
            SpeechSegment(start=5.5, end=10.0),
        ],
        detector_used="silero",
    )
    assert resp.output_path == Path("/tmp/output.mp4")
    assert resp.original_duration == 10.0
    assert resp.removed_duration == 2.5
    assert len(resp.segments_kept) == 2
    assert resp.detector_used == "silero"


def test_response_is_frozen():
    resp = SilenceRemovalResponse(
        output_path=Path("/tmp/output.mp4"),
        original_duration=10.0,
        output_duration=7.5,
        segments_kept=[],
        detector_used="ffmpeg",
    )
    with pytest.raises(ValidationError):
        resp.original_duration = 20.0
