"""Tests for data models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from tarash.tarash_audio_mixer.models import (
    AudioMixerConfig,
    AudioMixerRequest,
    AudioMixerResponse,
    SpeechSegment,
)


# ==================== AudioMixerConfig ====================


def test_config_defaults():
    config = AudioMixerConfig()
    assert config.duck_level_db == -12.0
    assert config.attack_ms == 200.0
    assert config.release_ms == 300.0
    assert config.speech_padding == 0.3
    assert config.base_music_volume_db == -6.0
    assert config.foreground_gain_db == 0.0
    assert config.loop_background is True
    assert config.loop_crossfade == 2.0
    assert config.vad_threshold == 0.5
    assert config.output_format is None
    assert config.ffmpeg_path == "ffmpeg"
    assert config.device is None


def test_config_custom_values():
    config = AudioMixerConfig(
        duck_level_db=-18.0,
        attack_ms=100.0,
        release_ms=150.0,
        speech_padding=0.5,
        base_music_volume_db=-10.0,
        foreground_gain_db=3.0,
        loop_background=False,
        loop_crossfade=1.0,
        vad_threshold=0.7,
        output_format="wav",
        ffmpeg_path="/usr/local/bin/ffmpeg",
        device="cuda",
    )
    assert config.duck_level_db == -18.0
    assert config.attack_ms == 100.0
    assert config.release_ms == 150.0
    assert config.speech_padding == 0.5
    assert config.base_music_volume_db == -10.0
    assert config.foreground_gain_db == 3.0
    assert config.loop_background is False
    assert config.loop_crossfade == 1.0
    assert config.vad_threshold == 0.7
    assert config.output_format == "wav"
    assert config.ffmpeg_path == "/usr/local/bin/ffmpeg"
    assert config.device == "cuda"


def test_config_is_frozen():
    config = AudioMixerConfig()
    with pytest.raises(ValidationError):
        config.duck_level_db = -6.0


def test_config_duck_level_must_be_nonpositive():
    with pytest.raises(ValidationError):
        AudioMixerConfig(duck_level_db=5.0)


def test_config_negative_attack_rejected():
    with pytest.raises(ValidationError):
        AudioMixerConfig(attack_ms=-10.0)


def test_config_negative_release_rejected():
    with pytest.raises(ValidationError):
        AudioMixerConfig(release_ms=-10.0)


def test_config_negative_padding_rejected():
    with pytest.raises(ValidationError):
        AudioMixerConfig(speech_padding=-0.1)


def test_config_vad_threshold_out_of_range():
    with pytest.raises(ValidationError):
        AudioMixerConfig(vad_threshold=1.5)
    with pytest.raises(ValidationError):
        AudioMixerConfig(vad_threshold=-0.1)


# ==================== AudioMixerRequest ====================


def test_request_fields():
    req = AudioMixerRequest(
        foreground_path=Path("/tmp/speech.wav"),
        background_path=Path("/tmp/music.mp3"),
    )
    assert req.foreground_path == Path("/tmp/speech.wav")
    assert req.background_path == Path("/tmp/music.mp3")


def test_request_output_path_defaults_to_none():
    req = AudioMixerRequest(
        foreground_path=Path("/tmp/speech.wav"),
        background_path=Path("/tmp/music.mp3"),
    )
    assert req.output_path is None


# ==================== SpeechSegment ====================


def test_speech_segment_creation():
    seg = SpeechSegment(start=1.5, end=3.2)
    assert seg.start == 1.5
    assert seg.end == 3.2


def test_speech_segment_is_frozen():
    seg = SpeechSegment(start=1.0, end=2.0)
    with pytest.raises(ValidationError):
        seg.start = 5.0


# ==================== AudioMixerResponse ====================


def test_response_fields():
    resp = AudioMixerResponse(
        output_path=Path("/tmp/output.wav"),
        foreground_duration=10.0,
        background_duration=30.0,
        output_duration=10.0,
        speech_segments=[
            SpeechSegment(start=0.0, end=3.0),
            SpeechSegment(start=5.5, end=10.0),
        ],
        loops_used=0,
    )
    assert resp.output_path == Path("/tmp/output.wav")
    assert resp.foreground_duration == 10.0
    assert resp.background_duration == 30.0
    assert resp.output_duration == 10.0
    assert len(resp.speech_segments) == 2
    assert resp.loops_used == 0


def test_response_is_frozen():
    resp = AudioMixerResponse(
        output_path=Path("/tmp/output.wav"),
        foreground_duration=10.0,
        background_duration=30.0,
        output_duration=10.0,
        speech_segments=[],
        loops_used=0,
    )
    with pytest.raises(ValidationError):
        resp.foreground_duration = 20.0
