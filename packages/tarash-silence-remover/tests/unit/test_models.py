"""Tests for data models."""

from pathlib import Path

import pytest
from pydantic import ValidationError

from tarash.tarash_silence_remover.models import (
    AsyncProgressCallback,
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


# ==================== ProcessingUpdate ====================


def test_processing_update_valid_construction():
    """ProcessingUpdate can be constructed with valid fields."""
    update = ProcessingUpdate(
        phase="extracting",
        progress_percent=50,
        current_step=3,
        total_steps=6,
        message="Extracting segment 3/6",
    )
    assert update.phase == "extracting"
    assert update.progress_percent == 50
    assert update.current_step == 3
    assert update.total_steps == 6
    assert update.message == "Extracting segment 3/6"


def test_processing_update_is_frozen():
    """ProcessingUpdate is immutable."""
    update = ProcessingUpdate(
        phase="probing",
        progress_percent=0,
        current_step=1,
        total_steps=10,
        message="Probing media file",
    )
    with pytest.raises(ValidationError):
        update.phase = "detecting"


def test_processing_update_rejects_progress_below_zero():
    """progress_percent rejects values < 0."""
    with pytest.raises(ValidationError):
        ProcessingUpdate(
            phase="extracting",
            progress_percent=-1,
            current_step=1,
            total_steps=5,
            message="test",
        )


def test_processing_update_rejects_progress_above_100():
    """progress_percent rejects values > 100."""
    with pytest.raises(ValidationError):
        ProcessingUpdate(
            phase="extracting",
            progress_percent=101,
            current_step=1,
            total_steps=5,
            message="test",
        )


def test_processing_update_rejects_current_step_below_one():
    """current_step rejects values < 1."""
    with pytest.raises(ValidationError):
        ProcessingUpdate(
            phase="extracting",
            progress_percent=50,
            current_step=0,
            total_steps=5,
            message="test",
        )


def test_processing_update_rejects_total_steps_below_one():
    """total_steps rejects values < 1."""
    with pytest.raises(ValidationError):
        ProcessingUpdate(
            phase="extracting",
            progress_percent=50,
            current_step=1,
            total_steps=0,
            message="test",
        )


def test_processing_update_all_five_phases_accepted():
    """All five ProcessingPhase values are accepted."""
    phases = [
        "probing",
        "detecting",
        "extracting",
        "generating_silence",
        "concatenating",
    ]
    for phase in phases:
        update = ProcessingUpdate(
            phase=phase,
            progress_percent=50,
            current_step=1,
            total_steps=1,
            message="test",
        )
        assert update.phase == phase


def test_processing_update_rejects_invalid_phase():
    """Invalid phase value is rejected."""
    with pytest.raises(ValidationError):
        ProcessingUpdate(
            phase="invalid_phase",
            progress_percent=50,
            current_step=1,
            total_steps=5,
            message="test",
        )


def test_callback_type_aliases_exist():
    """Callback type aliases are importable from models."""
    # Smoke test: these should be type aliases, not None
    assert ProcessingPhase is not None
    assert SyncProgressCallback is not None
    assert AsyncProgressCallback is not None
    assert ProgressCallback is not None


# ==================== Package exports ====================


def test_progress_types_exported_from_package():
    """All five progress types are importable from tarash.tarash_silence_remover."""
    from tarash.tarash_silence_remover import (
        AsyncProgressCallback,
        ProcessingPhase,
        ProcessingUpdate,
        ProgressCallback,
        SyncProgressCallback,
    )

    assert ProcessingUpdate is not None
    assert ProcessingPhase is not None
    assert ProgressCallback is not None
    assert SyncProgressCallback is not None
    assert AsyncProgressCallback is not None


def test_progress_types_in_package_all():
    """Progress types are listed in __all__."""
    import tarash.tarash_silence_remover as pkg

    expected = [
        "ProcessingUpdate",
        "ProcessingPhase",
        "ProgressCallback",
        "SyncProgressCallback",
        "AsyncProgressCallback",
    ]
    for name in expected:
        assert name in pkg.__all__, f"{name} not in __all__"


# ==================== SilenceRemovalPreview ====================


def test_preview_construction_with_valid_fields():
    """SilenceRemovalPreview can be constructed with all required fields."""
    segments = [
        SpeechSegment(start=0.0, end=3.0),
        SpeechSegment(start=5.0, end=8.0),
    ]
    preview = SilenceRemovalPreview(
        original_duration=10.0,
        estimated_output_duration=6.3,
        segments_to_keep=segments,
        silence_gaps_to_insert=1,
        detector_used="silero",
    )
    assert preview.original_duration == 10.0
    assert preview.estimated_output_duration == 6.3
    assert len(preview.segments_to_keep) == 2
    assert preview.silence_gaps_to_insert == 1
    assert preview.detector_used == "silero"


def test_preview_estimated_removed_duration():
    """estimated_removed_duration returns original minus estimated output."""
    preview = SilenceRemovalPreview(
        original_duration=10.0,
        estimated_output_duration=6.0,
        segments_to_keep=[SpeechSegment(start=0.0, end=6.0)],
        silence_gaps_to_insert=0,
        detector_used="ffmpeg",
    )
    assert preview.estimated_removed_duration == 4.0


def test_preview_reduction_percent():
    """reduction_percent returns correct percentage."""
    preview = SilenceRemovalPreview(
        original_duration=10.0,
        estimated_output_duration=7.0,
        segments_to_keep=[SpeechSegment(start=0.0, end=7.0)],
        silence_gaps_to_insert=0,
        detector_used="silero",
    )
    assert preview.reduction_percent == pytest.approx(30.0)


def test_preview_reduction_percent_zero_duration():
    """reduction_percent returns 0.0 when original_duration is 0."""
    preview = SilenceRemovalPreview(
        original_duration=0.0,
        estimated_output_duration=0.0,
        segments_to_keep=[],
        silence_gaps_to_insert=0,
        detector_used="silero",
    )
    assert preview.reduction_percent == 0.0


def test_preview_is_frozen():
    """SilenceRemovalPreview is immutable."""
    preview = SilenceRemovalPreview(
        original_duration=10.0,
        estimated_output_duration=7.0,
        segments_to_keep=[],
        silence_gaps_to_insert=0,
        detector_used="silero",
    )
    with pytest.raises(ValidationError):
        preview.original_duration = 20.0


def test_preview_accepts_empty_segments():
    """segments_to_keep accepts an empty list (all silence case)."""
    preview = SilenceRemovalPreview(
        original_duration=5.0,
        estimated_output_duration=0.0,
        segments_to_keep=[],
        silence_gaps_to_insert=0,
        detector_used="ffmpeg",
    )
    assert preview.segments_to_keep == []
    assert preview.estimated_removed_duration == 5.0
    assert preview.reduction_percent == 100.0


def test_preview_exported_from_package():
    """SilenceRemovalPreview is importable from tarash.tarash_silence_remover."""
    from tarash.tarash_silence_remover import SilenceRemovalPreview as Imported

    assert Imported is SilenceRemovalPreview


def test_preview_in_package_all():
    """SilenceRemovalPreview is listed in __all__."""
    import tarash.tarash_silence_remover as pkg

    assert "SilenceRemovalPreview" in pkg.__all__
