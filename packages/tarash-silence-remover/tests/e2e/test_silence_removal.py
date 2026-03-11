"""End-to-end tests for silence removal with real FFmpeg processing."""

import subprocess
from pathlib import Path

import pytest

from tarash.tarash_silence_remover import (
    SilenceRemovalConfig,
    SilenceRemovalRequest,
    SilenceRemovalResponse,
    detect_silence,
    detect_silence_async,
    remove_silence,
    remove_silence_async,
)


@pytest.fixture
def audio_with_silence(tmp_path, ffmpeg_available) -> Path:
    """Generate a test WAV: 1s tone, 2s silence, 1s tone, 1s silence, 1s tone = 6s total.

    Speech regions: 0-1s, 3-4s, 5-6s
    Silence regions: 1-3s, 4-5s
    """
    output = tmp_path / "test_audio.wav"
    # Generate using FFmpeg: concatenate tone-silence-tone-silence-tone
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        (
            "aevalsrc='if(between(t,0,1),sin(440*2*PI*t),"
            "if(between(t,3,4),sin(440*2*PI*t),"
            "if(between(t,5,6),sin(440*2*PI*t),0)))'"
            ":s=44100:d=6"
        ),
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"FFmpeg failed: {result.stderr}"
    return output


@pytest.fixture
def video_with_silence(tmp_path, ffmpeg_available) -> Path:
    """Generate a test MP4: video with audio that has silence gaps."""
    output = tmp_path / "test_video.mp4"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "color=c=blue:s=320x240:r=30:d=6",
        "-f",
        "lavfi",
        "-i",
        (
            "aevalsrc='if(between(t,0,1),sin(440*2*PI*t),"
            "if(between(t,3,4),sin(440*2*PI*t),"
            "if(between(t,5,6),sin(440*2*PI*t),0)))'"
            ":s=44100:d=6"
        ),
        "-c:v",
        "libx264",
        "-c:a",
        "aac",
        "-shortest",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"FFmpeg failed: {result.stderr}"
    return output


# ==================== FFmpeg Detector E2E ====================


@pytest.mark.e2e
def test_detect_silence_ffmpeg_audio(audio_with_silence):
    """FFmpeg detector finds speech segments in audio file."""
    config = SilenceRemovalConfig(
        detector="ffmpeg",
        silence_threshold_db=-30.0,
        min_silence_duration=0.5,
    )
    segments = detect_silence(config, audio_with_silence)

    assert len(segments) >= 2  # At least the tone regions
    # First speech segment should start near 0
    assert segments[0].start < 0.5


@pytest.mark.e2e
def test_remove_silence_ffmpeg_audio(audio_with_silence):
    """Full removal pipeline with FFmpeg detector on audio."""
    config = SilenceRemovalConfig(
        detector="ffmpeg",
        min_silence_duration=0.5,
        target_silence_duration=0.3,
        padding=0.1,
    )
    request = SilenceRemovalRequest(input_path=audio_with_silence)

    response = remove_silence(config, request)

    assert isinstance(response, SilenceRemovalResponse)
    assert response.output_path.exists()
    assert response.output_duration < response.original_duration
    assert response.removed_duration > 0
    assert response.detector_used == "ffmpeg"
    assert len(response.segments_kept) >= 2
    print(f"  Original: {response.original_duration:.1f}s")
    print(f"  Output: {response.output_duration:.1f}s")
    print(f"  Removed: {response.removed_duration:.1f}s")


@pytest.mark.e2e
def test_remove_silence_ffmpeg_video(video_with_silence):
    """Full removal pipeline with FFmpeg detector on video."""
    config = SilenceRemovalConfig(
        detector="ffmpeg",
        min_silence_duration=0.5,
        target_silence_duration=0.3,
    )
    request = SilenceRemovalRequest(input_path=video_with_silence)

    response = remove_silence(config, request)

    assert response.output_path.exists()
    assert response.output_duration < response.original_duration
    assert response.removed_duration > 0


@pytest.mark.e2e
def test_remove_silence_full_removal(audio_with_silence):
    """target_silence_duration=0 fully removes silence."""
    config = SilenceRemovalConfig(
        detector="ffmpeg",
        target_silence_duration=0.0,
    )
    request = SilenceRemovalRequest(input_path=audio_with_silence)

    response = remove_silence(config, request)
    assert response.output_duration < response.original_duration


@pytest.mark.e2e
def test_remove_silence_custom_output_path(audio_with_silence, tmp_path):
    """Custom output path is respected."""
    output = tmp_path / "custom_output.wav"
    config = SilenceRemovalConfig(detector="ffmpeg")
    request = SilenceRemovalRequest(
        input_path=audio_with_silence,
        output_path=output,
    )

    response = remove_silence(config, request)
    assert response.output_path == output
    assert output.exists()


@pytest.mark.e2e
async def test_remove_silence_async_ffmpeg(audio_with_silence):
    """Async pipeline works correctly."""
    config = SilenceRemovalConfig(detector="ffmpeg")
    request = SilenceRemovalRequest(input_path=audio_with_silence)

    response = await remove_silence_async(config, request)

    assert response.output_path.exists()
    assert response.output_duration < response.original_duration


# ==================== Silero Detector E2E ====================


def _silero_available() -> bool:
    try:
        import silero_vad  # noqa: F401

        return True
    except ImportError:
        return False


silero_required = pytest.mark.skipif(
    not _silero_available(), reason="silero-vad not installed"
)


@pytest.mark.e2e
@silero_required
def test_detect_silence_silero_audio(audio_with_silence):
    """Silero VAD detector finds speech segments in audio file."""
    config = SilenceRemovalConfig(
        detector="silero",
        vad_threshold=0.5,
        min_silence_duration=0.5,
    )
    segments = detect_silence(config, audio_with_silence)

    assert len(segments) >= 2  # At least the tone regions
    assert segments[0].start < 0.5


@pytest.mark.e2e
@silero_required
def test_remove_silence_silero_audio(audio_with_silence):
    """Full removal pipeline with Silero detector on audio."""
    config = SilenceRemovalConfig(
        detector="silero",
        min_silence_duration=0.5,
        target_silence_duration=0.3,
        padding=0.1,
    )
    request = SilenceRemovalRequest(input_path=audio_with_silence)

    response = remove_silence(config, request)

    assert isinstance(response, SilenceRemovalResponse)
    assert response.output_path.exists()
    assert response.output_duration < response.original_duration
    assert response.removed_duration > 0
    assert response.detector_used == "silero"
    assert len(response.segments_kept) >= 2
    print(f"  Original: {response.original_duration:.1f}s")
    print(f"  Output: {response.output_duration:.1f}s")
    print(f"  Removed: {response.removed_duration:.1f}s")


@pytest.mark.e2e
@silero_required
def test_remove_silence_silero_video(video_with_silence):
    """Full removal pipeline with Silero detector on video."""
    config = SilenceRemovalConfig(
        detector="silero",
        min_silence_duration=0.5,
        target_silence_duration=0.3,
    )
    request = SilenceRemovalRequest(input_path=video_with_silence)

    response = remove_silence(config, request)

    assert response.output_path.exists()
    assert response.output_duration < response.original_duration
    assert response.removed_duration > 0


@pytest.mark.e2e
@silero_required
def test_remove_silence_silero_full_removal(audio_with_silence):
    """target_silence_duration=0 fully removes silence with Silero."""
    config = SilenceRemovalConfig(
        detector="silero",
        target_silence_duration=0.0,
    )
    request = SilenceRemovalRequest(input_path=audio_with_silence)

    response = remove_silence(config, request)
    assert response.output_duration < response.original_duration


@pytest.mark.e2e
@silero_required
async def test_remove_silence_async_silero(audio_with_silence):
    """Async pipeline works correctly with Silero detector."""
    config = SilenceRemovalConfig(detector="silero")
    request = SilenceRemovalRequest(input_path=audio_with_silence)

    response = await remove_silence_async(config, request)

    assert response.output_path.exists()
    assert response.output_duration < response.original_duration
    assert response.detector_used == "silero"


@pytest.mark.e2e
@silero_required
async def test_detect_silence_async_silero(audio_with_silence):
    """Async Silero detection returns speech segments."""
    config = SilenceRemovalConfig(
        detector="silero",
        vad_threshold=0.5,
        min_silence_duration=0.5,
    )
    segments = await detect_silence_async(config, audio_with_silence)

    assert len(segments) >= 2
    assert segments[0].start < 0.5
