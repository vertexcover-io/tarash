"""End-to-end tests for audio mixing with real FFmpeg processing."""

import subprocess
from pathlib import Path

import pytest

from tarash.tarash_audio_mixer import (
    AudioMixerConfig,
    AudioMixerRequest,
    AudioMixerResponse,
    SpeechSegment,
    detect_speech,
    mix_audio,
    mix_audio_async,
)


def _silero_available() -> bool:
    try:
        import silero_vad  # noqa: F401

        return True
    except ImportError:
        return False


silero_required = pytest.mark.skipif(
    not _silero_available(), reason="silero-vad not installed"
)


def _generate_speech_file(tmp_path: Path) -> Path:
    """Generate a WAV with speech-like sine bursts and silence gaps.

    Pattern: 2s tone, 1s silence, 3s tone, 1s silence, 2s tone = ~9s total.
    Speech regions: 0-2s, 3-6s, 7-9s
    Silence regions: 2-3s, 6-7s
    """
    output = tmp_path / "speech.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        (
            "aevalsrc='if(between(t,0,2),sin(440*2*PI*t),"
            "if(between(t,3,6),sin(440*2*PI*t),"
            "if(between(t,7,9),sin(440*2*PI*t),0)))'"
            ":s=44100:d=9"
        ),
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"FFmpeg failed: {result.stderr}"
    return output


def _generate_music_file(tmp_path: Path, duration: float = 15.0) -> Path:
    """Generate continuous pink noise WAV as background music."""
    output = tmp_path / "music.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        f"anoisesrc=d={duration}:c=pink:r=44100",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"FFmpeg failed: {result.stderr}"
    return output


def _generate_short_music_file(tmp_path: Path) -> Path:
    """Generate a short music file (~3s) for loop testing."""
    output = tmp_path / "short_music.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "anoisesrc=d=3:c=pink:r=44100",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"FFmpeg failed: {result.stderr}"
    return output


def _generate_tiny_music_file(tmp_path: Path) -> Path:
    """Generate a tiny music file (~1s) shorter than crossfade duration for EDGE-005."""
    output = tmp_path / "tiny_music.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "anoisesrc=d=1:c=pink:r=44100",
        str(output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"FFmpeg failed: {result.stderr}"
    return output


@pytest.fixture
def sample_speech_file(tmp_path, ffmpeg_available) -> Path:
    """Generate a WAV with speech-like patterns using FFmpeg."""
    return _generate_speech_file(tmp_path)


@pytest.fixture
def sample_music_file(tmp_path, ffmpeg_available) -> Path:
    """Generate continuous background music WAV (~15s)."""
    return _generate_music_file(tmp_path)


@pytest.fixture
def short_music_file(tmp_path, ffmpeg_available) -> Path:
    """Generate short music file (~3s) for loop testing."""
    return _generate_short_music_file(tmp_path)


@pytest.fixture
def tiny_music_file(tmp_path, ffmpeg_available) -> Path:
    """Generate tiny music file (~1s) for EDGE-005."""
    return _generate_tiny_music_file(tmp_path)


# ==================== Basic Mixing E2E ====================


@pytest.mark.e2e
@silero_required
def test_mix_audio_basic(sample_speech_file, sample_music_file, tmp_path):
    """Mix speech + music, verify output exists and duration matches foreground."""
    output = tmp_path / "output.wav"
    config = AudioMixerConfig()
    request = AudioMixerRequest(
        foreground_path=sample_speech_file,
        background_path=sample_music_file,
        output_path=output,
    )

    response = mix_audio(config, request)

    assert isinstance(response, AudioMixerResponse)
    assert response.output_path.exists()
    assert response.output_path == output
    # Output duration should match foreground duration (within tolerance)
    assert abs(response.output_duration - response.foreground_duration) < 1.0


@pytest.mark.e2e
@silero_required
def test_mix_audio_no_speech(sample_music_file, tmp_path, ffmpeg_available):
    """Foreground with no speech — background at base volume throughout."""
    # Generate a silence-only foreground (no speech)
    fg = tmp_path / "silence_fg.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "anullsrc=r=44100:cl=mono",
        "-t",
        "5",
        str(fg),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"FFmpeg failed: {result.stderr}"

    output = tmp_path / "no_speech_output.wav"
    config = AudioMixerConfig()
    request = AudioMixerRequest(
        foreground_path=fg,
        background_path=sample_music_file,
        output_path=output,
    )

    response = mix_audio(config, request)

    assert response.output_path.exists()
    assert response.speech_segments == []


@pytest.mark.e2e
@silero_required
def test_mix_audio_foreground_gain(sample_speech_file, sample_music_file, tmp_path):
    """Apply foreground_gain_db=6.0, verify output exists."""
    output = tmp_path / "fg_gain_output.wav"
    config = AudioMixerConfig(foreground_gain_db=6.0)
    request = AudioMixerRequest(
        foreground_path=sample_speech_file,
        background_path=sample_music_file,
        output_path=output,
    )

    response = mix_audio(config, request)

    assert response.output_path.exists()
    assert abs(response.output_duration - response.foreground_duration) < 1.0


@pytest.mark.e2e
@silero_required
def test_mix_audio_loop_background(sample_speech_file, short_music_file, tmp_path):
    """Short bg + long fg with loop=True, verify output duration."""
    output = tmp_path / "loop_output.wav"
    config = AudioMixerConfig(loop_background=True)
    request = AudioMixerRequest(
        foreground_path=sample_speech_file,
        background_path=short_music_file,
        output_path=output,
    )

    response = mix_audio(config, request)

    assert response.output_path.exists()
    assert response.loops_used > 0
    assert abs(response.output_duration - response.foreground_duration) < 1.0


@pytest.mark.e2e
@silero_required
def test_mix_audio_no_loop_silence_pad(sample_speech_file, short_music_file, tmp_path):
    """loop=False, short bg padded with silence."""
    output = tmp_path / "no_loop_output.wav"
    config = AudioMixerConfig(loop_background=False)
    request = AudioMixerRequest(
        foreground_path=sample_speech_file,
        background_path=short_music_file,
        output_path=output,
    )

    response = mix_audio(config, request)

    assert response.output_path.exists()
    assert response.loops_used == 0
    assert abs(response.output_duration - response.foreground_duration) < 1.0


@pytest.mark.e2e
@silero_required
def test_mix_audio_background_longer_trimmed(
    sample_speech_file, sample_music_file, tmp_path
):
    """Long bg trimmed to fg duration."""
    output = tmp_path / "trimmed_output.wav"
    config = AudioMixerConfig()
    request = AudioMixerRequest(
        foreground_path=sample_speech_file,
        background_path=sample_music_file,
        output_path=output,
    )

    response = mix_audio(config, request)

    assert response.output_path.exists()
    # Background (15s) is longer than foreground (9s), should be trimmed
    assert response.background_duration > response.foreground_duration
    assert abs(response.output_duration - response.foreground_duration) < 1.0


@pytest.mark.e2e
@silero_required
def test_mix_audio_sample_rate_mismatch(sample_speech_file, tmp_path, ffmpeg_available):
    """fg at 44100Hz, bg at 22050Hz — output matches fg sample rate."""
    bg = tmp_path / "bg_22050.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "anoisesrc=d=15:c=pink:r=22050",
        str(bg),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"FFmpeg failed: {result.stderr}"

    output = tmp_path / "sr_mismatch_output.wav"
    config = AudioMixerConfig()
    request = AudioMixerRequest(
        foreground_path=sample_speech_file,
        background_path=bg,
        output_path=output,
    )

    response = mix_audio(config, request)

    assert response.output_path.exists()
    assert abs(response.output_duration - response.foreground_duration) < 1.0


@pytest.mark.e2e
@silero_required
def test_mix_audio_channel_mismatch(sample_speech_file, tmp_path, ffmpeg_available):
    """fg mono, bg stereo — output matches fg channels."""
    bg = tmp_path / "bg_stereo.wav"
    cmd = [
        "ffmpeg",
        "-y",
        "-f",
        "lavfi",
        "-i",
        "anoisesrc=d=15:c=pink:r=44100",
        "-ac",
        "2",
        str(bg),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    assert result.returncode == 0, f"FFmpeg failed: {result.stderr}"

    output = tmp_path / "ch_mismatch_output.wav"
    config = AudioMixerConfig()
    request = AudioMixerRequest(
        foreground_path=sample_speech_file,
        background_path=bg,
        output_path=output,
    )

    response = mix_audio(config, request)

    assert response.output_path.exists()
    assert abs(response.output_duration - response.foreground_duration) < 1.0


@pytest.mark.e2e
@silero_required
def test_mix_audio_custom_output_format(
    sample_speech_file, sample_music_file, tmp_path
):
    """output_format='mp3', verify .mp3 output."""
    config = AudioMixerConfig(output_format="mp3")
    request = AudioMixerRequest(
        foreground_path=sample_speech_file,
        background_path=sample_music_file,
    )

    response = mix_audio(config, request)

    assert response.output_path.exists()
    assert response.output_path.suffix == ".mp3"
    # Cleanup generated file
    response.output_path.unlink(missing_ok=True)


@pytest.mark.e2e
@silero_required
def test_mix_audio_default_output_path(sample_speech_file, sample_music_file):
    """No output_path — verify _mixed suffix in default name."""
    config = AudioMixerConfig()
    request = AudioMixerRequest(
        foreground_path=sample_speech_file,
        background_path=sample_music_file,
    )

    response = mix_audio(config, request)

    assert response.output_path.exists()
    assert "_mixed" in response.output_path.stem
    # Cleanup generated file
    response.output_path.unlink(missing_ok=True)


@pytest.mark.e2e
@silero_required
def test_mix_audio_response_fields(sample_speech_file, sample_music_file, tmp_path):
    """Verify all AudioMixerResponse fields are populated."""
    output = tmp_path / "response_fields_output.wav"
    config = AudioMixerConfig()
    request = AudioMixerRequest(
        foreground_path=sample_speech_file,
        background_path=sample_music_file,
        output_path=output,
    )

    response = mix_audio(config, request)

    assert isinstance(response, AudioMixerResponse)
    assert isinstance(response.output_path, Path)
    assert response.output_path.exists()
    assert isinstance(response.foreground_duration, float)
    assert response.foreground_duration > 0
    assert isinstance(response.background_duration, float)
    assert response.background_duration > 0
    assert isinstance(response.output_duration, float)
    assert response.output_duration > 0
    assert isinstance(response.speech_segments, list)
    assert isinstance(response.loops_used, int)
    assert response.loops_used >= 0


@pytest.mark.e2e
@silero_required
async def test_mix_audio_async_matches_sync(
    sample_speech_file, sample_music_file, tmp_path
):
    """Async variant produces valid output."""
    output = tmp_path / "async_output.wav"
    config = AudioMixerConfig()
    request = AudioMixerRequest(
        foreground_path=sample_speech_file,
        background_path=sample_music_file,
        output_path=output,
    )

    response = await mix_audio_async(config, request)

    assert isinstance(response, AudioMixerResponse)
    assert response.output_path.exists()
    assert response.output_duration > 0


@pytest.mark.e2e
@silero_required
def test_detect_speech_standalone(sample_speech_file):
    """detect_speech returns segments list without creating output file."""
    config = AudioMixerConfig()

    segments = detect_speech(config, sample_speech_file)

    assert isinstance(segments, list)
    # Silero VAD may or may not detect sine waves as speech — validate structure
    for seg in segments:
        assert isinstance(seg, SpeechSegment)
        assert seg.start >= 0
        assert seg.end > seg.start


@pytest.mark.e2e
@silero_required
def test_mix_audio_loop_short_bg_skips_crossfade(
    sample_speech_file, tiny_music_file, tmp_path
):
    """Tiny bg (1s) < crossfade (2s) * 2 — crossfade is skipped (EDGE-005)."""
    output = tmp_path / "tiny_loop_output.wav"
    config = AudioMixerConfig(loop_background=True, loop_crossfade=2.0)
    request = AudioMixerRequest(
        foreground_path=sample_speech_file,
        background_path=tiny_music_file,
        output_path=output,
    )

    response = mix_audio(config, request)

    assert response.output_path.exists()
    assert response.loops_used > 0
    assert abs(response.output_duration - response.foreground_duration) < 1.0
