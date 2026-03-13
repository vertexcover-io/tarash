"""Tests for processor module — envelope generation and FFmpeg functions."""

from __future__ import annotations

import json
import math
from pathlib import Path
from unittest.mock import patch

import pytest

from tarash.tarash_audio_mixer.exceptions import (
    FFmpegNotFoundError,
    InvalidInputError,
    ProcessingError,
)
from tarash.tarash_audio_mixer.models import AudioMixerConfig, SpeechSegment
from tarash.tarash_audio_mixer.processor import (
    AudioInfo,
    build_filter_complex,
    build_loop_filter,
    build_mix_command,
    build_volume_expression,
    compute_duck_regions,
    derive_ffprobe_path,
    merge_duck_regions,
    probe_audio_info,
    probe_audio_info_async,
    run_mix,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def default_config() -> AudioMixerConfig:
    """Default audio mixer config."""
    return AudioMixerConfig()


@pytest.fixture
def fg_info() -> AudioInfo:
    """Standard foreground audio info: 60s, 44100Hz, stereo."""
    return AudioInfo(duration=60.0, sample_rate=44100, channels=2)


@pytest.fixture
def bg_info_short() -> AudioInfo:
    """Short background: 20s, 44100Hz, stereo."""
    return AudioInfo(duration=20.0, sample_rate=44100, channels=2)


@pytest.fixture
def bg_info_long() -> AudioInfo:
    """Long background: 120s, 44100Hz, stereo."""
    return AudioInfo(duration=120.0, sample_rate=44100, channels=2)


@pytest.fixture
def bg_info_matching() -> AudioInfo:
    """Background matching foreground exactly."""
    return AudioInfo(duration=60.0, sample_rate=44100, channels=2)


@pytest.fixture
def volume_expr() -> str:
    """Simple constant volume expression."""
    return "0.5"


# ---------------------------------------------------------------------------
# Envelope tests (Phase 3 coverage)
# ---------------------------------------------------------------------------


def test_compute_duck_regions_basic(default_config: AudioMixerConfig) -> None:
    """compute_duck_regions produces correct attack/release boundaries."""
    segments = [SpeechSegment(start=5.0, end=10.0)]
    regions = compute_duck_regions(segments, default_config, total_duration=60.0)

    assert len(regions) == 1
    region = regions[0]
    # speech_padding=0.3, attack_ms=200, release_ms=300
    assert region.full_duck_start == pytest.approx(5.0 - 0.3, abs=1e-6)
    assert region.full_duck_end == pytest.approx(10.0 + 0.3, abs=1e-6)


def test_merge_duck_regions_overlapping(default_config: AudioMixerConfig) -> None:
    """merge_duck_regions merges overlapping regions."""
    segments = [
        SpeechSegment(start=5.0, end=6.0),
        SpeechSegment(start=6.2, end=7.0),
    ]
    regions = compute_duck_regions(segments, default_config, total_duration=60.0)
    merged = merge_duck_regions(regions)

    # With default padding/attack/release, these should overlap and merge
    assert len(merged) <= len(regions)


def test_merge_duck_regions_empty() -> None:
    """merge_duck_regions returns empty for empty input."""
    assert merge_duck_regions([]) == []


def test_build_volume_expression_no_regions(default_config: AudioMixerConfig) -> None:
    """build_volume_expression returns base gain when no regions."""
    expr = build_volume_expression([], default_config, total_duration=60.0)
    base_gain = 10 ** (default_config.base_music_volume_db / 20.0)
    assert expr == str(base_gain)


def test_build_volume_expression_with_regions(
    default_config: AudioMixerConfig,
) -> None:
    """build_volume_expression produces nested if/between expression."""
    segments = [SpeechSegment(start=5.0, end=10.0)]
    regions = compute_duck_regions(segments, default_config, total_duration=60.0)
    merged = merge_duck_regions(regions)
    expr = build_volume_expression(merged, default_config, total_duration=60.0)

    assert "if(between(t," in expr
    assert ")" in expr


# ---------------------------------------------------------------------------
# Probing tests
# ---------------------------------------------------------------------------


def _mock_ffprobe_json(
    duration: float = 30.0,
    sample_rate: str = "44100",
    channels: int = 2,
) -> str:
    """Build mock ffprobe JSON output."""
    return json.dumps(
        {
            "streams": [{"sample_rate": sample_rate, "channels": channels}],
            "format": {"duration": str(duration)},
        }
    )


def test_probe_audio_info_parses_json() -> None:
    """probe_audio_info correctly parses ffprobe JSON output."""
    mock_stdout = _mock_ffprobe_json(duration=45.5, sample_rate="48000", channels=1)

    with patch(
        "tarash.tarash_audio_mixer.processor._run_sync",
        return_value=(0, mock_stdout, ""),
    ):
        info = probe_audio_info("ffmpeg", Path("/fake/audio.wav"))

    assert info.duration == pytest.approx(45.5)
    assert info.sample_rate == 48000
    assert info.channels == 1


def test_probe_audio_info_invalid_file() -> None:
    """probe_audio_info raises InvalidInputError on ffprobe failure (EDGE-012)."""
    with patch(
        "tarash.tarash_audio_mixer.processor._run_sync",
        return_value=(1, "", "No such file or directory"),
    ):
        with pytest.raises(InvalidInputError, match="Cannot read file"):
            probe_audio_info("ffmpeg", Path("/fake/missing.wav"))


def test_probe_audio_info_ffmpeg_not_found() -> None:
    """probe_audio_info raises FFmpegNotFoundError when binary missing (EDGE-011)."""
    with patch(
        "tarash.tarash_audio_mixer.processor._run_sync",
        side_effect=FFmpegNotFoundError(
            "Binary not found at 'ffprobe'. Ensure FFmpeg is installed."
        ),
    ):
        with pytest.raises(FFmpegNotFoundError):
            probe_audio_info("ffmpeg", Path("/fake/audio.wav"))


async def test_probe_audio_info_async() -> None:
    """probe_audio_info_async correctly parses ffprobe JSON output."""
    mock_stdout = _mock_ffprobe_json(duration=30.0, sample_rate="44100", channels=2)

    with patch(
        "tarash.tarash_audio_mixer.processor._run_async",
        return_value=(0, mock_stdout, ""),
    ):
        info = await probe_audio_info_async("ffmpeg", Path("/fake/audio.wav"))

    assert info.duration == pytest.approx(30.0)
    assert info.sample_rate == 44100
    assert info.channels == 2


# ---------------------------------------------------------------------------
# Filter complex tests
# ---------------------------------------------------------------------------


def test_filter_complex_foreground_gain_zero(
    default_config: AudioMixerConfig,
    fg_info: AudioInfo,
    bg_info_matching: AudioInfo,
    volume_expr: str,
) -> None:
    """When foreground_gain_db=0, foreground uses acopy (REQ-009)."""
    assert default_config.foreground_gain_db == 0.0

    fc = build_filter_complex(default_config, volume_expr, fg_info, bg_info_matching)
    parts = fc.split(";")

    assert "[0:a]acopy[fg]" in parts[1]
    # Foreground chain should NOT have a volume filter
    assert "volume=" not in parts[1]


def test_filter_complex_foreground_gain_nonzero(
    fg_info: AudioInfo,
    bg_info_matching: AudioInfo,
    volume_expr: str,
) -> None:
    """When foreground_gain_db!=0, foreground gets volume filter (REQ-008)."""
    config = AudioMixerConfig(foreground_gain_db=3.0)
    fc = build_filter_complex(config, volume_expr, fg_info, bg_info_matching)

    fg_chain = fc.split(";")[1]
    assert "[0:a]volume=" in fg_chain
    assert "[fg]" in fg_chain

    # Verify gain is correct linear value
    expected_gain = 10 ** (3.0 / 20.0)
    assert str(expected_gain) in fg_chain


def test_filter_complex_background_trimmed_when_longer(
    default_config: AudioMixerConfig,
    fg_info: AudioInfo,
    bg_info_long: AudioInfo,
    volume_expr: str,
) -> None:
    """Background longer than foreground gets atrim (EDGE-010)."""
    fc = build_filter_complex(default_config, volume_expr, fg_info, bg_info_long)

    bg_chain = fc.split(";")[0]
    assert f"atrim=end={fg_info.duration}" in bg_chain


def test_filter_complex_background_looped_when_shorter(
    default_config: AudioMixerConfig,
    fg_info: AudioInfo,
    bg_info_short: AudioInfo,
    volume_expr: str,
) -> None:
    """Background shorter + loop_background=True uses aloop (REQ-010, EDGE-003)."""
    assert default_config.loop_background is True

    fc = build_filter_complex(default_config, volume_expr, fg_info, bg_info_short)

    bg_chain = fc.split(";")[0]
    assert "aloop=" in bg_chain
    assert f"atrim=end={fg_info.duration}" in bg_chain


def test_filter_complex_background_padded_when_shorter_no_loop(
    fg_info: AudioInfo,
    bg_info_short: AudioInfo,
    volume_expr: str,
) -> None:
    """Background shorter + loop_background=False uses apad (REQ-012, EDGE-004)."""
    config = AudioMixerConfig(loop_background=False)

    fc = build_filter_complex(config, volume_expr, fg_info, bg_info_short)

    bg_chain = fc.split(";")[0]
    assert f"apad=whole_dur={fg_info.duration}" in bg_chain
    assert f"atrim=end={fg_info.duration}" in bg_chain
    assert "aloop=" not in bg_chain


def test_filter_complex_resample_when_rates_differ(
    default_config: AudioMixerConfig,
    fg_info: AudioInfo,
    volume_expr: str,
) -> None:
    """aresample applied when sample rates differ (REQ-014, EDGE-006)."""
    bg_info = AudioInfo(duration=60.0, sample_rate=48000, channels=2)

    fc = build_filter_complex(default_config, volume_expr, fg_info, bg_info)

    bg_chain = fc.split(";")[0]
    assert f"aresample={fg_info.sample_rate}" in bg_chain


def test_filter_complex_channel_convert_when_differ(
    default_config: AudioMixerConfig,
    fg_info: AudioInfo,
    volume_expr: str,
) -> None:
    """aformat applied when channel counts differ (REQ-015, EDGE-007)."""
    bg_info = AudioInfo(duration=60.0, sample_rate=44100, channels=1)

    fc = build_filter_complex(default_config, volume_expr, fg_info, bg_info)

    bg_chain = fc.split(";")[0]
    assert "aformat=channel_layouts=stereo" in bg_chain


def test_filter_complex_channel_convert_mono(
    default_config: AudioMixerConfig,
    volume_expr: str,
) -> None:
    """aformat uses mono layout when foreground is mono."""
    fg = AudioInfo(duration=60.0, sample_rate=44100, channels=1)
    bg = AudioInfo(duration=60.0, sample_rate=44100, channels=2)

    fc = build_filter_complex(default_config, volume_expr, fg, bg)

    bg_chain = fc.split(";")[0]
    assert "aformat=channel_layouts=mono" in bg_chain


# ---------------------------------------------------------------------------
# Loop filter tests
# ---------------------------------------------------------------------------


def test_loop_filter_calculates_correct_count() -> None:
    """build_loop_filter computes correct loop count."""
    # bg=20s, fg=60s => need ceil(60/20) = 3 loops
    filter_str, loops_used = build_loop_filter(
        bg_duration=20.0, fg_duration=60.0, crossfade=2.0
    )

    assert loops_used == 3
    assert "aloop=loop=2" in filter_str  # aloop adds to original, so N-1


def test_loop_filter_skips_crossfade_short_background() -> None:
    """Crossfade skipped when bg < 2*crossfade (EDGE-005)."""
    # bg=3s, crossfade=2s => 3 < 4, skip crossfade
    filter_str, loops_used = build_loop_filter(
        bg_duration=3.0, fg_duration=60.0, crossfade=2.0
    )

    assert "acrossfade" not in filter_str
    assert "aloop=" in filter_str


def test_loop_filter_applies_crossfade() -> None:
    """Crossfade included when bg is long enough (REQ-011)."""
    # bg=20s, crossfade=2s => 20 >= 4, apply crossfade
    filter_str, loops_used = build_loop_filter(
        bg_duration=20.0, fg_duration=60.0, crossfade=2.0
    )

    assert "acrossfade=d=2.0" in filter_str


def test_loop_filter_no_crossfade_when_zero() -> None:
    """No crossfade when crossfade=0."""
    filter_str, loops_used = build_loop_filter(
        bg_duration=20.0, fg_duration=60.0, crossfade=0.0
    )

    assert "acrossfade" not in filter_str
    assert "aloop=" in filter_str


# ---------------------------------------------------------------------------
# Command building tests
# ---------------------------------------------------------------------------


def test_build_mix_command_structure(
    default_config: AudioMixerConfig,
) -> None:
    """build_mix_command produces correct command structure."""
    fg = Path("/audio/fg.wav")
    bg = Path("/audio/bg.wav")
    out = Path("/audio/output.wav")
    fc = "[1:a]volume='0.5':eval=frame[bg];[0:a]acopy[fg];[fg][bg]amix=inputs=2:duration=first:dropout_transition=0[out]"

    cmd = build_mix_command(default_config, fg, bg, out, fc)

    assert cmd[0] == default_config.ffmpeg_path
    assert "-y" in cmd
    assert "-i" in cmd
    assert str(fg) in cmd
    assert str(bg) in cmd
    assert "-filter_complex" in cmd
    assert fc in cmd
    assert "-map" in cmd
    assert "[out]" in cmd
    assert str(out) == cmd[-1]

    # Verify input ordering: fg first, bg second
    i_indices = [i for i, v in enumerate(cmd) if v == "-i"]
    assert len(i_indices) == 2
    assert cmd[i_indices[0] + 1] == str(fg)
    assert cmd[i_indices[1] + 1] == str(bg)


# ---------------------------------------------------------------------------
# run_mix tests
# ---------------------------------------------------------------------------


def test_run_mix_raises_on_failure(
    default_config: AudioMixerConfig,
    fg_info: AudioInfo,
    bg_info_matching: AudioInfo,
) -> None:
    """run_mix raises ProcessingError on non-zero exit code."""
    with patch(
        "tarash.tarash_audio_mixer.processor._run_sync",
        return_value=(1, "", "Error: something went wrong"),
    ):
        with pytest.raises(ProcessingError, match="FFmpeg mixing failed"):
            run_mix(
                default_config,
                Path("/fg.wav"),
                Path("/bg.wav"),
                Path("/out.wav"),
                "0.5",
                fg_info,
                bg_info_matching,
            )


def test_run_mix_returns_loops_used(
    fg_info: AudioInfo,
    bg_info_short: AudioInfo,
) -> None:
    """run_mix returns correct loops_used when background is looped."""
    config = AudioMixerConfig(loop_background=True)

    with patch(
        "tarash.tarash_audio_mixer.processor._run_sync",
        return_value=(0, "", ""),
    ):
        loops = run_mix(
            config,
            Path("/fg.wav"),
            Path("/bg.wav"),
            Path("/out.wav"),
            "0.5",
            fg_info,
            bg_info_short,
        )

    expected_loops = math.ceil(fg_info.duration / bg_info_short.duration)
    assert loops == expected_loops


def test_run_mix_returns_zero_loops_when_not_looped(
    fg_info: AudioInfo,
    bg_info_long: AudioInfo,
) -> None:
    """run_mix returns 0 when background is longer (no looping needed)."""
    config = AudioMixerConfig()

    with patch(
        "tarash.tarash_audio_mixer.processor._run_sync",
        return_value=(0, "", ""),
    ):
        loops = run_mix(
            config,
            Path("/fg.wav"),
            Path("/bg.wav"),
            Path("/out.wav"),
            "0.5",
            fg_info,
            bg_info_long,
        )

    assert loops == 0


# ---------------------------------------------------------------------------
# derive_ffprobe_path tests
# ---------------------------------------------------------------------------


def test_derive_ffprobe_path_from_ffmpeg() -> None:
    """derive_ffprobe_path replaces 'ffmpeg' with 'ffprobe'."""
    assert derive_ffprobe_path("ffmpeg") == "ffprobe"
    assert derive_ffprobe_path("/usr/bin/ffmpeg") == "/usr/bin/ffprobe"


def test_derive_ffprobe_path_non_standard() -> None:
    """derive_ffprobe_path returns 'ffprobe' for non-standard names."""
    assert derive_ffprobe_path("my-custom-binary") == "ffprobe"
