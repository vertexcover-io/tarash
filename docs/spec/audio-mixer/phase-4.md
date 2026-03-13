# Phase 4: FFmpeg Processor (Background Prep + Mix)

## Covers
- REQ-008 (foreground gain applied when non-zero)
- REQ-009 (foreground passed through when gain is 0)
- REQ-010 (loop background when shorter)
- REQ-011 (crossfade at loop boundaries)
- REQ-012 (pad with silence when no loop)
- REQ-013 (output duration matches foreground)
- REQ-014 (resample background to match foreground)
- REQ-015 (channel layout matching)
- REQ-016 (custom output format)
- REQ-017 (default format matches foreground)
- EDGE-003 (loop with crossfade)
- EDGE-004 (silence pad when no loop)
- EDGE-005 (short background skips crossfade)
- EDGE-006 (sample rate mismatch)
- EDGE-007 (channel mismatch)
- EDGE-010 (background longer, trimmed)
- EDGE-011 (FFmpeg not found)
- EDGE-012 (invalid input file)

## Dependencies
- Phase 1 (models, exceptions)

## Files to Modify

### `packages/tarash-audio-mixer/src/tarash/tarash_audio_mixer/processor.py`

Add FFmpeg command building and execution functions alongside the envelope functions from Phase 3.

#### Probing functions:
- `derive_ffprobe_path(ffmpeg_path: str) -> str` — same pattern as silence-remover
- `probe_audio_info(ffmpeg_path: str, file_path: Path) -> AudioInfo` — probe duration, sample_rate, channels using ffprobe JSON output
- `probe_audio_info_async(ffmpeg_path: str, file_path: Path) -> AudioInfo`
- `AudioInfo` (internal frozen model or NamedTuple): duration, sample_rate, channels

#### Subprocess runners:
- `_run_sync(cmd: list[str]) -> tuple[int, str, str]` — same pattern as silence-remover, raises `FFmpegNotFoundError` on FileNotFoundError
- `_run_async(cmd: list[str]) -> tuple[int, str, str]` — async variant

#### Filter complex builder:
- `build_filter_complex(config: AudioMixerConfig, volume_expr: str, fg_info: AudioInfo, bg_info: AudioInfo) -> str`:
  - Input [0] = foreground, Input [1] = background
  - Background chain:
    - If bg shorter than fg and `loop_background=True`: use `aloop` or manual loop logic
    - If bg shorter than fg and `loop_background=False`: use `apad` + `atrim` for silence padding
    - If bg longer than fg: `atrim=end=<fg_duration>`
    - Resample if sample rates differ: `aresample=<fg_sample_rate>` (REQ-014)
    - Channel convert if different: `aformat=channel_layouts=<fg_layout>` (REQ-015)
    - Apply volume expression: `volume='<volume_expr>':eval=frame`
  - Foreground chain:
    - If `foreground_gain_db != 0`: `volume=<gain_linear>` (REQ-008)
    - If `foreground_gain_db == 0`: pass through (REQ-009)
  - Mix: `amix=inputs=2:duration=first:dropout_transition=0`
  - Output trimmed to foreground duration (REQ-013)

#### Background looping:
- `build_loop_filter(bg_duration: float, fg_duration: float, crossfade: float) -> tuple[str, int]`:
  - Calculate loops_needed = ceil(fg_duration / bg_duration)
  - If bg_duration < crossfade * 2: skip crossfade, use simple `aloop` (EDGE-005)
  - Return (filter string fragment, loops_used count)

#### Mix command builder:
- `build_mix_command(config: AudioMixerConfig, fg_path: Path, bg_path: Path, output_path: Path, filter_complex: str) -> list[str]`:
  - Build full FFmpeg command: `-i fg -i bg -filter_complex "..." -y output`

#### Execution:
- `run_mix(config: AudioMixerConfig, fg_path: Path, bg_path: Path, output_path: Path, volume_expr: str, fg_info: AudioInfo, bg_info: AudioInfo) -> int`:
  - Build filter_complex, build command, run sync, raise ProcessingError on failure
  - Return loops_used
- `run_mix_async(...)` — async variant

## Tests to Write

### `packages/tarash-audio-mixer/tests/unit/test_processor.py` (FFmpeg-related tests)

All tests mock subprocess — no real FFmpeg calls.

- **Probing:**
  - `test_probe_audio_info_parses_json` — mock ffprobe output with sample_rate, channels, duration
  - `test_probe_audio_info_invalid_file` — mock ffprobe failure, verify `InvalidInputError` (EDGE-012)
  - `test_probe_audio_info_ffmpeg_not_found` — mock FileNotFoundError, verify `FFmpegNotFoundError` (EDGE-011)
  - `test_probe_audio_info_async` — verify async variant works

- **Filter complex building:**
  - `test_filter_complex_foreground_gain_zero` — no volume filter on foreground chain (REQ-009)
  - `test_filter_complex_foreground_gain_nonzero` — volume filter applied (REQ-008)
  - `test_filter_complex_background_trimmed_when_longer` — atrim applied (EDGE-010)
  - `test_filter_complex_background_looped_when_shorter` — aloop applied when loop_background=True (REQ-010, EDGE-003)
  - `test_filter_complex_background_padded_when_shorter_no_loop` — apad applied when loop_background=False (REQ-012, EDGE-004)
  - `test_filter_complex_resample_when_rates_differ` — aresample in chain (REQ-014, EDGE-006)
  - `test_filter_complex_channel_convert_when_differ` — aformat in chain (REQ-015, EDGE-007)
  - `test_filter_complex_output_format` — verify output path uses correct extension (REQ-016, REQ-017)

- **Looping:**
  - `test_loop_filter_calculates_correct_count` — loops_needed computed correctly
  - `test_loop_filter_skips_crossfade_short_background` — bg < 2*crossfade skips crossfade (EDGE-005)
  - `test_loop_filter_applies_crossfade` — crossfade included when bg is long enough (REQ-011)

- **Command building:**
  - `test_build_mix_command_structure` — verify command has correct flags and ordering
  - `test_run_mix_raises_on_failure` — non-zero exit raises ProcessingError

## Verification
- `uv run pytest tests/unit/test_processor.py -v`
- All processor tests pass (both envelope and FFmpeg tests)
