# Phase 5: Public API + Logging + Response

## Covers
- REQ-018 (default output path generation)
- REQ-019 (AudioMixerResponse with all fields)
- REQ-020 (async variant)
- REQ-021 (detect_speech / detect_speech_async standalone)

## Dependencies
- Phase 2 (detector)
- Phase 3 (envelope generation)
- Phase 4 (processor / FFmpeg execution)

## Files to Modify

### `packages/tarash-audio-mixer/src/tarash/tarash_audio_mixer/api.py`

Public API functions — orchestrate detection, envelope building, and mixing.

#### Functions:

- `_resolve_output_path(fg_path: Path, output_path: Path | None, output_format: str | None) -> Path`:
  - If output_path is provided, return it
  - If output_format is provided: `fg_path.parent / f"{fg_path.stem}_mixed.{output_format}"` (REQ-016)
  - Otherwise: `fg_path.parent / f"{fg_path.stem}_mixed{fg_path.suffix}"` (REQ-017, REQ-018)

- `detect_speech(config: AudioMixerConfig, foreground_path: Path) -> list[SpeechSegment]` (REQ-021):
  - Validate foreground_path exists
  - Call `detector.detect_speech_segments(foreground_path, config)`
  - Return segments (no mixing, no output file)

- `detect_speech_async(config: AudioMixerConfig, foreground_path: Path) -> list[SpeechSegment]` (REQ-021):
  - Async variant of above

- `mix_audio(config: AudioMixerConfig, request: AudioMixerRequest) -> AudioMixerResponse`:
  1. Validate foreground_path and background_path exist, raise `InvalidInputError` if not
  2. Resolve output_path via `_resolve_output_path`
  3. Probe both files: `probe_audio_info` for fg and bg
  4. Detect speech: `detector.detect_speech_segments(fg_path, config)`
  5. Compute duck regions: `compute_duck_regions(segments, config, fg_duration)`
  6. Merge regions: `merge_duck_regions(regions)`
  7. Build volume expression: `build_volume_expression(merged_regions, config, fg_duration)`
  8. Run mix: `run_mix(config, fg_path, bg_path, output_path, volume_expr, fg_info, bg_info)`
  9. Probe output duration
  10. Log completion
  11. Return `AudioMixerResponse` with all fields

- `mix_audio_async(config: AudioMixerConfig, request: AudioMixerRequest) -> AudioMixerResponse` (REQ-020):
  - Same flow as sync but using async variants of all I/O operations

### `packages/tarash-audio-mixer/src/tarash/tarash_audio_mixer/__init__.py`

Update to export all public API functions (should already be set up from Phase 1, but verify exports match).

## Tests to Write

### `packages/tarash-audio-mixer/tests/unit/test_api.py`

Mock detector and processor — test orchestration logic only.

- `test_resolve_output_path_explicit` — provided output_path returned as-is
- `test_resolve_output_path_default` — generates `{stem}_mixed.{ext}` (REQ-018)
- `test_resolve_output_path_custom_format` — uses output_format extension (REQ-016)
- `test_resolve_output_path_default_format_matches_foreground` — suffix matches fg (REQ-017)
- `test_mix_audio_validates_foreground_exists` — non-existent fg raises InvalidInputError
- `test_mix_audio_validates_background_exists` — non-existent bg raises InvalidInputError
- `test_mix_audio_returns_response` — mock all internals, verify AudioMixerResponse fields (REQ-019)
- `test_mix_audio_response_has_speech_segments` — segments from detector in response
- `test_mix_audio_response_has_loops_used` — loops_used from processor in response
- `test_mix_audio_async_produces_same_result` — async variant returns equivalent response (REQ-020)
- `test_detect_speech_returns_segments` — standalone detection returns list[SpeechSegment] (REQ-021)
- `test_detect_speech_validates_path` — non-existent path raises InvalidInputError
- `test_detect_speech_async_returns_segments` — async detection works (REQ-021)

## Verification
- `uv run pytest tests/unit/test_api.py -v`
- All API orchestration tests pass
- `uv run pytest tests/unit/ -v` — full unit test suite passes
