# Phase 2: Speech Detection (detector.py)

## Covers
- REQ-001 (detect speech segments using Silero VAD)
- EDGE-001 (no speech detected)
- EDGE-013 (Silero not installed)

## Dependencies
- Phase 1 (models, exceptions)

## Files to Create

### `packages/tarash-audio-mixer/src/tarash/tarash_audio_mixer/detector.py`

Single-file detector (no `detectors/` package — spec excludes FFmpeg fallback).

- `_silero_available() -> bool` — check if silero-vad is importable
- `_SILERO_SAMPLE_RATE = 16000`
- `_LOGGER_NAME = "tarash.tarash_audio_mixer.detector"`

- `detect_speech_segments(audio_path: Path, config: AudioMixerConfig) -> list[SpeechSegment]`:
  - Check `_silero_available()`, raise `DetectionError` if not installed (no fallback)
  - Load Silero VAD model
  - Read audio at 16kHz
  - Run `get_speech_timestamps` with `config.vad_threshold`
  - Convert timestamps to `SpeechSegment` list
  - Return empty list if no speech (EDGE-001)
  - Raise `DetectionError` on any Silero failure

- `detect_speech_segments_async(audio_path: Path, config: AudioMixerConfig) -> list[SpeechSegment]`:
  - Same checks as sync
  - Use `asyncio.to_thread` for CPU-bound Silero inference

Key difference from silence-remover: **no fallback to FFmpeg detector**. If Silero is not installed, raise `DetectionError` with install hint.

## Tests to Write

### `packages/tarash-audio-mixer/tests/unit/test_detector.py`

All tests mock Silero imports — no real VAD model needed for unit tests.

- `test_detect_returns_speech_segments` — mock silero_vad imports, verify SpeechSegment list returned
- `test_detect_no_speech_returns_empty_list` — mock returns empty timestamps, verify empty list
- `test_detect_raises_when_silero_unavailable` — mock `_silero_available` to return False, verify `DetectionError` raised with install hint message
- `test_detect_raises_on_model_load_failure` — mock model loading to raise, verify `DetectionError`
- `test_detect_raises_on_processing_failure` — mock `get_speech_timestamps` to raise, verify `DetectionError`
- `test_detect_async_delegates_to_thread` — mock `asyncio.to_thread`, verify it's called with sync detection
- `test_detect_async_raises_when_silero_unavailable` — async version also raises `DetectionError`
- `test_detect_uses_config_vad_threshold` — verify threshold from config is passed to `get_speech_timestamps`

## Verification
- `uv run pytest tests/unit/test_detector.py -v`
- All tests pass, no real Silero dependency needed for unit tests
