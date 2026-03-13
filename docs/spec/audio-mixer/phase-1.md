# Phase 1: Project Scaffold + Models + Exceptions

## Covers
- REQ-022 (frozen models)

## Dependencies
- None

## Files to Create

### `packages/tarash-audio-mixer/pyproject.toml`
- Mirror silence-remover's pyproject.toml structure
- Name: `tarash-audio-mixer`
- Dependencies: `pydantic>=2.0.0`
- Optional deps: `silero = ["silero-vad>=6.0.0", "torchcodec>=0.1.0"]`, `all = ["tarash-audio-mixer[silero]"]`
- Dev deps: pytest, pytest-asyncio, pytest-cov
- Build system: hatchling + uv-dynamic-versioning
- Wheel packages: `["src/tarash"]`
- pytest config: asyncio_mode = "auto", markers for unit/e2e, `--e2e` flag

### `packages/tarash-audio-mixer/src/tarash/tarash_audio_mixer/__init__.py`
- Version from importlib.metadata
- Re-export all public API functions, models, and exceptions
- `__all__` list

### `packages/tarash-audio-mixer/src/tarash/tarash_audio_mixer/models.py`
- `AudioMixerConfig` (frozen BaseModel):
  - `duck_level_db: float` (default -12.0, le=0) — dB reduction relative to base during speech
  - `attack_ms: float` (default 200.0, ge=0) — attack ramp in ms
  - `release_ms: float` (default 300.0, ge=0) — release ramp in ms
  - `speech_padding: float` (default 0.3, ge=0) — seconds of padding around speech for duck regions
  - `base_music_volume_db: float` (default -6.0) — background volume in non-speech sections
  - `foreground_gain_db: float` (default 0.0) — gain applied to foreground
  - `loop_background: bool` (default True) — loop background if shorter than foreground
  - `loop_crossfade: float` (default 2.0, ge=0) — crossfade seconds at loop boundaries
  - `vad_threshold: float` (default 0.5, ge=0, le=1) — Silero VAD threshold
  - `output_format: str | None` (default None) — output container format
  - `ffmpeg_path: str` (default "ffmpeg")
  - `device: str | None` (default None) — torch device for Silero

- `AudioMixerRequest` (BaseModel):
  - `foreground_path: Path` — path to speech audio
  - `background_path: Path` — path to music audio
  - `output_path: Path | None` (default None)

- `SpeechSegment` (frozen BaseModel):
  - `start: float`
  - `end: float`

- `AudioMixerResponse` (frozen BaseModel):
  - `output_path: Path`
  - `foreground_duration: float`
  - `background_duration: float`
  - `output_duration: float`
  - `speech_segments: list[SpeechSegment]`
  - `loops_used: int` — how many times background was looped (0 if not looped)

### `packages/tarash-audio-mixer/src/tarash/tarash_audio_mixer/exceptions.py`
- `AudioMixerException(Exception)` — base, with `message: str` attribute
- `FFmpegNotFoundError(AudioMixerException)`
- `InvalidInputError(AudioMixerException)`
- `ProcessingError(AudioMixerException)`
- `DetectionError(AudioMixerException)`

### `packages/tarash-audio-mixer/src/tarash/tarash_audio_mixer/logging.py`
- Copy pattern from silence-remover's logging.py
- Change default logger name to `"tarash.tarash_audio_mixer"`
- Functions: `log_debug`, `log_info`, `log_warning`, `log_error`
- Include `_redact_context` and `_redact_value` helpers

## Tests to Write

### `packages/tarash-audio-mixer/tests/conftest.py`
- `pytest_addoption` for `--e2e` flag
- `pytest_collection_modifyitems` to skip e2e tests without flag
- `ffmpeg_available` session fixture
- `make_async_proc` helper

### `packages/tarash-audio-mixer/tests/unit/test_models.py`
- `test_config_defaults` — verify all default values
- `test_config_custom_values` — verify custom values accepted
- `test_config_is_frozen` — mutating raises ValidationError
- `test_config_duck_level_must_be_nonpositive` — positive duck_level_db rejected
- `test_config_negative_attack_rejected` — negative attack_ms rejected
- `test_config_negative_release_rejected` — negative release_ms rejected
- `test_config_negative_padding_rejected` — negative speech_padding rejected
- `test_config_vad_threshold_out_of_range` — >1 or <0 rejected
- `test_request_fields` — foreground_path, background_path set correctly
- `test_request_output_path_defaults_to_none`
- `test_speech_segment_creation` — start/end set correctly
- `test_speech_segment_is_frozen` — mutating raises ValidationError
- `test_response_fields` — all fields populated correctly
- `test_response_is_frozen` — mutating raises ValidationError

### `packages/tarash-audio-mixer/tests/unit/test_exceptions.py`
- `test_base_exception_message` — message attribute set
- `test_ffmpeg_not_found_inherits_base`
- `test_invalid_input_inherits_base`
- `test_processing_error_inherits_base`
- `test_detection_error_inherits_base`
- `test_exception_str_matches_message`

### `packages/tarash-audio-mixer/tests/unit/test_logging.py`
- `test_log_info_with_context` — verify logger called with formatted message
- `test_log_info_without_context`
- `test_log_error_with_exc_info`
- `test_redact_sensitive_fields` — api_key, password, token redacted
- `test_redact_bytes_shows_length`
- `test_redact_long_strings_truncated`

## Verification
- `uv run pytest tests/unit/test_models.py tests/unit/test_exceptions.py tests/unit/test_logging.py -v`
- All tests pass, all models are frozen, all exceptions have correct hierarchy
