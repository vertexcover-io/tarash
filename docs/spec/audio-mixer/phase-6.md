# Phase 6: E2E Tests + Workspace Integration

## Covers
- All REQ/EDGE IDs (integration verification)
- Workspace configuration in root pyproject.toml

## Dependencies
- Phase 5 (all code complete)

## Files to Modify

### Root `pyproject.toml`

Add workspace source and dependency:

```toml
# In [project] dependencies, add:
"tarash-audio-mixer[all]",

# In [tool.uv.sources], add:
tarash-audio-mixer = { workspace = true }
```

### `packages/tarash-audio-mixer/tests/e2e/test_audio_mixer.py`

E2E tests require FFmpeg and Silero VAD installed. All marked with `@pytest.mark.e2e`.

#### Test fixtures:
- `sample_speech_file` — generate a short WAV with speech-like content (sine wave bursts with silence gaps) using FFmpeg
- `sample_music_file` — generate a continuous tone/noise WAV using FFmpeg
- `short_music_file` — generate a very short music file (shorter than speech) for loop testing
- `tiny_music_file` — generate a music file shorter than crossfade duration for EDGE-005

#### Tests:

- `test_mix_audio_basic` — mix speech + music, verify output exists, duration matches foreground (REQ-013, REQ-019)
- `test_mix_audio_no_speech` — foreground with no speech, background at base volume throughout (EDGE-001)
- `test_mix_audio_foreground_gain` — apply foreground gain, verify it's different from no-gain output (REQ-008)
- `test_mix_audio_foreground_gain_zero_passthrough` — gain=0 produces same foreground level (REQ-009 — not easily testable in E2E, verify output exists)
- `test_mix_audio_loop_background` — short bg + long fg with loop=True, verify output duration (REQ-010, EDGE-003)
- `test_mix_audio_loop_with_crossfade` — verify looped output exists and has correct duration (REQ-011)
- `test_mix_audio_no_loop_silence_pad` — loop=False, short bg padded with silence (REQ-012, EDGE-004)
- `test_mix_audio_background_longer_trimmed` — long bg trimmed to fg duration (EDGE-010)
- `test_mix_audio_sample_rate_mismatch` — fg 44100Hz, bg 22050Hz, output matches fg rate (REQ-014, EDGE-006)
- `test_mix_audio_channel_mismatch` — fg mono, bg stereo, output matches fg channels (REQ-015, EDGE-007)
- `test_mix_audio_custom_output_format` — output_format="mp3", verify .mp3 output (REQ-016)
- `test_mix_audio_default_output_path` — no output_path, verify _mixed suffix (REQ-018)
- `test_mix_audio_response_fields` — verify all AudioMixerResponse fields populated (REQ-019)
- `test_mix_audio_async_matches_sync` — async produces valid output (REQ-020)
- `test_detect_speech_standalone` — detect_speech returns segments without output file (REQ-021)
- `test_mix_audio_loop_short_bg_skips_crossfade` — tiny bg, crossfade skipped (EDGE-005)

## Verification
- `uv run pytest tests/unit/ -v` — all unit tests pass
- `uv run pytest tests/ --e2e -v` — all E2E tests pass (requires FFmpeg + Silero)
- `uv sync` from root — workspace resolves correctly
