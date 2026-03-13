# SPEC: tarash-audio-mixer

**Source:** Brainstorm conversation (2026-03-13)
**Generated:** 2026-03-13

## Requirements

| ID | Type | Requirement | Acceptance Criterion | Priority |
|----|------|-------------|---------------------|----------|
| REQ-001 | Event-driven | When `mix_audio` is called with foreground and background paths, the system shall detect speech segments in the foreground using Silero VAD | Returns a list of `SpeechSegment` objects with start/end times in seconds | Must |
| REQ-002 | Event-driven | When speech segments are detected, the system shall generate an FFmpeg volume filter expression with piecewise linear gain ducking for the background track | Filter expression contains `if`/`between` clauses for each speech segment with correct time boundaries | Must |
| REQ-003 | Ubiquitous | The system shall apply attack ramps (configurable via `attack_ms`) that linearly reduce background volume from base level to duck level before each speech segment | Volume at `duck_start` equals base level; volume at `duck_full_start` equals duck level; transition is linear | Must |
| REQ-004 | Ubiquitous | The system shall apply release ramps (configurable via `release_ms`) that linearly restore background volume from duck level to base level after each speech segment | Volume at `duck_full_end` equals duck level; volume at `duck_end` equals base level; transition is linear | Must |
| REQ-005 | Ubiquitous | The system shall pad speech segments by `speech_padding` seconds on each side when computing duck boundaries | Duck region starts `speech_padding` seconds before segment start and ends `speech_padding` seconds after segment end | Must |
| REQ-006 | Ubiquitous | The system shall apply `base_music_volume_db` to background audio in non-speech sections | Background volume in non-speech regions equals `base_music_volume_db` (default -6.0 dB) | Must |
| REQ-007 | Ubiquitous | The system shall apply `duck_level_db` relative to `base_music_volume_db` during speech sections | Background volume during full duck equals `base_music_volume_db + duck_level_db` | Must |
| REQ-008 | Event-driven | When `foreground_gain_db` is non-zero, the system shall apply the specified gain to the foreground audio | Foreground output level is adjusted by exactly `foreground_gain_db` dB | Must |
| REQ-009 | Event-driven | When `foreground_gain_db` is 0.0 (default), the system shall pass foreground audio through unchanged | Foreground audio is not processed through a volume filter | Must |
| REQ-010 | Event-driven | When `loop_background` is True and background is shorter than foreground, the system shall loop background audio to match foreground duration | Output duration matches foreground duration; background content repeats | Must |
| REQ-011 | Event-driven | When looping background audio, the system shall apply a crossfade at loop points to avoid audible jumps | Crossfade of `loop_crossfade` seconds (default 2.0s) is applied at each loop boundary | Must |
| REQ-012 | Event-driven | When `loop_background` is False and background is shorter than foreground, the system shall pad background with silence | Background track is silence-padded to match foreground duration | Must |
| REQ-013 | Ubiquitous | The system shall produce output with duration matching the foreground audio duration | `output_duration` equals `foreground_duration` within 0.1 second tolerance | Must |
| REQ-014 | Event-driven | When foreground and background have different sample rates, the system shall resample background to match foreground | Output sample rate matches foreground sample rate | Must |
| REQ-015 | Event-driven | When foreground and background have different channel layouts, the system shall convert background to match foreground | Output channel count matches foreground channel count | Must |
| REQ-016 | Event-driven | When `output_format` is specified, the system shall encode output in that format | Output file has the specified format/extension | Should |
| REQ-017 | Event-driven | When `output_format` is None (default), the system shall encode output in the same format as the foreground file | Output format matches foreground file format | Must |
| REQ-018 | Event-driven | When `output_path` is None, the system shall generate a default output path based on the foreground filename | Output path is `foreground_stem_mixed.ext` in the same directory | Must |
| REQ-019 | Ubiquitous | The system shall return an `AudioMixerResponse` containing output_path, durations, speech_segments, and loops_used | All response fields are populated with correct values | Must |
| REQ-020 | Event-driven | When `mix_audio_async` is called, the system shall perform the same operations asynchronously | Async variant produces identical output to sync variant | Must |
| REQ-021 | Ubiquitous | The system shall expose `detect_speech` and `detect_speech_async` functions for speech detection only (without mixing) | Functions return `list[SpeechSegment]` without producing output audio | Should |
| REQ-022 | Ubiquitous | All Pydantic models (Config, Request, Response, SpeechSegment) shall be frozen (immutable) | `model.field = value` raises a `ValidationError` | Must |

## Edge Cases

| ID | Scenario | Expected Behavior | Derived From |
|----|----------|-------------------|-------------|
| EDGE-001 | No speech detected in foreground | Background plays at `base_music_volume_db` throughout, foreground passed through, `speech_segments` is empty list | REQ-001, REQ-006 |
| EDGE-002 | Adjacent speech segments with overlapping duck regions | `min()` of envelope values applied — duck regions merge seamlessly without volume spikes between them | REQ-002, REQ-003, REQ-004 |
| EDGE-003 | Background shorter than foreground with `loop_background=True` | Background loops with crossfade, output duration matches foreground | REQ-010, REQ-011, REQ-013 |
| EDGE-004 | Background shorter than foreground with `loop_background=False` | Background padded with silence, output duration matches foreground | REQ-012, REQ-013 |
| EDGE-005 | Background shorter than crossfade duration (e.g., 1s background, 2s crossfade) | Crossfade skipped, simple loop used instead | REQ-011 |
| EDGE-006 | Foreground and background have different sample rates | Background resampled to match foreground automatically | REQ-014 |
| EDGE-007 | Foreground mono, background stereo (or vice versa) | Background channel layout converted to match foreground | REQ-015 |
| EDGE-008 | Speech segment at very start of foreground (t=0) | Attack ramp truncated to available time (no negative timestamps) | REQ-003, REQ-005 |
| EDGE-009 | Speech segment at very end of foreground | Release ramp truncated to foreground duration (no overshoot) | REQ-004, REQ-005 |
| EDGE-010 | Background longer than foreground | Background trimmed to foreground duration, no looping needed | REQ-013 |
| EDGE-011 | FFmpeg binary not found | `FFmpegNotFoundError` raised with clear message | REQ-002 |
| EDGE-012 | Invalid input file (corrupted or unsupported format) | `InvalidInputError` raised with file path and reason | REQ-001 |
| EDGE-013 | Silero VAD not installed | `DetectionError` raised (no fallback detector in audio-mixer) | REQ-001 |
| EDGE-014 | `attack_ms` or `release_ms` set to 0 | Instant volume change (step function), no ramp | REQ-003, REQ-004 |

## Verification Matrix

| REQ/EDGE ID | Unit Test | E2E Test | Notes |
|-------------|-----------|----------|-------|
| REQ-001 | Yes | Yes | Mock Silero in unit, real Silero in E2E |
| REQ-002 | Yes | No | Test filter expression generation with known segments |
| REQ-003 | Yes | No | Verify envelope values at attack boundaries |
| REQ-004 | Yes | No | Verify envelope values at release boundaries |
| REQ-005 | Yes | No | Verify duck boundaries include padding offset |
| REQ-006 | Yes | Yes | Verify base volume in non-speech sections |
| REQ-007 | Yes | No | Verify duck volume = base + duck_level |
| REQ-008 | Yes | Yes | Non-zero foreground gain applied |
| REQ-009 | Yes | No | Zero gain means no foreground volume filter |
| REQ-010 | Yes | Yes | Verify looping when background < foreground |
| REQ-011 | Yes | Yes | Verify crossfade at loop points |
| REQ-012 | Yes | Yes | Verify silence padding when no loop |
| REQ-013 | Yes | Yes | Output duration matches foreground |
| REQ-014 | Yes | Yes | Sample rate matching |
| REQ-015 | Yes | Yes | Channel layout matching |
| REQ-016 | Yes | Yes | Custom output format |
| REQ-017 | Yes | No | Default format matches foreground |
| REQ-018 | Yes | No | Default output path generation |
| REQ-019 | Yes | Yes | Response fields populated correctly |
| REQ-020 | Yes | Yes | Async produces same result as sync |
| REQ-021 | Yes | No | Detection-only functions |
| REQ-022 | Yes | No | Model immutability |
| EDGE-001 | Yes | Yes | No speech → base volume throughout |
| EDGE-002 | Yes | No | Overlapping duck regions merge |
| EDGE-003 | Yes | Yes | Loop with crossfade |
| EDGE-004 | Yes | Yes | No loop, silence pad |
| EDGE-005 | Yes | No | Short background skips crossfade |
| EDGE-006 | Yes | Yes | Sample rate mismatch |
| EDGE-007 | Yes | Yes | Channel mismatch |
| EDGE-008 | Yes | No | Speech at t=0, truncated attack |
| EDGE-009 | Yes | No | Speech at end, truncated release |
| EDGE-010 | Yes | Yes | Background longer, trimmed |
| EDGE-011 | Yes | No | FFmpeg not found error |
| EDGE-012 | Yes | No | Invalid input error |
| EDGE-013 | Yes | No | Silero not installed error |
| EDGE-014 | Yes | No | Zero attack/release = step function |

## Out of Scope

- Real-time / streaming audio mixing (batch processing only)
- Video file support (audio-only mixing)
- Multiple foreground or background tracks (single pair only)
- Noise reduction or audio enhancement on foreground
- Automatic gain normalization / loudness targeting (LUFS)
- FFmpeg silencedetect fallback detector (Silero VAD only)
- GUI or interactive preview
