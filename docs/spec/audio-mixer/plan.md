# Implementation Plan: tarash-audio-mixer

## Phase Table

| Phase | Name | REQ/EDGE IDs | Dependencies | Parallelizable With |
|-------|------|-------------|--------------|-------------------|
| 1 | Project scaffold + models + exceptions | REQ-022 | None | None |
| 2 | Speech detection (detector.py) | REQ-001, EDGE-001, EDGE-013 | Phase 1 | None |
| 3 | Envelope generation (volume filter builder) | REQ-002, REQ-003, REQ-004, REQ-005, REQ-006, REQ-007, EDGE-002, EDGE-008, EDGE-009, EDGE-014 | Phase 1 | Phase 2 |
| 4 | FFmpeg processor (background prep + mix) | REQ-008, REQ-009, REQ-010, REQ-011, REQ-012, REQ-013, REQ-014, REQ-015, REQ-016, REQ-017, EDGE-003, EDGE-004, EDGE-005, EDGE-006, EDGE-007, EDGE-010, EDGE-011, EDGE-012 | Phase 1 | Phases 2, 3 |
| 5 | Public API + logging + response | REQ-018, REQ-019, REQ-020, REQ-021 | Phases 2, 3, 4 | None |
| 6 | E2E tests + workspace integration | All | Phase 5 | None |

## Dependency Graph

```
Phase 1 (scaffold + models + exceptions)
  |
  +---> Phase 2 (detector)
  |        |
  +---> Phase 3 (envelope)  [can run parallel with Phase 2]
  |        |
  +---> Phase 4 (processor) [can run parallel with Phases 2, 3]
  |        |
  +--------+-------+
           |
           v
       Phase 5 (API + logging)
           |
           v
       Phase 6 (E2E + workspace)
```

## Package Structure

```
packages/tarash-audio-mixer/
├── pyproject.toml
├── src/tarash/tarash_audio_mixer/
│   ├── __init__.py
│   ├── api.py          # Public functions: mix_audio, mix_audio_async, detect_speech, detect_speech_async
│   ├── models.py       # AudioMixerConfig, AudioMixerRequest, AudioMixerResponse, SpeechSegment
│   ├── exceptions.py   # AudioMixerException, FFmpegNotFoundError, InvalidInputError, ProcessingError, DetectionError
│   ├── processor.py    # FFmpeg command building, background prep, mixing
│   ├── detector.py     # Silero VAD speech detection (single file, no fallback)
│   └── logging.py      # Logging utilities (same pattern as silence-remover)
└── tests/
    ├── conftest.py
    ├── unit/
    │   ├── test_models.py
    │   ├── test_exceptions.py
    │   ├── test_processor.py
    │   ├── test_detector.py
    │   ├── test_api.py
    │   └── test_logging.py
    └── e2e/
        └── test_audio_mixer.py
```

## Root Files to Modify

- `pyproject.toml` — add `tarash-audio-mixer[all]` to dependencies and `tool.uv.sources`

## Key Design Decisions

1. **Single `detector.py`** instead of `detectors/` package — spec explicitly excludes FFmpeg fallback for audio-mixer
2. **Multi-step FFmpeg via `filter_complex`** — build the filter graph string in Python (testable), execute as a single FFmpeg command (efficient)
3. **Envelope computed as keyframes in Python** — handles overlapping duck regions by taking `min()` of overlapping gains before generating FFmpeg expression
4. **No cross-package dependency on silence-remover** — own `SpeechSegment` model, own detector, own exceptions
