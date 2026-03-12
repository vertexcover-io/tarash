# Plan: Preview Mode (Dry-Run)

> **Source:** `dev-docs/plans/2026-03-12-preview-mode.md`
> **Created:** 2026-03-12
> **Status:** planning

## Goal

Add `preview_silence` / `preview_silence_async` functions that run the detection + padding + merging pipeline but stop before FFmpeg processing, returning a `SilenceRemovalPreview` model with estimated metrics.

## Acceptance Criteria

- [ ] `SilenceRemovalPreview` model exists with all fields and computed properties from spec
- [ ] `preview_silence(config, input_path)` returns a preview without running FFmpeg processing
- [ ] `preview_silence_async(config, input_path)` provides the same functionality asynchronously
- [ ] Both functions reuse existing pipeline steps (probe, detect, pad, merge)
- [ ] Duration estimation logic correctly computes speech + inserted silence gaps
- [ ] All new symbols exported from `__init__.py`
- [ ] Unit tests cover: normal case, edge cases (no segments, zero duration, single segment, all silence)
- [ ] E2E test: preview then remove, compare estimate vs actual duration (error < 0.5s)

## Codebase Context

### Existing Patterns to Follow
- **`detect_silence` / `detect_silence_async`**: `api.py` lines 62-95 -- same `(config, input_path)` signature, same detector delegation pattern. Preview functions follow this exact structure but add probe + pad + merge steps.
- **`remove_silence` / `remove_silence_async`**: `api.py` lines 98-249 -- preview reuses steps 1-4 (validate, probe, detect, pad+merge) and replaces step 5 (process_segments) with duration estimation.
- **`SilenceRemovalResponse`**: `models.py` lines 113-129 -- preview model follows same style (frozen Pydantic model, computed properties).
- **Processor helpers**: `apply_padding`, `merge_overlapping_segments`, `probe_media_info`, `probe_media_info_async` from `processor.py`.

### Test Infrastructure
- Runner: `uv run pytest packages/tarash-silence-remover/tests/`
- Unit tests: `tests/unit/test_api.py` -- function-based, mock detector/processor with `unittest.mock.patch`
- E2E tests: `tests/e2e/test_silence_removal.py` -- require `--e2e` flag, use `ffmpeg_available` fixture
- Fixtures: `conftest.py` has `ffmpeg_available`, `make_async_proc`; e2e has `audio_with_silence`, `video_with_silence`

## Phases

| # | Phase | Status | Depends On |
|---|-------|--------|------------|
| 1 | `SilenceRemovalPreview` model | pending | -- |
| 2 | `preview_silence` / `preview_silence_async` API functions | pending | Phase 1 |
| 3 | Unit tests for preview functions | pending | Phase 2 |
| 4 | E2E test: preview vs actual comparison | pending | Phase 2 |

## Phase Dependency Graph

```
Phase 1 --> Phase 2 --> Phase 3
                    \-> Phase 4
```

Phases 3 and 4 can run in parallel (both depend on Phase 2, not on each other).
