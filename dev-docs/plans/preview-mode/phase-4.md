# Phase 4: E2E Test -- Preview vs Actual Comparison

> **Status:** pending
> **Depends on:** Phase 2

## Overview

Add an end-to-end test that runs `preview_silence` followed by `remove_silence` on the same file and asserts the estimated duration is within tolerance of the actual output duration. This validates the estimation logic against real FFmpeg processing.

## Implementation

**Files:**
- Modify: `packages/tarash-silence-remover/tests/e2e/test_silence_removal.py` -- add preview-then-remove comparison test

**Pattern to follow:** Existing e2e tests in the same file -- use `audio_with_silence` fixture, `@pytest.mark.e2e`, require `--e2e` flag.

**What to test:**

- `test_preview_then_remove_duration_within_tolerance`:
  1. Run `preview_silence(config, audio_path)` with ffmpeg detector
  2. Run `remove_silence(config, request)` on same file
  3. Assert `abs(preview.estimated_output_duration - response.output_duration) < 0.5`
  4. Assert `preview.segments_to_keep` matches `response.segments_kept`
  5. Assert `preview.detector_used == response.detector_used`

- `test_preview_then_remove_async_duration_within_tolerance`:
  - Same comparison but using async variants

- `test_preview_reports_correct_segment_count`:
  - With the known `audio_with_silence` fixture (3 tone segments), verify `len(preview.segments_to_keep)` is reasonable (2-4 depending on merge behavior)
  - Verify `preview.silence_gaps_to_insert >= 1`

**Commit:** `✅ Add E2E test comparing preview estimate vs actual duration`

## Done When

- [ ] E2E tests pass: `uv run pytest packages/tarash-silence-remover/tests/e2e/ --e2e -v`
- [ ] Estimated duration is within 0.5s of actual output duration
- [ ] Segments and detector match between preview and removal
- [ ] All existing tests still pass
