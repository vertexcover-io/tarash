# Phase 3: Unit Tests for Preview Functions

> **Status:** pending
> **Depends on:** Phase 2

## Overview

Add comprehensive unit tests for `preview_silence`, `preview_silence_async`, and `_estimate_output_duration`. Tests mock the detector and probe calls to run without FFmpeg. After this phase, the preview feature has full unit test coverage.

## Implementation

**Files:**
- Modify: `packages/tarash-silence-remover/tests/unit/test_api.py` -- add tests for preview functions

**Pattern to follow:** Existing tests in `test_api.py` -- function-based, use `unittest.mock.patch` to mock detector and processor calls.

**What to test:**

`_estimate_output_duration`:
- Multiple segments with gaps exceeding `min_silence_duration` -- counts gaps, computes speech + inserted silence
- Single segment -- no gaps, duration equals segment length
- Empty segment list -- returns (0.0, 0)
- Gaps smaller than `min_silence_duration` -- not counted as silence gaps
- `target_silence_duration=0` -- no silence inserted even with gaps

`preview_silence` (sync):
- Happy path: mocks detector + probe, verifies `SilenceRemovalPreview` fields are correct
- Input file does not exist: raises `InvalidInputError`
- No speech detected (empty segments): returns preview with `estimated_output_duration=0`
- All speech (no silence): returns preview with duration equal to total speech

`preview_silence_async`:
- Happy path (async version): same as sync but with `AsyncMock` for detector and probe
- Input file does not exist: raises `InvalidInputError`

**Mocking strategy:**
- Patch `tarash.tarash_silence_remover.api._get_detector` to return a mock detector
- Patch `tarash.tarash_silence_remover.api.probe_media_info` / `probe_media_info_async` to return a `MediaInfo` with known duration
- Use `tmp_path` fixture to create a real file for the "exists" check

**Commit:** `✅ Add unit tests for preview_silence functions`

## Done When

- [ ] All tests pass: `uv run pytest packages/tarash-silence-remover/tests/unit/test_api.py -v`
- [ ] Edge cases covered: empty segments, single segment, no gaps, zero target silence
- [ ] Both sync and async variants tested
- [ ] Existing tests still pass
