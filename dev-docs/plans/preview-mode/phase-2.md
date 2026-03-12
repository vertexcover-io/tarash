# Phase 2: `preview_silence` / `preview_silence_async` API Functions

> **Status:** pending
> **Depends on:** Phase 1

## Overview

Add the two public API functions that run the detection + padding + merging pipeline but stop before FFmpeg processing, returning a `SilenceRemovalPreview`. After this phase, callers can preview silence removal without any expensive processing.

## Implementation

**Files:**
- Modify: `packages/tarash-silence-remover/src/tarash/tarash_silence_remover/api.py` -- add `preview_silence` and `preview_silence_async`
- Modify: `packages/tarash-silence-remover/src/tarash/tarash_silence_remover/__init__.py` -- export both functions

**Pattern to follow:** `detect_silence` / `detect_silence_async` in `api.py` lines 62-95 for signature style. `remove_silence` lines 118-141 for the pipeline steps to reuse.

**What to build:**

Both functions follow the same structure:

1. Validate input path exists (raise `InvalidInputError` if not)
2. `probe_media_info()` / `probe_media_info_async()` to get duration
3. Detect speech segments via `_get_detector(config)`
4. `apply_padding()` + `merge_overlapping_segments()` (pure functions, no sync/async split)
5. Compute estimated output duration from merged segments + config
6. Return `SilenceRemovalPreview`

Duration estimation logic (step 5):

```python
def _estimate_output_duration(
    merged: list[SpeechSegment],
    config: SilenceRemovalConfig,
) -> tuple[float, int]:
    """Estimate output duration and count silence gaps.

    Returns:
        Tuple of (estimated_duration, silence_gaps_count).
    """
    speech_duration = sum(seg.end - seg.start for seg in merged)

    gaps = 0
    for i in range(len(merged) - 1):
        gap = merged[i + 1].start - merged[i].end
        if gap > config.min_silence_duration and config.target_silence_duration > 0:
            gaps += 1

    estimated_output = speech_duration + (gaps * config.target_silence_duration)
    return estimated_output, gaps
```

Extract this as a private helper `_estimate_output_duration` in `api.py` so it can be tested independently if needed.

Sync function signature:
```python
def preview_silence(
    config: SilenceRemovalConfig,
    input_path: Path,
) -> SilenceRemovalPreview:
```

Async function signature:
```python
async def preview_silence_async(
    config: SilenceRemovalConfig,
    input_path: Path,
) -> SilenceRemovalPreview:
```

Add both to `__init__.py` imports and `__all__` list under the API section.

**Commit:** `✨ Add preview_silence / preview_silence_async API functions`

## Done When

- [ ] Both functions exist in `api.py` and are exported from `__init__.py`
- [ ] `_estimate_output_duration` helper is extracted as a private function
- [ ] Functions validate input, probe, detect, pad, merge, estimate, and return `SilenceRemovalPreview`
- [ ] Existing tests still pass: `uv run pytest packages/tarash-silence-remover/tests/unit/`
