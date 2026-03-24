# Phase 2: Migrate Public Methods

> **Status:** complete

## Overview

Replace the bodies of all 8 public execute methods and 3 collect methods with thin wrappers that delegate to the generic infrastructure from Phase 1. Delete the duplicated code. Update existing tests to work with the new structure.

## Implementation

**Files:**
- Modify: `packages/tarash-gateway/src/tarash/tarash_gateway/orchestrator.py` — rewrite 8 execute methods as thin wrappers, remove 3 old collect methods
- Modify: `packages/tarash-gateway/tests/unit/video/test_orchestrator.py` — update `collect_fallback_chain` tests to use the unified method

**What to build:**

### 1. Replace collect methods

Remove `collect_fallback_chain`, `collect_image_fallback_chain`, and `collect_audio_fallback_chain`. Keep `_collect_fallback_chain` (from Phase 1) as the single implementation.

Update public method references:
- `self.collect_fallback_chain(config)` → `self._collect_fallback_chain(config)`
- `self.collect_image_fallback_chain(config)` → `self._collect_fallback_chain(config)`
- `self.collect_audio_fallback_chain(config)` → `self._collect_fallback_chain(config)`

Note: `collect_fallback_chain` was `@staticmethod` and called as `ExecutionOrchestrator.collect_fallback_chain(...)` in tests. The unified `_collect_fallback_chain` remains a static method.

### 2. Rewrite 8 execute methods as thin wrappers

Each public method becomes ~5 lines:

```python
async def execute_async(
    self,
    config: VideoGenerationConfig,
    request: VideoGenerationRequest,
    on_progress: ProgressCallback | None = None,
) -> VideoGenerationResponse:
    chain = self._collect_fallback_chain(config)
    return await self._execute_with_fallback_async(
        chain,
        lambda handler, cfg: handler.generate_video_async(cfg, request, on_progress=on_progress),
        "video",
    )
```

Same pattern for all 8, varying only:
- `execute_sync` → `_execute_with_fallback_sync`, `handler.generate_video`
- `execute_image_async` → `_execute_with_fallback_async`, `handler.generate_image_async`
- `execute_image_sync` → `_execute_with_fallback_sync`, `handler.generate_image`
- `execute_tts_async` → `_execute_with_fallback_async`, `handler.generate_tts_async`
- `execute_tts_sync` → `_execute_with_fallback_sync`, `handler.generate_tts`
- `execute_sts_async` → `_execute_with_fallback_async`, `handler.generate_sts_async`
- `execute_sts_sync` → `_execute_with_fallback_sync`, `handler.generate_sts`

### 3. Update existing tests

- `test_collect_fallback_chain_*` tests: Update to call `ExecutionOrchestrator._collect_fallback_chain(config)` instead of `ExecutionOrchestrator.collect_fallback_chain(config)`
- `test_execute_async_*` and `test_execute_sync_*` tests: These should continue to pass unchanged since the public API signatures haven't changed. The mock patching of `get_handler` still works since the generic methods call the same function.

### 4. Remove dead code

Delete the section comments (`# ==================== Image Generation ====================` etc.) and all the old method bodies. The file went from ~815 lines to ~474 lines (the generic fallback methods contain the shared logic that was previously duplicated).

**What to test:**
- All existing tests in `test_orchestrator.py` pass (REQ-010)
- Each public method is a thin wrapper delegating to generic (REQ-011)

**Traces to:** REQ-010, REQ-011

**Commit:** `♻️ Migrate orchestrator execute methods to generic fallback infrastructure`

## Done When

- [x] All 8 public execute methods are thin wrappers (max 5 lines body each)
- [x] 3 old collect methods removed, replaced by single `_collect_fallback_chain`
- [x] All existing tests pass
- [x] Ruff passes clean
- [ ] orchestrator.py is under 250 lines — actual: 474 lines (generic methods carry the shared logic; wrappers are thin as required)
