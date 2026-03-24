# Phase 1: Generic Fallback Infrastructure

> **Status:** complete

## Overview

Add three new private methods to `ExecutionOrchestrator`: `_collect_fallback_chain`, `_execute_with_fallback_async`, and `_execute_with_fallback_sync`. These contain the shared fallback logic that all 8 public methods will delegate to in Phase 2. The existing public methods remain unchanged in this phase.

## Implementation

**Files:**
- Modify: `packages/tarash-gateway/src/tarash/tarash_gateway/orchestrator.py` — add 3 new private methods
- Create: `packages/tarash-gateway/tests/unit/test_orchestrator_generic.py` — tests for the generic methods

**What to build:**

### 1. `_collect_fallback_chain(config)` static method

Replace the three specialized methods with one. Uses duck typing — reads `config.fallback_configs` via `getattr(config, 'fallback_configs', None)`. Recursive depth-first traversal, identical to current logic.

Signature:
```python
@staticmethod
def _collect_fallback_chain(config: T) -> list[T]:
```

Where `T` is a TypeVar. In practice, use `Any` for the internal helper since configs don't share a base class, but the public wrappers will maintain typed signatures.

### 2. `_execute_with_fallback_async(chain, invoke_handler, label)` method

The async generic execute. Parameters:
- `chain: list` — the fallback chain (already collected)
- `invoke_handler: Callable[[Any, Any], Awaitable[Any]]` — async callable that takes `(handler, cfg)` and returns response. The caller binds `request` and `on_progress` into this callable via closure/lambda.
- `label: str` — modality name for logging (e.g., "video", "image", "tts", "sts")

Contains the full algorithm:
1. Log chain start with provider/model/chain length
2. Iterate chain, create `AttemptMetadata` per attempt
3. Call `get_handler(cfg)` then `await invoke_handler(handler, cfg)`
4. On success: attach `ExecutionMetadata`, log success, return
5. On `NotImplementedError`: re-raise immediately
6. On other exception: check `is_retryable_error`, log error details, continue or raise
7. After loop: raise last exception or RuntimeError

### 3. `_execute_with_fallback_sync(chain, invoke_handler, label)` method

Same as async but synchronous — `invoke_handler` is `Callable[[Any, Any], Any]`, no `await`.

**What to test:**
- Chain collection with no fallbacks, with fallbacks, depth-first order (EDGE-001, EDGE-002)
- Async success on first attempt (EDGE-003, REQ-004)
- Async fallback on retryable error (REQ-005)
- Async stop on non-retryable error (REQ-006)
- Async all providers fail (EDGE-004, REQ-007)
- Async NotImplementedError re-raised immediately (EDGE-005, REQ-008)
- Sync success on first attempt (REQ-003)
- Mixed errors: retryable then non-retryable (EDGE-006)
- Single config chain with failure (EDGE-007)
- Logging calls present for all modality labels (REQ-009)

**Traces to:** REQ-001, REQ-002, REQ-003, REQ-004, REQ-005, REQ-006, REQ-007, REQ-008, REQ-009, EDGE-001 through EDGE-007

**Commit:** `♻️ Add generic fallback execution infrastructure to orchestrator`

## Done When

- [x] `_collect_fallback_chain` works with all config types
- [x] `_execute_with_fallback_async` and `_execute_with_fallback_sync` pass all tests
- [x] Existing public methods and tests still work (unchanged in this phase)
- [x] Ruff passes clean
