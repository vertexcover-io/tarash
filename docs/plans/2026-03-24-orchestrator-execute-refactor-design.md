# Orchestrator Execute Methods Refactor — Design

## Problem Statement

`ExecutionOrchestrator` has 8 execute methods and 3 fallback-chain collection methods that are structurally identical (~60 lines each), differing only in config type, request/response types, and the handler method called. This is ~480 lines of near-duplicate code that must be kept in sync manually.

## Context

The orchestrator manages provider execution with automatic fallback for four modalities (video, image, TTS, STS), each with sync and async variants. All 8 methods follow the same algorithm:

1. Collect fallback chain from config
2. Iterate chain, creating `AttemptMetadata` per attempt
3. Get handler via `get_handler(cfg)`
4. Call the modality-specific handler method
5. On success: attach `ExecutionMetadata` to response, return
6. On error: check retryability, continue or raise

The three `collect_*_fallback_chain` methods are identical recursive traversals differing only in config type.

Currently, only the video methods include logging. Image/TTS/STS methods catch `NotImplementedError` separately; video methods don't.

## Requirements

### Functional Requirements
1. Replace 8 execute methods with 2 generic methods (async + sync) that accept the handler invocation as a parameter
2. Replace 3 `collect_*_fallback_chain` methods with 1 generic method
3. All modalities get consistent logging (matching current video behavior)
4. All modalities catch and re-raise `NotImplementedError` before the general error handler
5. Public API signatures in `api.py` remain unchanged
6. Existing orchestrator tests continue to pass with minimal changes

### Non-Functional Requirements
- No new dependencies
- No runtime performance regression
- Type safety preserved (mypy should not regress)
- Maintain readability — the generic method should be easy to understand

### Edge Cases and Boundary Conditions
- Empty fallback chain (should not happen but handle gracefully)
- Handler method that doesn't exist on a handler (would raise AttributeError — existing behavior, no change)
- Config types without `fallback_configs` field (not possible today, but the generic collect method should use `getattr` with default)

## Key Insights

1. **The only varying part is the handler call.** Everything else — chain traversal, metadata tracking, error handling, logging — is identical.
2. **Python can't share a method body between sync and async** without code generation or `asyncio.iscoroutinefunction` tricks. Two methods (one sync, one async) is the cleanest approach.
3. **Config types don't share a base class** but all have `provider: str`, `model: str`, and `fallback_configs: list[Self] | None`. A `Protocol` or duck typing with generics handles this cleanly.
4. **The handler call can be passed as a callable.** For async: `Callable[[handler, cfg, request, on_progress], Awaitable[Response]]`. For sync: `Callable[[handler, cfg, request, on_progress], Response]`.

## Architectural Challenges

**Sync/async split:** Python requires separate method bodies for sync and async. Options: (A) two methods with duplicated logic, (B) async-only with sync wrapper using `asyncio.run`, (C) code generation. Option A is the pragmatic choice — two ~50-line methods is far better than eight ~60-line methods, and avoids the complexity of B and C.

**Type safety without a common base class:** The generic `_collect_fallback_chain` needs to work with all three config types. A `Protocol` with `provider`, `model`, and `fallback_configs` fields would work, but adds complexity. Simpler: use `TypeVar` bound to the union, or just use duck typing with `Any` for the internal helper (it's a private method).

## Approaches Considered

### Approach A: Callable Parameter
Pass the handler invocation as a lambda/callable to `_execute_with_fallback_async` / `_execute_with_fallback_sync`. The callable receives `(handler, cfg, request, on_progress)` and returns the response. Each public method becomes a 3-line wrapper.

**Trade-offs:** Simple, no new abstractions. Lambdas can be slightly less readable but are standard Python.

### Approach B: Method Name String + getattr
Pass the handler method name as a string (e.g., `"generate_video_async"`). The generic method uses `getattr(handler, method_name)` to call it.

**Trade-offs:** Even simpler call sites, but loses type safety and IDE navigation. String-based dispatch is fragile.

### Approach C: Protocol + Strategy Pattern
Define a `FallbackConfig` protocol and a `ExecutionStrategy` that encapsulates the handler call. More OOP, more abstractions.

**Trade-offs:** Most "correct" from a design patterns perspective, but over-engineered for this use case. Adds types that exist only to serve the refactoring.

## Chosen Approach

**Approach A: Callable Parameter.** It's the simplest approach that eliminates the duplication without adding unnecessary abstractions. The callable parameter is explicit about what varies, and the generic methods contain all the shared logic.

For `collect_fallback_chain`, a single generic static method using duck typing (`getattr(config, 'fallback_configs', None)`) replaces all three specialized methods.

## High-Level Design

**Components:**

1. **`_collect_fallback_chain(config)`** — Single static method. Takes any config object, reads `fallback_configs` via attribute access, recursively collects the chain. Returns `list[ConfigT]`.

2. **`_execute_with_fallback_async(chain, invoke_handler, label)`** — Generic async execution. `invoke_handler` is an async callable `(handler, cfg) -> Response`. `label` is a string for logging (e.g., "video", "image").

3. **`_execute_with_fallback_sync(chain, invoke_handler, label)`** — Same as above but synchronous. `invoke_handler` is a sync callable.

4. **Public methods** — Thin wrappers that build the chain, define the handler callable (binding request and on_progress), and delegate to the generic method.

**Data flow:**
```
api.py -> execute_async(config, request, on_progress)
       -> _collect_fallback_chain(config)
       -> _execute_with_fallback_async(
            chain,
            lambda handler, cfg: handler.generate_video_async(cfg, request, on_progress=on_progress),
            "video"
          )
       -> iterates chain, calls invoke_handler for each cfg
       -> returns response with ExecutionMetadata
```

**Behavioral unification:**
- All modalities get the verbose logging currently only in video methods
- All modalities catch `NotImplementedError` before the general except clause
- Error handling (retryable check, chain exhaustion) identical for all

## Open Questions

None — the problem and solution are well-understood.

## Risks and Mitigations

1. **Test breakage due to behavioral changes** (logging added to image/TTS/STS, NotImplementedError handling added to video): Low risk. Existing tests mock the handler, so logging additions are invisible. NotImplementedError catch is additive.
2. **Type checker complaints with duck-typed config**: Mitigated by keeping the generic methods private and typing the public methods with concrete types.

## Assumptions

- No new modalities are being added concurrently (would cause merge conflict but not architectural issues)
- The `response.request_id` attribute exists on all response types (verified: it does)
- The `response.model_copy(update=...)` method exists on all response types (all are Pydantic models)
