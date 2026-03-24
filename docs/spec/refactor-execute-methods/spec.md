# SPEC: Orchestrator Execute Methods Refactor

**Source:** docs/plans/2026-03-24-orchestrator-execute-refactor-design.md
**Generated:** 2026-03-24

## Requirements

| ID | Type | Requirement | Acceptance Criterion | Priority |
|----|------|-------------|---------------------|----------|
| REQ-001 | Ubiquitous | The system shall provide a single `_collect_fallback_chain` static method that replaces the three existing `collect_fallback_chain`, `collect_image_fallback_chain`, and `collect_audio_fallback_chain` methods | Calling the unified method with any of the three config types returns the same ordered chain as the original specialized method | Must |
| REQ-002 | Ubiquitous | The system shall provide a generic `_execute_with_fallback_async` method that accepts a fallback chain and an async handler callable | All four async execute methods (`execute_async`, `execute_image_async`, `execute_tts_async`, `execute_sts_async`) delegate to this single method | Must |
| REQ-003 | Ubiquitous | The system shall provide a generic `_execute_with_fallback_sync` method that accepts a fallback chain and a sync handler callable | All four sync execute methods (`execute_sync`, `execute_image_sync`, `execute_tts_sync`, `execute_sts_sync`) delegate to this single method | Must |
| REQ-004 | Event-driven | When a handler call succeeds, the system shall attach `ExecutionMetadata` to the response via `model_copy(update=...)` | The returned response contains `execution_metadata` with correct `total_attempts`, `successful_attempt`, `attempts` list, `fallback_triggered` flag, and `configs_in_chain` count | Must |
| REQ-005 | Event-driven | When a handler call raises a retryable error and more fallbacks remain, the system shall continue to the next config in the chain | The next config's handler is called after a retryable error | Must |
| REQ-006 | Event-driven | When a handler call raises a non-retryable error, the system shall stop the chain and re-raise immediately | No further handlers are attempted after a non-retryable error | Must |
| REQ-007 | Event-driven | When all fallbacks in the chain are exhausted with retryable errors, the system shall raise the last exception | The final retryable exception propagates to the caller | Must |
| REQ-008 | Event-driven | When a handler call raises `NotImplementedError`, the system shall re-raise it immediately without checking retryability | `NotImplementedError` propagates directly, bypassing the fallback logic | Must |
| REQ-009 | Ubiquitous | The system shall log chain start, per-attempt progress, success details, and error details for all modalities | Log calls include provider, model, attempt number, chain length, and error context for every modality (video, image, TTS, STS) | Must |
| REQ-010 | Ubiquitous | The public API signatures in `api.py` shall remain unchanged | All existing callers in `api.py` compile and function without modification | Must |
| REQ-011 | Ubiquitous | Each public execute method shall be a thin wrapper (no more than 5 lines of body) that builds the chain, defines the handler callable, and delegates to the generic method | No public execute method contains fallback iteration logic directly | Should |

## Edge Cases

| ID | Scenario | Expected Behavior | Derived From |
|----|----------|-------------------|-------------|
| EDGE-001 | Config has no `fallback_configs` (field is `None`) | `_collect_fallback_chain` returns a single-element list containing only the root config | REQ-001 |
| EDGE-002 | Config has deeply nested fallback chain (3+ levels) | `_collect_fallback_chain` returns all configs in depth-first order | REQ-001 |
| EDGE-003 | First provider succeeds immediately | Response returned with `fallback_triggered=False`, `total_attempts=1` | REQ-004 |
| EDGE-004 | All providers fail with retryable errors | Last exception is raised; `attempts` list contains all attempts | REQ-007 |
| EDGE-005 | Second provider raises `NotImplementedError` | `NotImplementedError` raised immediately; no further fallbacks attempted | REQ-008 |
| EDGE-006 | Mixed error types: first retryable, second non-retryable | Chain stops at second provider; non-retryable error is raised | REQ-005, REQ-006 |
| EDGE-007 | Single-config chain (no fallbacks) with failure | The single error is raised directly | REQ-006, REQ-007 |

## Verification Matrix

| REQ ID | Unit Test | Integration Test | Manual Test | Notes |
|--------|-----------|-----------------|-------------|-------|
| REQ-001 | Yes | No | No | Test with all three config types |
| REQ-002 | Yes | No | No | Mock handler callable, verify delegation |
| REQ-003 | Yes | No | No | Mock handler callable, verify delegation |
| REQ-004 | Yes | No | No | Assert ExecutionMetadata fields on response |
| REQ-005 | Yes | No | No | Mock chain with retryable error then success |
| REQ-006 | Yes | No | No | Mock non-retryable error, assert no further calls |
| REQ-007 | Yes | No | No | Mock all-retryable chain, assert last exception |
| REQ-008 | Yes | No | No | Mock NotImplementedError, assert immediate raise |
| REQ-009 | Yes | No | No | Capture log calls, verify for non-video modalities |
| REQ-010 | No | No | Yes | Verify api.py callers unchanged via code review |
| REQ-011 | No | No | Yes | Verify via code review that wrappers are thin |
| EDGE-001 | Yes | No | No | |
| EDGE-002 | Yes | No | No | |
| EDGE-003 | Yes | No | No | |
| EDGE-004 | Yes | No | No | |
| EDGE-005 | Yes | No | No | |
| EDGE-006 | Yes | No | No | |
| EDGE-007 | Yes | No | No | |

## Out of Scope

- Adding a common base class for config types (VideoGenerationConfig, ImageGenerationConfig, AudioGenerationConfig) — duck typing is sufficient for the private helper
- Merging `_execute_with_fallback_async` and `_execute_with_fallback_sync` into a single method — Python's async/sync split makes this impractical without added complexity
- Changing the public API surface in `api.py` or adding new public methods to the orchestrator
- Refactoring the `get_handler` registry or provider handler interfaces
- Adding retry logic (e.g., retrying the same provider) — the orchestrator only does fallback, not retry
