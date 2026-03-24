"""Tests for generic fallback infrastructure on ExecutionOrchestrator.

These tests exercise the private _collect_fallback_chain,
_execute_with_fallback_async, and _execute_with_fallback_sync methods
added in the orchestrator refactor (Phase 1).

NOTE: We deliberately avoid importing anything from tarash.tarash_gateway.models
because Pydantic is incompatible with Python 3.14 RC2 at the module level.
Instead we use plain dataclasses that satisfy the duck-typed interfaces and
mock the import chain to bypass the Pydantic crash.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys
import types
from dataclasses import dataclass
from datetime import datetime
from typing import Literal
from unittest.mock import MagicMock

import pytest

# ---------------------------------------------------------------------------
# Resolve the source directory once so we can load modules by file path.
# ---------------------------------------------------------------------------
_SRC_DIR = (
    pathlib.Path(__file__).resolve().parents[2]
    / "src"
    / "tarash"
    / "tarash_gateway"
)


def _load_module_from_file(fqn: str, filename: str) -> types.ModuleType:
    """Load a module from *_SRC_DIR / filename* and register it in sys.modules."""
    path = _SRC_DIR / filename
    spec = importlib.util.spec_from_file_location(fqn, path)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[fqn] = mod
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Stub out the models module so orchestrator.py can be imported without
# triggering the Pydantic crash on Python 3.14 RC2.
# ---------------------------------------------------------------------------


@dataclass
class AttemptMetadata:
    """Minimal replica of the real AttemptMetadata dataclass."""

    provider: str
    model: str
    attempt_number: int
    started_at: datetime
    ended_at: datetime | None
    status: Literal["success", "failed", "skipped"]
    error_type: str | None
    error_message: str | None
    is_retryable: bool | None
    request_id: str | None


@dataclass
class ExecutionMetadata:
    """Minimal replica of the real ExecutionMetadata dataclass."""

    total_attempts: int
    successful_attempt: int | None
    attempts: list[AttemptMetadata]
    fallback_triggered: bool
    configs_in_chain: int


# Build a fake models module with the names orchestrator.py imports at the top
_fake_models = types.ModuleType("tarash.tarash_gateway.models")
_fake_models.__package__ = "tarash.tarash_gateway"
_fake_models.AttemptMetadata = AttemptMetadata  # type: ignore[attr-defined]
_fake_models.ExecutionMetadata = ExecutionMetadata  # type: ignore[attr-defined]

for _name in (
    "AudioGenerationConfig",
    "ImageGenerationConfig",
    "ImageGenerationRequest",
    "ImageGenerationResponse",
    "ImageProgressCallback",
    "ProgressCallback",
    "STSProgressCallback",
    "STSRequest",
    "STSResponse",
    "TTSProgressCallback",
    "TTSRequest",
    "TTSResponse",
    "VideoGenerationConfig",
    "VideoGenerationRequest",
    "VideoGenerationResponse",
):
    setattr(_fake_models, _name, MagicMock())

# Provide fake top-level packages so sub-module imports resolve correctly.
if "tarash" not in sys.modules:
    _pkg = types.ModuleType("tarash")
    _pkg.__path__ = [str(_SRC_DIR.parent)]
    sys.modules["tarash"] = _pkg
if "tarash.tarash_gateway" not in sys.modules:
    _gw = types.ModuleType("tarash.tarash_gateway")
    _gw.__package__ = "tarash.tarash_gateway"
    _gw.__path__ = [str(_SRC_DIR)]
    sys.modules["tarash.tarash_gateway"] = _gw

# Register fake models before anything else
sys.modules["tarash.tarash_gateway.models"] = _fake_models

# Fake logging (orchestrator calls log_info / log_error)
_fake_logging = types.ModuleType("tarash.tarash_gateway.logging")
_fake_logging.log_info = MagicMock()  # type: ignore[attr-defined]
_fake_logging.log_error = MagicMock()  # type: ignore[attr-defined]
sys.modules["tarash.tarash_gateway.logging"] = _fake_logging

# Load the real exceptions module (only uses Pydantic under TYPE_CHECKING)
_load_module_from_file("tarash.tarash_gateway.exceptions", "exceptions.py")

# Fake registry (orchestrator calls get_handler)
_fake_registry = types.ModuleType("tarash.tarash_gateway.registry")
_fake_registry.get_handler = MagicMock()  # type: ignore[attr-defined]
sys.modules["tarash.tarash_gateway.registry"] = _fake_registry

# NOW load the orchestrator module from its source file
_orch_mod = _load_module_from_file(
    "tarash.tarash_gateway.orchestrator", "orchestrator.py"
)
ExecutionOrchestrator = _orch_mod.ExecutionOrchestrator

# Provide module-level references to the fake singletons so tests can
# patch / inspect them.
_get_handler = _fake_registry.get_handler
_log_info = _fake_logging.log_info
_log_error = _fake_logging.log_error


# ---------------------------------------------------------------------------
# Lightweight duck-typed stand-ins (no Pydantic dependency)
# ---------------------------------------------------------------------------


@dataclass
class FakeConfig:
    """Minimal stand-in for any *GenerationConfig.

    Only the attributes read by _collect_fallback_chain and the generic
    execute helpers are present: provider, model, fallback_configs.
    """

    provider: str = "fake"
    model: str = "fake-model"
    fallback_configs: list[FakeConfig] | None = None


# --- Tests: _collect_fallback_chain (REQ-001, EDGE-001, EDGE-002) ----------


def test_collect_fallback_chain_no_fallbacks():
    """Single config with no fallback_configs returns one-element list.

    EDGE-001
    """
    cfg = FakeConfig(provider="p1", model="m1")
    chain = ExecutionOrchestrator._collect_fallback_chain(cfg)

    assert chain == [cfg]


def test_collect_fallback_chain_with_flat_fallbacks():
    """Config with flat fallback list returns primary + fallbacks in order.

    REQ-001
    """
    fb1 = FakeConfig(provider="p2", model="m2")
    fb2 = FakeConfig(provider="p3", model="m3")
    cfg = FakeConfig(provider="p1", model="m1", fallback_configs=[fb1, fb2])

    chain = ExecutionOrchestrator._collect_fallback_chain(cfg)

    assert len(chain) == 3
    assert chain[0] is cfg
    assert chain[1] is fb1
    assert chain[2] is fb2


def test_collect_fallback_chain_depth_first():
    """Nested fallbacks are traversed depth-first (3+ levels).

    EDGE-002
    """
    deep = FakeConfig(provider="p3", model="m3")
    mid = FakeConfig(provider="p2", model="m2", fallback_configs=[deep])
    sibling = FakeConfig(provider="p4", model="m4")
    root = FakeConfig(provider="p1", model="m1", fallback_configs=[mid, sibling])

    chain = ExecutionOrchestrator._collect_fallback_chain(root)

    assert [c.provider for c in chain] == ["p1", "p2", "p3", "p4"]


def test_collect_fallback_chain_none_fallback_configs():
    """Config with fallback_configs explicitly set to None returns single-element.

    EDGE-001 variant -- explicit None rather than missing attribute.
    """
    cfg = FakeConfig(provider="p1", model="m1", fallback_configs=None)
    chain = ExecutionOrchestrator._collect_fallback_chain(cfg)

    assert chain == [cfg]


# ---------------------------------------------------------------------------
# Helpers for execute tests
# ---------------------------------------------------------------------------

# Import exceptions we need for testing
from tarash.tarash_gateway.exceptions import (  # noqa: E402
    GenerationFailedError,
    ValidationError,
)

# We need Any for type annotations in FakeResponse
from typing import Any  # noqa: E402


@dataclass
class FakeResponse:
    """Minimal stand-in for any *GenerationResponse.

    Has ``request_id`` and a ``model_copy`` method that mimics Pydantic's
    ``BaseModel.model_copy(update=...)``.
    """

    request_id: str = "resp-1"

    def model_copy(self, *, update: dict[str, Any] | None = None) -> FakeResponse:
        """Return a shallow copy, merging *update* as extra attributes."""
        import copy

        new = copy.copy(self)
        if update:
            for k, v in update.items():
                object.__setattr__(new, k, v)
        return new


# --- Tests: _execute_with_fallback_async (REQ-002/004/005/006/007/008/009) --


@pytest.mark.asyncio
async def test_execute_with_fallback_async_success_first_attempt():
    """Async: first provider succeeds, metadata attached correctly.

    EDGE-003, REQ-004
    """
    cfg = FakeConfig(provider="p1", model="m1")
    chain = [cfg]

    _get_handler.reset_mock()
    _get_handler.return_value = MagicMock()

    async def invoke_handler(handler: Any, c: Any) -> FakeResponse:
        return FakeResponse(request_id="req-1")

    orch = ExecutionOrchestrator()
    result = await orch._execute_with_fallback_async(chain, invoke_handler, "video")

    assert result.request_id == "req-1"
    assert result.execution_metadata.total_attempts == 1
    assert result.execution_metadata.successful_attempt == 1
    assert result.execution_metadata.fallback_triggered is False
    assert result.execution_metadata.configs_in_chain == 1


@pytest.mark.asyncio
async def test_execute_with_fallback_async_fallback_on_retryable_error():
    """Async: retryable error on first provider triggers fallback to second.

    REQ-005
    """
    cfg1 = FakeConfig(provider="p1", model="m1")
    cfg2 = FakeConfig(provider="p2", model="m2")
    chain = [cfg1, cfg2]

    _get_handler.reset_mock()
    _get_handler.return_value = MagicMock()

    call_count = 0

    async def invoke_handler(handler: Any, c: Any) -> FakeResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise GenerationFailedError("provider down", provider="p1")
        return FakeResponse(request_id="req-2")

    orch = ExecutionOrchestrator()
    result = await orch._execute_with_fallback_async(chain, invoke_handler, "video")

    assert result.request_id == "req-2"
    assert result.execution_metadata.total_attempts == 2
    assert result.execution_metadata.successful_attempt == 2
    assert result.execution_metadata.fallback_triggered is True


@pytest.mark.asyncio
async def test_execute_with_fallback_async_non_retryable_stops():
    """Async: non-retryable error stops chain immediately.

    REQ-006
    """
    cfg1 = FakeConfig(provider="p1", model="m1")
    cfg2 = FakeConfig(provider="p2", model="m2")
    chain = [cfg1, cfg2]

    _get_handler.reset_mock()
    _get_handler.return_value = MagicMock()

    async def invoke_handler(handler: Any, c: Any) -> FakeResponse:
        raise ValidationError("bad input", provider="p1")

    orch = ExecutionOrchestrator()
    with pytest.raises(ValidationError, match="bad input"):
        await orch._execute_with_fallback_async(chain, invoke_handler, "video")


@pytest.mark.asyncio
async def test_execute_with_fallback_async_all_fail():
    """Async: all providers fail with retryable errors, last exception raised.

    EDGE-004, REQ-007
    """
    cfg1 = FakeConfig(provider="p1", model="m1")
    cfg2 = FakeConfig(provider="p2", model="m2")
    chain = [cfg1, cfg2]

    _get_handler.reset_mock()
    _get_handler.return_value = MagicMock()

    call_count = 0

    async def invoke_handler(handler: Any, c: Any) -> FakeResponse:
        nonlocal call_count
        call_count += 1
        raise GenerationFailedError(f"fail-{call_count}", provider=f"p{call_count}")

    orch = ExecutionOrchestrator()
    with pytest.raises(GenerationFailedError, match="fail-2"):
        await orch._execute_with_fallback_async(chain, invoke_handler, "video")


@pytest.mark.asyncio
async def test_execute_with_fallback_async_not_implemented_reraise():
    """Async: NotImplementedError is re-raised immediately.

    EDGE-005, REQ-008
    """
    cfg1 = FakeConfig(provider="p1", model="m1")
    cfg2 = FakeConfig(provider="p2", model="m2")
    chain = [cfg1, cfg2]

    _get_handler.reset_mock()
    _get_handler.return_value = MagicMock()

    call_count = 0

    async def invoke_handler(handler: Any, c: Any) -> FakeResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return FakeResponse(request_id="req-1")
        raise NotImplementedError("not supported")

    orch = ExecutionOrchestrator()
    # The first provider succeeds so we won't hit NotImplemented on attempt 1.
    # Rearrange: NotImplementedError on first provider
    call_count_2 = 0

    async def invoke_handler_2(handler: Any, c: Any) -> FakeResponse:
        nonlocal call_count_2
        call_count_2 += 1
        raise NotImplementedError("not supported")

    with pytest.raises(NotImplementedError, match="not supported"):
        await orch._execute_with_fallback_async(
            chain, invoke_handler_2, "video"
        )
    # Second provider should never be called
    assert call_count_2 == 1


@pytest.mark.asyncio
async def test_execute_with_fallback_async_mixed_errors():
    """Async: first retryable, second non-retryable -- chain stops at second.

    EDGE-006
    """
    cfg1 = FakeConfig(provider="p1", model="m1")
    cfg2 = FakeConfig(provider="p2", model="m2")
    cfg3 = FakeConfig(provider="p3", model="m3")
    chain = [cfg1, cfg2, cfg3]

    _get_handler.reset_mock()
    _get_handler.return_value = MagicMock()

    call_count = 0

    async def invoke_handler(handler: Any, c: Any) -> FakeResponse:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise GenerationFailedError("retryable", provider="p1")
        raise ValidationError("non-retryable", provider="p2")

    orch = ExecutionOrchestrator()
    with pytest.raises(ValidationError, match="non-retryable"):
        await orch._execute_with_fallback_async(chain, invoke_handler, "image")

    # Third provider should never be called
    assert call_count == 2


@pytest.mark.asyncio
async def test_execute_with_fallback_async_single_config_failure():
    """Async: single-config chain with failure raises directly.

    EDGE-007
    """
    cfg = FakeConfig(provider="p1", model="m1")
    chain = [cfg]

    _get_handler.reset_mock()
    _get_handler.return_value = MagicMock()

    async def invoke_handler(handler: Any, c: Any) -> FakeResponse:
        raise GenerationFailedError("only one", provider="p1")

    orch = ExecutionOrchestrator()
    with pytest.raises(GenerationFailedError, match="only one"):
        await orch._execute_with_fallback_async(chain, invoke_handler, "video")


# --- Tests: _execute_with_fallback_sync (REQ-003) --


def test_execute_with_fallback_sync_success_first_attempt():
    """Sync: first provider succeeds, metadata attached correctly.

    REQ-003
    """
    cfg = FakeConfig(provider="p1", model="m1")
    chain = [cfg]

    _get_handler.reset_mock()
    _get_handler.return_value = MagicMock()

    def invoke_handler(handler: Any, c: Any) -> FakeResponse:
        return FakeResponse(request_id="req-sync-1")

    orch = ExecutionOrchestrator()
    result = orch._execute_with_fallback_sync(chain, invoke_handler, "tts")

    assert result.request_id == "req-sync-1"
    assert result.execution_metadata.total_attempts == 1
    assert result.execution_metadata.successful_attempt == 1
    assert result.execution_metadata.fallback_triggered is False
    assert result.execution_metadata.configs_in_chain == 1


# --- Tests: logging (REQ-009) --


@pytest.mark.asyncio
async def test_execute_with_fallback_async_logs_with_label():
    """Log calls include the modality label for non-video modalities.

    REQ-009
    """
    cfg = FakeConfig(provider="p1", model="m1")
    chain = [cfg]

    _get_handler.reset_mock()
    _get_handler.return_value = MagicMock()
    _log_info.reset_mock()

    async def invoke_handler(handler: Any, c: Any) -> FakeResponse:
        return FakeResponse(request_id="req-log")

    orch = ExecutionOrchestrator()
    await orch._execute_with_fallback_async(chain, invoke_handler, "image")

    # Verify at least one log_info call mentions "image"
    log_messages = [str(call) for call in _log_info.call_args_list]
    assert any("image" in msg for msg in log_messages)
