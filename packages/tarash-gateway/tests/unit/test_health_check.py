"""Unit tests for health_check and health_check_async."""

import pytest

import tarash.tarash_gateway.mock  # noqa: F401
from tarash.tarash_gateway.api import (
    health_check,
    health_check_async,
    register_provider,
)
from tarash.tarash_gateway.models import (
    AudioGenerationConfig,
    ImageGenerationConfig,
    VideoGenerationConfig,
)
from tarash.tarash_gateway.registry import _HANDLER_INSTANCES


class StubProviderHandler:
    """Minimal stub that satisfies ProviderHandler protocol."""

    async def generate_video_async(self, config, request, on_progress=None):  # type: ignore[override]
        raise NotImplementedError

    def generate_video(self, config, request, on_progress=None):  # type: ignore[override]
        raise NotImplementedError

    async def generate_image_async(self, config, request, on_progress=None):  # type: ignore[override]
        raise NotImplementedError

    def generate_image(self, config, request, on_progress=None):  # type: ignore[override]
        raise NotImplementedError

    async def generate_tts_async(self, config, request, on_progress=None):  # type: ignore[override]
        raise NotImplementedError

    def generate_tts(self, config, request, on_progress=None):  # type: ignore[override]
        raise NotImplementedError

    async def generate_sts_async(self, config, request, on_progress=None):  # type: ignore[override]
        raise NotImplementedError

    def generate_sts(self, config, request, on_progress=None):  # type: ignore[override]
        raise NotImplementedError


@pytest.fixture(autouse=True)
def clear_handler_cache():
    _HANDLER_INSTANCES.clear()
    yield
    _HANDLER_INSTANCES.clear()


# ==================== Async Tests ====================


@pytest.mark.anyio
async def test_health_check_async_ok_single_provider():
    handler = StubProviderHandler()
    register_provider("stub", handler)

    config = VideoGenerationConfig(provider="stub", model="m", api_key="k")
    results = await health_check_async({"stub": config})

    assert "stub" in results
    assert results["stub"]["status"] == "ok"
    assert results["stub"]["latency_ms"] >= 0
    assert "error" not in results["stub"]


@pytest.mark.anyio
async def test_health_check_async_error_for_unknown_provider():
    config = VideoGenerationConfig(
        provider="nonexistent-provider", model="m", api_key="k"
    )
    results = await health_check_async({"bad": config})

    assert results["bad"]["status"] == "error"
    assert results["bad"]["latency_ms"] >= 0
    assert "error" in results["bad"]
    assert "nonexistent-provider" in results["bad"]["error"]


@pytest.mark.anyio
async def test_health_check_async_multiple_providers_mixed():
    handler = StubProviderHandler()
    register_provider("good", handler)

    configs: dict[
        str, VideoGenerationConfig | ImageGenerationConfig | AudioGenerationConfig
    ] = {
        "good": VideoGenerationConfig(provider="good", model="m", api_key="k"),
        "bad": VideoGenerationConfig(provider="missing", model="m", api_key="k"),
    }
    results = await health_check_async(configs)

    assert results["good"]["status"] == "ok"
    assert results["bad"]["status"] == "error"


@pytest.mark.anyio
async def test_health_check_async_with_image_config():
    handler = StubProviderHandler()
    register_provider("img-provider", handler)

    config = ImageGenerationConfig(provider="img-provider", model="m", api_key="k")
    results = await health_check_async({"img": config})

    assert results["img"]["status"] == "ok"


@pytest.mark.anyio
async def test_health_check_async_with_audio_config():
    handler = StubProviderHandler()
    register_provider("audio-provider", handler)

    config = AudioGenerationConfig(provider="audio-provider", model="m", api_key="k")
    results = await health_check_async({"audio": config})

    assert results["audio"]["status"] == "ok"


@pytest.mark.anyio
async def test_health_check_async_empty_configs():
    results = await health_check_async({})
    assert results == {}


# ==================== Sync Tests ====================


def test_health_check_sync_ok():
    handler = StubProviderHandler()
    register_provider("sync-stub", handler)

    config = VideoGenerationConfig(provider="sync-stub", model="m", api_key="k")
    results = health_check({"sync-stub": config})

    assert results["sync-stub"]["status"] == "ok"
    assert results["sync-stub"]["latency_ms"] >= 0


def test_health_check_sync_error_for_unknown_provider():
    config = VideoGenerationConfig(provider="no-such-provider", model="m", api_key="k")
    results = health_check({"bad": config})

    assert results["bad"]["status"] == "error"
    assert "no-such-provider" in results["bad"]["error"]


def test_health_check_sync_multiple_providers():
    handler = StubProviderHandler()
    register_provider("ok-prov", handler)

    configs: dict[
        str, VideoGenerationConfig | ImageGenerationConfig | AudioGenerationConfig
    ] = {
        "ok-prov": VideoGenerationConfig(provider="ok-prov", model="m", api_key="k"),
        "bad-prov": VideoGenerationConfig(provider="nope", model="m", api_key="k"),
    }
    results = health_check(configs)

    assert results["ok-prov"]["status"] == "ok"
    assert results["bad-prov"]["status"] == "error"


def test_health_check_result_has_correct_type():
    handler = StubProviderHandler()
    register_provider("typed", handler)

    config = VideoGenerationConfig(provider="typed", model="m", api_key="k")
    results = health_check({"typed": config})
    result = results["typed"]

    assert isinstance(result["status"], str)
    assert result["status"] in ("ok", "error")
    assert isinstance(result["latency_ms"], int)
