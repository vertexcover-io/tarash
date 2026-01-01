"""Unit tests for video generation API module."""

import pytest

from tarash.tarash_gateway.video.api import (
    _get_handler,
    generate_video,
    generate_video_async,
    get_provider_field_mapping,
    register_provider,
    register_provider_field_mapping,
    _HANDLER_INSTANCES,
    _FIELD_MAPPER_REGISTRIES,
)
from tarash.tarash_gateway.video.exceptions import ValidationError
from tarash.tarash_gateway.video.models import (
    ProviderHandler,
    VideoGenerationConfig,
    VideoGenerationRequest,
    VideoGenerationResponse,
)
from tarash.tarash_gateway.video.providers.field_mappers import (
    passthrough_field_mapper,
)


class MockProviderHandler(ProviderHandler):
    """Mock provider handler for testing."""

    def __init__(self):
        self.generate_called = False
        self.generate_async_called = False

    async def generate_video_async(
        self, config, request, on_progress=None
    ) -> VideoGenerationResponse:
        self.generate_async_called = True
        return VideoGenerationResponse(
            request_id="test-123",
            status="completed",
            video="https://example.com/video.mp4",
            raw_response={"test": "response"},
        )

    def generate_video(
        self, config, request, on_progress=None
    ) -> VideoGenerationResponse:
        self.generate_called = True
        return VideoGenerationResponse(
            request_id="test-456",
            status="completed",
            video="https://example.com/video.mp4",
            raw_response={"test": "response"},
        )


@pytest.fixture(autouse=True)
def clear_handler_cache():
    """Clear handler instances before each test."""
    _HANDLER_INSTANCES.clear()
    yield
    _HANDLER_INSTANCES.clear()


@pytest.fixture
def sample_config():
    """Create a sample video generation config."""
    return VideoGenerationConfig(
        provider="fal",
        model="fal-ai/minimax",
        api_key="test-key",
    )


@pytest.fixture
def sample_request():
    """Create a sample video generation request."""
    return VideoGenerationRequest(
        prompt="A cat playing piano",
    )


class TestGetHandler:
    """Tests for _get_handler function."""

    def test_get_handler_fal(self):
        """Test getting fal provider handler."""
        handler = _get_handler("fal")
        assert handler is not None
        assert "fal" in _HANDLER_INSTANCES
        # Subsequent calls should return same instance
        handler2 = _get_handler("fal")
        assert handler is handler2

    def test_get_handler_veo3(self):
        """Test getting veo3 provider handler."""
        handler = _get_handler("veo3")
        assert handler is not None
        assert "veo3" in _HANDLER_INSTANCES

    def test_get_handler_replicate(self):
        """Test getting replicate provider handler."""
        handler = _get_handler("replicate")
        assert handler is not None
        assert "replicate" in _HANDLER_INSTANCES

    def test_get_handler_openai(self):
        """Test getting openai provider handler."""
        handler = _get_handler("openai")
        assert handler is not None
        assert "openai" in _HANDLER_INSTANCES

    def test_get_handler_unsupported_provider(self):
        """Test getting unsupported provider raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            _get_handler("invalid-provider")

        assert "Unsupported provider: invalid-provider" in str(exc_info.value)
        assert exc_info.value.provider == "invalid-provider"


class TestRegisterProvider:
    """Tests for register_provider function."""

    def test_register_new_provider(self):
        """Test registering a new custom provider."""
        handler = MockProviderHandler()
        register_provider("custom-provider", handler)

        assert "custom-provider" in _HANDLER_INSTANCES
        assert _HANDLER_INSTANCES["custom-provider"] is handler

    def test_register_provider_overwrites_existing(self):
        """Test that registering overwrites existing provider."""
        handler1 = MockProviderHandler()
        handler2 = MockProviderHandler()

        register_provider("my-provider", handler1)
        assert _HANDLER_INSTANCES["my-provider"] is handler1

        # Overwrite with handler2
        register_provider("my-provider", handler2)
        assert _HANDLER_INSTANCES["my-provider"] is handler2

    def test_register_provider_can_be_retrieved(self):
        """Test that registered provider can be retrieved via _get_handler."""
        handler = MockProviderHandler()
        register_provider("test-provider", handler)

        retrieved = _get_handler("test-provider")
        assert retrieved is handler


class TestRegisterProviderFieldMapping:
    """Tests for register_provider_field_mapping function."""

    def test_register_field_mapping(self):
        """Test registering field mappings for a provider."""
        mappings = {
            "custom/model-1": {
                "prompt": passthrough_field_mapper("prompt", required=True),
            },
            "custom/model-2": {
                "prompt": passthrough_field_mapper("prompt", required=True),
                "duration": passthrough_field_mapper("duration_seconds"),
            },
        }

        register_provider_field_mapping("custom-provider", mappings)

        assert "custom-provider" in _FIELD_MAPPER_REGISTRIES
        assert _FIELD_MAPPER_REGISTRIES["custom-provider"] == mappings

    def test_register_field_mapping_overwrites_existing(self):
        """Test that registering overwrites existing field mappings."""
        mappings1 = {
            "model-1": {"prompt": passthrough_field_mapper("prompt")},
        }
        mappings2 = {
            "model-2": {"prompt": passthrough_field_mapper("prompt")},
        }

        register_provider_field_mapping("provider", mappings1)
        register_provider_field_mapping("provider", mappings2)

        assert _FIELD_MAPPER_REGISTRIES["provider"] == mappings2


class TestGetProviderFieldMapping:
    """Tests for get_provider_field_mapping function."""

    def test_get_existing_field_mapping(self):
        """Test getting field mappings for existing provider."""
        # fal and replicate should be registered by default
        fal_mappings = get_provider_field_mapping("fal")
        assert fal_mappings is not None
        assert isinstance(fal_mappings, dict)

        replicate_mappings = get_provider_field_mapping("replicate")
        assert replicate_mappings is not None
        assert isinstance(replicate_mappings, dict)

    def test_get_nonexistent_field_mapping(self):
        """Test getting field mappings for non-existent provider returns None."""
        result = get_provider_field_mapping("nonexistent-provider")
        assert result is None

    def test_get_custom_registered_field_mapping(self):
        """Test getting custom registered field mappings."""
        mappings = {
            "model-1": {"prompt": passthrough_field_mapper("prompt")},
        }
        register_provider_field_mapping("custom", mappings)

        result = get_provider_field_mapping("custom")
        assert result == mappings


class TestGenerateVideoAsync:
    """Tests for generate_video_async function."""

    @pytest.mark.anyio
    async def test_generate_video_async_with_custom_provider(
        self, sample_config, sample_request
    ):
        """Test async video generation with custom provider."""
        handler = MockProviderHandler()
        register_provider("test-async", handler)

        config = VideoGenerationConfig(
            provider="test-async",
            model="test-model",
            api_key="test-key",
        )

        response = await generate_video_async(config, sample_request)

        assert handler.generate_async_called
        assert response.request_id == "test-123"
        assert response.status == "completed"
        assert response.video == "https://example.com/video.mp4"

    @pytest.mark.anyio
    async def test_generate_video_async_with_progress_callback(
        self, sample_config, sample_request
    ):
        """Test async video generation with progress callback."""
        handler = MockProviderHandler()
        register_provider("test-async-progress", handler)

        config = VideoGenerationConfig(
            provider="test-async-progress",
            model="test-model",
            api_key="test-key",
        )

        progress_calls = []

        async def on_progress(response):
            progress_calls.append(response)

        response = await generate_video_async(config, sample_request, on_progress)

        assert handler.generate_async_called
        assert response is not None

    @pytest.mark.anyio
    async def test_generate_video_async_with_unsupported_provider(self, sample_request):
        """Test async video generation with unsupported provider raises error."""
        config = VideoGenerationConfig(
            provider="unsupported-provider",
            model="test-model",
            api_key="test-key",
        )

        with pytest.raises(ValidationError) as exc_info:
            await generate_video_async(config, sample_request)

        assert "Unsupported provider: unsupported-provider" in str(exc_info.value)


class TestGenerateVideo:
    """Tests for generate_video synchronous function."""

    def test_generate_video_with_custom_provider(self, sample_config, sample_request):
        """Test synchronous video generation with custom provider."""
        handler = MockProviderHandler()
        register_provider("test-sync", handler)

        config = VideoGenerationConfig(
            provider="test-sync",
            model="test-model",
            api_key="test-key",
        )

        response = generate_video(config, sample_request)

        assert handler.generate_called
        assert response.request_id == "test-456"
        assert response.status == "completed"
        assert response.video == "https://example.com/video.mp4"

    def test_generate_video_with_progress_callback(self, sample_config, sample_request):
        """Test synchronous video generation with progress callback."""
        handler = MockProviderHandler()
        register_provider("test-sync-progress", handler)

        config = VideoGenerationConfig(
            provider="test-sync-progress",
            model="test-model",
            api_key="test-key",
        )

        progress_calls = []

        def on_progress(response):
            progress_calls.append(response)

        response = generate_video(config, sample_request, on_progress)

        assert handler.generate_called
        assert response is not None

    def test_generate_video_with_unsupported_provider(self, sample_request):
        """Test synchronous video generation with unsupported provider raises error."""
        config = VideoGenerationConfig(
            provider="unsupported-provider-sync",
            model="test-model",
            api_key="test-key",
        )

        with pytest.raises(ValidationError) as exc_info:
            generate_video(config, sample_request)

        assert "Unsupported provider: unsupported-provider-sync" in str(exc_info.value)
