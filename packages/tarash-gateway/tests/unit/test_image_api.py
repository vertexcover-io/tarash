"""Unit tests for image generation API module."""

import pytest

# Import mock module to trigger model rebuilds
import tarash.tarash_gateway.mock  # noqa: F401
from tarash.tarash_gateway.api import (
    generate_image,
    generate_image_async,
    register_provider,
)
from tarash.tarash_gateway.exceptions import ValidationError
from tarash.tarash_gateway.models import (
    ImageGenerationConfig,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ProviderHandler,
    VideoGenerationResponse,
)
from tarash.tarash_gateway.registry import _HANDLER_INSTANCES


class MockImageProviderHandler(ProviderHandler):
    """Mock provider handler for testing image generation."""

    def __init__(self, images: list[str] | None = None):
        self.generate_called = False
        self.generate_async_called = False
        self.last_config = None
        self.last_request = None
        self._images = images or ["https://example.com/image.png"]

    # Video methods (required by protocol)
    async def generate_video_async(
        self, config, request, on_progress=None
    ) -> VideoGenerationResponse:
        raise NotImplementedError("Video not supported")

    def generate_video(
        self, config, request, on_progress=None
    ) -> VideoGenerationResponse:
        raise NotImplementedError("Video not supported")

    # Image methods
    async def generate_image_async(
        self, config, request, on_progress=None
    ) -> ImageGenerationResponse:
        self.generate_async_called = True
        self.last_config = config
        self.last_request = request
        return ImageGenerationResponse(
            request_id="img-async-123",
            status="completed",
            images=self._images,
            raw_response={"test": "response"},
        )

    def generate_image(
        self, config, request, on_progress=None
    ) -> ImageGenerationResponse:
        self.generate_called = True
        self.last_config = config
        self.last_request = request
        return ImageGenerationResponse(
            request_id="img-sync-456",
            status="completed",
            images=self._images,
            raw_response={"test": "response"},
        )


class MockImageProviderThatRaisesNotImplemented(ProviderHandler):
    """Mock provider that raises NotImplementedError for image generation."""

    async def generate_video_async(
        self, config, request, on_progress=None
    ) -> VideoGenerationResponse:
        return VideoGenerationResponse(
            request_id="video-123",
            status="completed",
            video="https://example.com/video.mp4",
            raw_response={},
        )

    def generate_video(
        self, config, request, on_progress=None
    ) -> VideoGenerationResponse:
        return VideoGenerationResponse(
            request_id="video-456",
            status="completed",
            video="https://example.com/video.mp4",
            raw_response={},
        )

    async def generate_image_async(
        self, config, request, on_progress=None
    ) -> ImageGenerationResponse:
        raise NotImplementedError("This provider does not support image generation")

    def generate_image(
        self, config, request, on_progress=None
    ) -> ImageGenerationResponse:
        raise NotImplementedError("This provider does not support image generation")


@pytest.fixture(autouse=True)
def clear_handler_cache():
    """Clear handler instances before each test."""
    _HANDLER_INSTANCES.clear()
    yield
    _HANDLER_INSTANCES.clear()


@pytest.fixture
def sample_image_config():
    """Create a sample image generation config."""
    return ImageGenerationConfig(
        provider="test-image-provider",
        model="test-image-model",
        api_key="test-key",
    )


@pytest.fixture
def sample_image_request():
    """Create a sample image generation request."""
    return ImageGenerationRequest(
        prompt="A beautiful sunset over the ocean",
    )


class TestGenerateImageAsync:
    """Tests for generate_image_async function."""

    @pytest.mark.anyio
    async def test_generate_image_async_returns_response(
        self, sample_image_config, sample_image_request
    ):
        """Test async image generation returns ImageGenerationResponse."""
        handler = MockImageProviderHandler()
        register_provider("test-image-provider", handler)

        response = await generate_image_async(sample_image_config, sample_image_request)

        assert handler.generate_async_called
        assert response.request_id == "img-async-123"
        assert response.status == "completed"
        assert response.images == ["https://example.com/image.png"]

    @pytest.mark.anyio
    async def test_generate_image_async_passes_config_and_request(
        self, sample_image_config, sample_image_request
    ):
        """Test that config and request are passed to handler."""
        handler = MockImageProviderHandler()
        register_provider("test-image-provider", handler)

        await generate_image_async(sample_image_config, sample_image_request)

        assert handler.last_config is sample_image_config
        assert handler.last_request is sample_image_request

    @pytest.mark.anyio
    async def test_generate_image_async_with_multiple_images(
        self, sample_image_request
    ):
        """Test async image generation with multiple images."""
        images = [
            "https://example.com/image1.png",
            "https://example.com/image2.png",
            "https://example.com/image3.png",
        ]
        handler = MockImageProviderHandler(images=images)
        register_provider("multi-image-provider", handler)

        config = ImageGenerationConfig(
            provider="multi-image-provider",
            model="multi-model",
            api_key="test-key",
        )

        response = await generate_image_async(config, sample_image_request)

        assert len(response.images) == 3

    @pytest.mark.anyio
    async def test_generate_image_async_with_progress_callback(
        self, sample_image_config, sample_image_request
    ):
        """Test async image generation with progress callback."""
        handler = MockImageProviderHandler()
        register_provider("test-image-provider", handler)

        progress_calls = []

        async def on_progress(update):
            progress_calls.append(update)

        response = await generate_image_async(
            sample_image_config, sample_image_request, on_progress
        )

        assert handler.generate_async_called
        assert response is not None

    @pytest.mark.anyio
    async def test_generate_image_async_unsupported_provider(
        self, sample_image_request
    ):
        """Test async image generation with unsupported provider raises error."""
        config = ImageGenerationConfig(
            provider="nonexistent-provider",
            model="test-model",
            api_key="test-key",
        )

        with pytest.raises(ValidationError) as exc_info:
            await generate_image_async(config, sample_image_request)

        assert "Unsupported provider: nonexistent-provider" in str(exc_info.value)

    @pytest.mark.anyio
    async def test_generate_image_async_not_implemented_propagates(
        self, sample_image_request
    ):
        """Test that NotImplementedError from handler propagates."""
        handler = MockImageProviderThatRaisesNotImplemented()
        register_provider("video-only-provider", handler)

        config = ImageGenerationConfig(
            provider="video-only-provider",
            model="video-model",
            api_key="test-key",
        )

        with pytest.raises(NotImplementedError) as exc_info:
            await generate_image_async(config, sample_image_request)

        assert "does not support image generation" in str(exc_info.value)


class TestGenerateImage:
    """Tests for generate_image synchronous function."""

    def test_generate_image_returns_response(
        self, sample_image_config, sample_image_request
    ):
        """Test sync image generation returns ImageGenerationResponse."""
        handler = MockImageProviderHandler()
        register_provider("test-image-provider", handler)

        response = generate_image(sample_image_config, sample_image_request)

        assert handler.generate_called
        assert response.request_id == "img-sync-456"
        assert response.status == "completed"
        assert response.images == ["https://example.com/image.png"]

    def test_generate_image_passes_config_and_request(
        self, sample_image_config, sample_image_request
    ):
        """Test that config and request are passed to handler."""
        handler = MockImageProviderHandler()
        register_provider("test-image-provider", handler)

        generate_image(sample_image_config, sample_image_request)

        assert handler.last_config is sample_image_config
        assert handler.last_request is sample_image_request

    def test_generate_image_with_all_request_params(self):
        """Test sync image generation with all request parameters."""
        handler = MockImageProviderHandler()
        register_provider("full-params-provider", handler)

        config = ImageGenerationConfig(
            provider="full-params-provider",
            model="full-model",
            api_key="test-key",
        )

        request = ImageGenerationRequest(
            prompt="A beautiful sunset",
            negative_prompt="blurry, low quality",
            size="1024x1024",
            quality="hd",
            style="vivid",
            n=4,
            seed=12345,
        )

        generate_image(config, request)

        assert handler.generate_called
        assert handler.last_request.prompt == "A beautiful sunset"
        assert handler.last_request.negative_prompt == "blurry, low quality"
        assert handler.last_request.size == "1024x1024"
        assert handler.last_request.n == 4
        assert handler.last_request.seed == 12345

    def test_generate_image_with_progress_callback(
        self, sample_image_config, sample_image_request
    ):
        """Test sync image generation with progress callback."""
        handler = MockImageProviderHandler()
        register_provider("test-image-provider", handler)

        progress_calls = []

        def on_progress(update):
            progress_calls.append(update)

        response = generate_image(
            sample_image_config, sample_image_request, on_progress
        )

        assert handler.generate_called
        assert response is not None

    def test_generate_image_unsupported_provider(self, sample_image_request):
        """Test sync image generation with unsupported provider raises error."""
        config = ImageGenerationConfig(
            provider="nonexistent-sync-provider",
            model="test-model",
            api_key="test-key",
        )

        with pytest.raises(ValidationError) as exc_info:
            generate_image(config, sample_image_request)

        assert "Unsupported provider: nonexistent-sync-provider" in str(exc_info.value)

    def test_generate_image_not_implemented_propagates(self, sample_image_request):
        """Test that NotImplementedError from handler propagates."""
        handler = MockImageProviderThatRaisesNotImplemented()
        register_provider("video-only-sync-provider", handler)

        config = ImageGenerationConfig(
            provider="video-only-sync-provider",
            model="video-model",
            api_key="test-key",
        )

        with pytest.raises(NotImplementedError) as exc_info:
            generate_image(config, sample_image_request)

        assert "does not support image generation" in str(exc_info.value)


class TestImageGenerationWithFallback:
    """Tests for image generation with fallback configurations."""

    @pytest.mark.anyio
    async def test_generate_image_async_includes_execution_metadata(
        self, sample_image_request
    ):
        """Test that response includes execution metadata."""
        handler = MockImageProviderHandler()
        register_provider("metadata-provider", handler)

        config = ImageGenerationConfig(
            provider="metadata-provider",
            model="metadata-model",
            api_key="test-key",
        )

        response = await generate_image_async(config, sample_image_request)

        # Execution metadata is added by orchestrator
        assert response.execution_metadata is not None
        assert response.execution_metadata.total_attempts == 1
        assert response.execution_metadata.successful_attempt == 1
        assert response.execution_metadata.fallback_triggered is False

    def test_generate_image_sync_includes_execution_metadata(
        self, sample_image_request
    ):
        """Test that sync response includes execution metadata."""
        handler = MockImageProviderHandler()
        register_provider("sync-metadata-provider", handler)

        config = ImageGenerationConfig(
            provider="sync-metadata-provider",
            model="sync-model",
            api_key="test-key",
        )

        response = generate_image(config, sample_image_request)

        assert response.execution_metadata is not None
        assert response.execution_metadata.total_attempts == 1
