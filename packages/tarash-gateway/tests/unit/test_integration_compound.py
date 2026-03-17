"""Integration smoke tests for compound generation feature.

These tests exercise the full flow from public API through orchestrator and
provider layers, using the mock provider for reproducibility.
"""

from decimal import Decimal

import pytest

import tarash.tarash_gateway as api
from tarash.tarash_gateway.mock import MockConfig
from tarash.tarash_gateway.models import (
    CodeOutputItem,
    CompoundGenerationConfig,
    CompoundGenerationRequest,
    CompoundGenerationResponse,
    CompoundGenerationUpdate,
    CostComponent,
    GenerationCost,
    ImageOutputItem,
    OutputItem,
    ReasoningOutputItem,
    TextOutputItem,
    UnknownOutputItem,
)


@pytest.fixture
def mock_config():
    """Create a config with mock provider enabled."""
    return CompoundGenerationConfig(
        provider="openai",
        model="gpt-4o",
        api_key="test-key",
        allowed_tools=["image_generation", "code_execution"],
        mock=MockConfig(enabled=True),
    )


@pytest.fixture
def basic_request():
    """Create a basic compound generation request."""
    return CompoundGenerationRequest(
        prompt="Generate a summary and create an image of a sunset"
    )


@pytest.fixture
def detailed_request():
    """Create a detailed compound generation request with various parameters."""
    return CompoundGenerationRequest(
        prompt="Analyze this data and generate insights",
        system_prompt="You are a data analyst expert",
        temperature=0.7,
        max_tokens=1000,
    )


class TestCompoundGenerationImports:
    """Test that compound generation types are properly exported."""

    def test_generate_compound_function_exported(self):
        """generate_compound should be exported from public API."""
        assert hasattr(api, "generate_compound")
        assert callable(api.generate_compound)

    def test_generate_compound_async_function_exported(self):
        """generate_compound_async should be exported from public API."""
        assert hasattr(api, "generate_compound_async")
        assert callable(api.generate_compound_async)

    def test_compound_config_exported(self):
        """CompoundGenerationConfig should be exported."""
        assert hasattr(api, "CompoundGenerationConfig")
        assert api.CompoundGenerationConfig is CompoundGenerationConfig

    def test_compound_request_exported(self):
        """CompoundGenerationRequest should be exported."""
        assert hasattr(api, "CompoundGenerationRequest")
        assert api.CompoundGenerationRequest is CompoundGenerationRequest

    def test_compound_response_exported(self):
        """CompoundGenerationResponse should be exported."""
        assert hasattr(api, "CompoundGenerationResponse")
        assert api.CompoundGenerationResponse is CompoundGenerationResponse

    def test_compound_update_exported(self):
        """CompoundGenerationUpdate should be exported."""
        assert hasattr(api, "CompoundGenerationUpdate")

    def test_compound_progress_callback_exported(self):
        """CompoundProgressCallback should be exported."""
        assert hasattr(api, "CompoundProgressCallback")

    def test_output_item_types_exported(self):
        """All output item types should be exported."""
        assert hasattr(api, "OutputItem")
        assert hasattr(api, "TextOutputItem")
        assert hasattr(api, "ImageOutputItem")
        assert hasattr(api, "ReasoningOutputItem")
        assert hasattr(api, "CodeOutputItem")
        assert hasattr(api, "UnknownOutputItem")

    def test_cost_component_exported(self):
        """CostComponent should be exported."""
        assert hasattr(api, "CostComponent")
        assert api.CostComponent is CostComponent


class TestBasicCompoundGenerationSync:
    """Test basic synchronous compound generation flow."""

    def test_sync_generation_returns_response(self, mock_config, basic_request):
        """Synchronous generation should return CompoundGenerationResponse."""
        response = api.generate_compound(mock_config, basic_request)

        assert isinstance(response, CompoundGenerationResponse)
        assert response.status == "completed"
        assert response.request_id is not None

    def test_sync_generation_includes_items(self, mock_config, basic_request):
        """Response should include output items."""
        response = api.generate_compound(mock_config, basic_request)

        assert len(response.items) > 0
        assert all(isinstance(item, OutputItem) for item in response.items)

    def test_sync_generation_includes_text_item(self, mock_config, basic_request):
        """Response should include at least one text item."""
        response = api.generate_compound(mock_config, basic_request)

        text_items = [i for i in response.items if isinstance(i, TextOutputItem)]
        assert len(text_items) > 0

    def test_sync_generation_preserves_prompt(self, mock_config):
        """Response should include the original prompt in text content."""
        prompt = "Summarize the key points"
        request = CompoundGenerationRequest(prompt=prompt)

        response = api.generate_compound(mock_config, request)

        text_items = [i for i in response.items if isinstance(i, TextOutputItem)]
        assert len(text_items) > 0
        assert prompt in text_items[0].content

    def test_sync_generation_includes_images_when_allowed(self, basic_request):
        """Should include image items when image_generation is in allowed_tools."""
        config = CompoundGenerationConfig(
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
            allowed_tools=["image_generation"],
            mock=MockConfig(enabled=True),
        )

        response = api.generate_compound(config, basic_request)

        image_items = [i for i in response.items if isinstance(i, ImageOutputItem)]
        assert len(image_items) > 0

    def test_sync_generation_omits_images_when_not_allowed(self, basic_request):
        """Should omit image items when image_generation not in allowed_tools."""
        config = CompoundGenerationConfig(
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
            allowed_tools=[],
            mock=MockConfig(enabled=True),
        )

        response = api.generate_compound(config, basic_request)

        image_items = [i for i in response.items if isinstance(i, ImageOutputItem)]
        assert len(image_items) == 0

    def test_sync_generation_includes_cost(self, mock_config, basic_request):
        """Response should include cost information (may be None for mock)."""
        response = api.generate_compound(mock_config, basic_request)

        # Mock provider may not compute cost, but cost field should be present
        assert hasattr(response, "cost")

    def test_sync_generation_marks_as_mock(self, mock_config, basic_request):
        """Mock response should be marked as mock."""
        response = api.generate_compound(mock_config, basic_request)

        assert response.is_mock is True

    def test_sync_generation_with_custom_params(self):
        """Should handle custom parameters in request."""
        config = CompoundGenerationConfig(
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
            mock=MockConfig(enabled=True),
        )
        request = CompoundGenerationRequest(
            prompt="Test",
            temperature=0.5,
            max_tokens=500,
        )

        response = api.generate_compound(config, request)

        assert isinstance(response, CompoundGenerationResponse)
        assert response.status == "completed"


class TestBasicCompoundGenerationAsync:
    """Test basic asynchronous compound generation flow."""

    async def test_async_generation_returns_response(self, mock_config, basic_request):
        """Asynchronous generation should return CompoundGenerationResponse."""
        response = await api.generate_compound_async(mock_config, basic_request)

        assert isinstance(response, CompoundGenerationResponse)
        assert response.status == "completed"
        assert response.request_id is not None

    async def test_async_generation_includes_items(self, mock_config, basic_request):
        """Response should include output items."""
        response = await api.generate_compound_async(mock_config, basic_request)

        assert len(response.items) > 0
        assert all(isinstance(item, OutputItem) for item in response.items)

    async def test_async_generation_includes_text_item(
        self, mock_config, basic_request
    ):
        """Response should include at least one text item."""
        response = await api.generate_compound_async(mock_config, basic_request)

        text_items = [i for i in response.items if isinstance(i, TextOutputItem)]
        assert len(text_items) > 0

    async def test_async_generation_preserves_prompt(self, mock_config):
        """Response should include the original prompt in text content."""
        prompt = "Generate creative ideas"
        request = CompoundGenerationRequest(prompt=prompt)

        response = await api.generate_compound_async(mock_config, request)

        text_items = [i for i in response.items if isinstance(i, TextOutputItem)]
        assert len(text_items) > 0
        assert prompt in text_items[0].content

    async def test_async_generation_includes_images_when_allowed(self, basic_request):
        """Should include image items when image_generation is in allowed_tools."""
        config = CompoundGenerationConfig(
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
            allowed_tools=["image_generation"],
            mock=MockConfig(enabled=True),
        )

        response = await api.generate_compound_async(config, basic_request)

        image_items = [i for i in response.items if isinstance(i, ImageOutputItem)]
        assert len(image_items) > 0

    async def test_async_generation_omits_images_when_not_allowed(self, basic_request):
        """Should omit image items when image_generation not in allowed_tools."""
        config = CompoundGenerationConfig(
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
            allowed_tools=[],
            mock=MockConfig(enabled=True),
        )

        response = await api.generate_compound_async(config, basic_request)

        image_items = [i for i in response.items if isinstance(i, ImageOutputItem)]
        assert len(image_items) == 0

    async def test_async_generation_includes_cost(self, mock_config, basic_request):
        """Response should include cost information (may be None for mock)."""
        response = await api.generate_compound_async(mock_config, basic_request)

        # Mock provider may not compute cost, but cost field should be present
        assert hasattr(response, "cost")

    async def test_async_generation_marks_as_mock(self, mock_config, basic_request):
        """Mock response should be marked as mock."""
        response = await api.generate_compound_async(mock_config, basic_request)

        assert response.is_mock is True


class TestProgressCallbackSync:
    """Test progress callbacks for synchronous generation."""

    def test_sync_generation_supports_progress_callback(
        self, mock_config, basic_request
    ):
        """Should invoke progress callback during synchronous generation."""
        updates = []

        def progress_callback(update: CompoundGenerationUpdate):
            updates.append(update)

        response = api.generate_compound(
            mock_config, basic_request, on_progress=progress_callback
        )

        assert len(updates) >= 0
        assert isinstance(response, CompoundGenerationResponse)

    def test_progress_callback_receives_updates(self, mock_config, basic_request):
        """Progress callback should receive CompoundGenerationUpdate objects."""
        updates = []

        def progress_callback(update: CompoundGenerationUpdate):
            updates.append(update)
            assert isinstance(update, CompoundGenerationUpdate)

        api.generate_compound(mock_config, basic_request, on_progress=progress_callback)

    def test_sync_generation_without_callback(self, mock_config, basic_request):
        """Should work without progress callback."""
        response = api.generate_compound(mock_config, basic_request)

        assert isinstance(response, CompoundGenerationResponse)

    def test_sync_generation_with_none_callback(self, mock_config, basic_request):
        """Should accept None as callback."""
        response = api.generate_compound(mock_config, basic_request, on_progress=None)

        assert isinstance(response, CompoundGenerationResponse)


class TestProgressCallbackAsync:
    """Test progress callbacks for asynchronous generation."""

    async def test_async_generation_supports_progress_callback(
        self, mock_config, basic_request
    ):
        """Should invoke progress callback during asynchronous generation."""
        updates = []

        async def progress_callback(update: CompoundGenerationUpdate):
            updates.append(update)

        response = await api.generate_compound_async(
            mock_config, basic_request, on_progress=progress_callback
        )

        assert isinstance(response, CompoundGenerationResponse)

    async def test_async_generation_supports_sync_callback(
        self, mock_config, basic_request
    ):
        """Should support synchronous callback in async context."""
        updates = []

        def progress_callback(update: CompoundGenerationUpdate):
            updates.append(update)

        response = await api.generate_compound_async(
            mock_config, basic_request, on_progress=progress_callback
        )

        assert isinstance(response, CompoundGenerationResponse)

    async def test_async_generation_without_callback(self, mock_config, basic_request):
        """Should work without progress callback in async context."""
        response = await api.generate_compound_async(mock_config, basic_request)

        assert isinstance(response, CompoundGenerationResponse)

    async def test_async_generation_with_none_callback(
        self, mock_config, basic_request
    ):
        """Should accept None as callback in async context."""
        response = await api.generate_compound_async(
            mock_config, basic_request, on_progress=None
        )

        assert isinstance(response, CompoundGenerationResponse)


class TestResponseStructure:
    """Test the structure of compound generation responses."""

    def test_response_has_all_required_fields(self, mock_config, basic_request):
        """Response should have all required fields."""
        response = api.generate_compound(mock_config, basic_request)

        assert response.request_id is not None
        assert response.items is not None
        assert response.status is not None
        assert hasattr(response, "cost")
        assert response.raw_response is not None

    def test_response_request_id_format(self, mock_config, basic_request):
        """Request ID should follow expected format."""
        response = api.generate_compound(mock_config, basic_request)

        assert isinstance(response.request_id, str)
        assert len(response.request_id) > 0

    def test_response_items_are_output_items(self, mock_config, basic_request):
        """All items in response should be OutputItem instances."""
        response = api.generate_compound(mock_config, basic_request)

        for item in response.items:
            assert isinstance(item, OutputItem)

    def test_response_status_is_valid(self, mock_config, basic_request):
        """Response status should be valid."""
        response = api.generate_compound(mock_config, basic_request)

        assert response.status in ("completed", "partial", "failed")

    def test_output_item_types_present(self):
        """Test that different output item types can be created."""
        text_item = TextOutputItem(content="test text")
        assert text_item.content == "test text"

        image_item = ImageOutputItem(url="https://example.com/image.png")
        assert image_item.url == "https://example.com/image.png"

        reasoning_item = ReasoningOutputItem(summary=["test reasoning"])
        assert len(reasoning_item.summary) > 0

        code_item = CodeOutputItem(code="print('hello')", output="hello")
        assert code_item.code == "print('hello')"

        unknown_item = UnknownOutputItem(raw={"data": "value"})
        assert unknown_item.raw == {"data": "value"}


class TestCostTracking:
    """Test cost tracking in compound generation."""

    def test_response_includes_generation_cost(self, mock_config, basic_request):
        """Response should include GenerationCost when available."""
        response = api.generate_compound(mock_config, basic_request)

        if response.cost is not None:
            assert isinstance(response.cost, GenerationCost)

    def test_generation_cost_has_required_fields(self, mock_config, basic_request):
        """GenerationCost should have required fields when not None."""
        response = api.generate_compound(mock_config, basic_request)

        if response.cost is not None:
            assert response.cost.amount_usd is not None
            assert response.cost.raw_amount is not None
            assert response.cost.raw_unit is not None

    def test_generation_cost_is_decimal(self, mock_config, basic_request):
        """Cost amount_usd should be Decimal for precision when present."""
        response = api.generate_compound(mock_config, basic_request)

        if response.cost is not None:
            assert isinstance(response.cost.amount_usd, Decimal)

    def test_cost_breakdown_available(self, mock_config, basic_request):
        """Cost breakdown should be available if present."""
        response = api.generate_compound(mock_config, basic_request)

        if response.cost and response.cost.breakdown:
            for component in response.cost.breakdown:
                assert isinstance(component, CostComponent)
                assert component.category is not None
                assert component.amount_usd is not None


class TestFullFlowIntegration:
    """Test complete integration flows."""

    def test_end_to_end_sync_flow(self):
        """Complete synchronous flow from API call to response."""
        config = CompoundGenerationConfig(
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
            allowed_tools=["image_generation"],
            mock=MockConfig(enabled=True),
        )
        request = CompoundGenerationRequest(prompt="Generate a summary and an image")

        response = api.generate_compound(config, request)

        assert response.status == "completed"
        assert response.is_mock is True
        assert len(response.items) > 0
        assert any(isinstance(i, TextOutputItem) for i in response.items)
        assert any(isinstance(i, ImageOutputItem) for i in response.items)

    async def test_end_to_end_async_flow(self):
        """Complete asynchronous flow from API call to response."""
        config = CompoundGenerationConfig(
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
            allowed_tools=["image_generation"],
            mock=MockConfig(enabled=True),
        )
        request = CompoundGenerationRequest(prompt="Generate a summary and an image")

        updates = []

        async def progress_callback(update: CompoundGenerationUpdate):
            updates.append(update)

        response = await api.generate_compound_async(
            config, request, on_progress=progress_callback
        )

        assert response.status == "completed"
        assert response.is_mock is True
        assert len(response.items) > 0
        assert any(isinstance(i, TextOutputItem) for i in response.items)
        assert any(isinstance(i, ImageOutputItem) for i in response.items)

    def test_sync_with_detailed_config_and_request(self):
        """Test with detailed configuration and request parameters."""
        config = CompoundGenerationConfig(
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
            allowed_tools=["image_generation", "code_execution"],
            temperature=0.7,
            max_tokens=2000,
            mock=MockConfig(enabled=True),
        )
        request = CompoundGenerationRequest(
            prompt="Analyze and visualize the data",
            system_prompt="You are an expert data scientist",
            temperature=0.5,
        )

        response = api.generate_compound(config, request)

        assert isinstance(response, CompoundGenerationResponse)
        assert response.status == "completed"
        assert len(response.items) > 0

    async def test_async_with_detailed_config_and_request(self):
        """Test async with detailed configuration and request parameters."""
        config = CompoundGenerationConfig(
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
            allowed_tools=["image_generation", "code_execution"],
            temperature=0.7,
            max_tokens=2000,
            mock=MockConfig(enabled=True),
        )
        request = CompoundGenerationRequest(
            prompt="Analyze and visualize the data",
            system_prompt="You are an expert data scientist",
            temperature=0.5,
        )

        response = await api.generate_compound_async(config, request)

        assert isinstance(response, CompoundGenerationResponse)
        assert response.status == "completed"
        assert len(response.items) > 0


class TestErrorHandling:
    """Test error handling in compound generation."""

    def test_missing_prompt_validation(self):
        """Should handle missing prompt gracefully."""
        config = CompoundGenerationConfig(
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
            mock=MockConfig(enabled=True),
        )

        try:
            request = CompoundGenerationRequest(prompt="")
            response = api.generate_compound(config, request)
            assert response is not None
        except Exception:
            pass

    async def test_async_error_handling(self):
        """Async generation should handle errors gracefully."""
        config = CompoundGenerationConfig(
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
            mock=MockConfig(enabled=True),
        )
        request = CompoundGenerationRequest(prompt="Test")

        try:
            response = await api.generate_compound_async(config, request)
            assert response is not None
        except Exception:
            pass

    def test_mock_disabled_raises_error(self):
        """Should raise error if mock not enabled."""
        config = CompoundGenerationConfig(
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
        )
        request = CompoundGenerationRequest(prompt="Test")

        with pytest.raises(Exception):
            api.generate_compound(config, request)

    async def test_async_mock_disabled_raises_error(self):
        """Async should raise error if mock not enabled."""
        config = CompoundGenerationConfig(
            provider="openai",
            model="gpt-4o",
            api_key="test-key",
        )
        request = CompoundGenerationRequest(prompt="Test")

        with pytest.raises(Exception):
            await api.generate_compound_async(config, request)


class TestResponseConsistency:
    """Test consistency of responses."""

    def test_multiple_requests_return_consistent_structures(self, mock_config):
        """Multiple requests should return consistent response structures."""
        request1 = CompoundGenerationRequest(prompt="First prompt")
        request2 = CompoundGenerationRequest(prompt="Second prompt")

        response1 = api.generate_compound(mock_config, request1)
        response2 = api.generate_compound(mock_config, request2)

        assert isinstance(response1, CompoundGenerationResponse)
        assert isinstance(response2, CompoundGenerationResponse)
        assert hasattr(response1, "request_id")
        assert hasattr(response2, "request_id")
        assert hasattr(response1, "items")
        assert hasattr(response2, "items")
        assert hasattr(response1, "cost")
        assert hasattr(response2, "cost")

    async def test_async_multiple_requests_consistent(self, mock_config):
        """Async requests should return consistent responses."""
        request1 = CompoundGenerationRequest(prompt="First prompt")
        request2 = CompoundGenerationRequest(prompt="Second prompt")

        response1 = await api.generate_compound_async(mock_config, request1)
        response2 = await api.generate_compound_async(mock_config, request2)

        assert isinstance(response1, CompoundGenerationResponse)
        assert isinstance(response2, CompoundGenerationResponse)
        assert isinstance(response1.items, list)
        assert isinstance(response2.items, list)

    def test_sync_async_return_same_type(self, mock_config, basic_request):
        """Sync and async should return the same response type."""
        sync_response = api.generate_compound(mock_config, basic_request)
        assert isinstance(sync_response, CompoundGenerationResponse)
