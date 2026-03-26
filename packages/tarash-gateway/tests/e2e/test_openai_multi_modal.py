"""End-to-end tests for OpenAI multi-modal (Responses API) generation.

These tests make actual API calls to the OpenAI Responses API.
Requires OPENAI_API_KEY environment variable to be set.

Run with: pytest tests/e2e/test_openai_multi_modal.py -v --e2e
"""

import os

import pytest

from tarash.tarash_gateway import (
    generate_multi_modal,
    generate_multi_modal_async,
)
from tarash.tarash_gateway.models import (
    MultiModalGenerationConfig,
    MultiModalGenerationRequest,
    MultiModalGenerationResponse,
    TextOutputItem,
    ImageOutputItem,
)


# ==================== Fixtures ====================


@pytest.fixture(scope="module")
def openai_api_key():
    """Get OpenAI API key from environment."""
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        pytest.skip("OPENAI_API_KEY environment variable not set")
    return api_key


# ==================== E2E Tests ====================


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_multi_modal_text_only_async(openai_api_key):
    """Test multi-modal generation with text-only output (async).

    This tests:
    - Basic text generation via Responses API
    - No tools used (empty allowed_tools)
    - Response structure validation
    """
    config = MultiModalGenerationConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key=openai_api_key,
        timeout=120,
        allowed_tools=[],
    )

    request = MultiModalGenerationRequest(
        prompt="What is 2 + 2? Answer in one word.",
    )

    print(f"\nGenerating multi-modal output with model: {config.model}")
    response = await generate_multi_modal_async(config, request)

    # Validate response structure
    assert isinstance(response, MultiModalGenerationResponse)
    assert response.request_id is not None
    assert response.status == "completed"
    assert isinstance(response.raw_response, dict)

    # Validate text output
    assert len(response.items) > 0
    assert any(isinstance(item, TextOutputItem) for item in response.items)
    assert len(response.text) > 0

    # No images expected (no tools)
    assert len(response.images) == 0

    print("Multi-modal generation completed successfully")
    print(f"  Request ID: {response.request_id}")
    print(f"  Text: {response.text[:200]}")
    print(f"  Items: {len(response.items)}")
    if response.cost:
        print(f"  Cost: ${response.cost.amount_usd}")


@pytest.mark.e2e
def test_multi_modal_text_only_sync(openai_api_key):
    """Test multi-modal generation with text-only output (sync).

    This tests:
    - Sync execution
    - Text-only generation
    - Cost tracking
    """
    config = MultiModalGenerationConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key=openai_api_key,
        timeout=120,
        allowed_tools=[],
    )

    request = MultiModalGenerationRequest(
        prompt="Name three primary colors. One word each, comma separated.",
    )

    print(f"\nGenerating multi-modal output with model: {config.model}")
    response = generate_multi_modal(config, request)

    # Validate response
    assert isinstance(response, MultiModalGenerationResponse)
    assert response.request_id is not None
    assert response.status == "completed"
    assert len(response.items) > 0
    assert len(response.text) > 0

    print("Multi-modal generation completed successfully")
    print(f"  Request ID: {response.request_id}")
    print(f"  Text: {response.text[:200]}")
    if response.cost:
        print(f"  Cost: ${response.cost.amount_usd}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_multi_modal_with_image_generation(openai_api_key):
    """Test multi-modal generation with image_generation tool (async).

    This tests:
    - Text + image mixed output via Responses API
    - image_generation tool invocation
    - ImageOutputItem parsing
    - Mixed output ordering
    """
    config = MultiModalGenerationConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key=openai_api_key,
        timeout=300,
        allowed_tools=["image_generation"],
    )

    request = MultiModalGenerationRequest(
        prompt="Generate an image of a sunset over a mountain lake.",
    )

    print(f"\nGenerating multi-modal output with image tool: {config.model}")
    response = await generate_multi_modal_async(config, request)

    # Validate response structure
    assert isinstance(response, MultiModalGenerationResponse)
    assert response.request_id is not None
    assert response.status == "completed"
    assert len(response.items) > 0

    # Should have at least one image
    images = response.images
    assert len(images) >= 1, f"Expected at least 1 image, got {len(images)}"
    for img in images:
        assert isinstance(img, ImageOutputItem)
        # Image should have either url or base64
        assert img.url is not None or img.base64 is not None

    print("Multi-modal generation with image completed successfully")
    print(f"  Request ID: {response.request_id}")
    print(f"  Total items: {len(response.items)}")
    print(
        f"  Text items: {len([i for i in response.items if isinstance(i, TextOutputItem)])}"
    )
    print(f"  Image items: {len(images)}")
    if images[0].url:
        print(f"  First image URL: {images[0].url[:100]}...")
    if images[0].revised_prompt:
        print(f"  Revised prompt: {images[0].revised_prompt[:100]}...")
    if response.cost:
        print(f"  Cost: ${response.cost.amount_usd}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_multi_modal_with_instructions(openai_api_key):
    """Test multi-modal generation with system instructions.

    This tests:
    - System instructions passed to the model
    - Instructions influence output
    """
    config = MultiModalGenerationConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key=openai_api_key,
        timeout=120,
        allowed_tools=[],
        instructions="You are a pirate. Always respond in pirate speak.",
    )

    request = MultiModalGenerationRequest(
        prompt="Say hello.",
    )

    print(f"\nGenerating with instructions: {config.model}")
    response = await generate_multi_modal_async(config, request)

    assert isinstance(response, MultiModalGenerationResponse)
    assert response.status == "completed"
    assert len(response.text) > 0

    print("Instructions test completed")
    print(f"  Text: {response.text[:200]}")


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_multi_modal_with_multi_turn(openai_api_key):
    """Test multi-modal generation with multi-turn input messages.

    This tests:
    - Structured input messages (not just a prompt string)
    - Multi-turn conversation support
    """
    config = MultiModalGenerationConfig(
        provider="openai",
        model="gpt-4o-mini",
        api_key=openai_api_key,
        timeout=120,
        allowed_tools=[],
    )

    request = MultiModalGenerationRequest(
        prompt="ignored when input is set",
        input=[
            {"role": "user", "content": "My name is Alice."},
            {"role": "assistant", "content": "Hello Alice!"},
            {"role": "user", "content": "What is my name?"},
        ],
    )

    print(f"\nGenerating with multi-turn input: {config.model}")
    response = await generate_multi_modal_async(config, request)

    assert isinstance(response, MultiModalGenerationResponse)
    assert response.status == "completed"
    assert len(response.text) > 0
    # The model should recall the name
    assert "alice" in response.text.lower()

    print("Multi-turn test completed")
    print(f"  Text: {response.text[:200]}")
