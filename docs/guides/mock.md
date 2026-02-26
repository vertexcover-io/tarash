# Mock Provider

The mock provider lets you run your full video and image generation pipeline locally
without hitting any real API. No API keys needed, no credits spent.

## Basic usage

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest
from tarash.tarash_gateway.mock import MockConfig

config = VideoGenerationConfig(
    provider="fal",        # provider field still required
    model="fal-ai/veo3.1/fast",   # model still required
    mock=MockConfig(enabled=True),
)

request = VideoGenerationRequest(prompt="A cat playing piano")

response = generate_video(config, request)
print(response.video)       # → URL to a stock mock video
print(response.is_mock)     # → True
```

When `mock=MockConfig(enabled=True)` is set, Tarash bypasses the real provider entirely
and returns a pre-configured mock response. The `provider` and `model` values are
recorded in the response but no real API call is made.

---

## Default behavior

By default, `MockConfig(enabled=True)` returns a randomly selected stock video
from a built-in library matched to your request. The response is a valid
`VideoGenerationResponse` with `is_mock=True`.

---

## Customizing mock responses

Control exactly what the mock returns using `MockResponse`:

```python
from tarash.tarash_gateway.mock import MockConfig, MockResponse

config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/veo3.1/fast",
    mock=MockConfig(
        enabled=True,
        responses=[
            MockResponse(
                weight=1.0,
                video_url="https://example.com/my-test-video.mp4",
            ),
        ],
    ),
)
```

---

## Injecting errors for testing

Test your error handling by configuring the mock to raise specific exceptions:

```python
from tarash.tarash_gateway.exceptions import GenerationFailedError
from tarash.tarash_gateway.mock import MockConfig, MockResponse

config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/veo3.1/fast",
    mock=MockConfig(
        enabled=True,
        responses=[
            MockResponse(weight=1.0, error=GenerationFailedError("Simulated failure")),
        ],
    ),
)

try:
    response = generate_video(config, request)
except GenerationFailedError as e:
    print(f"Caught: {e}")  # → "Caught: Simulated failure"
```

---

## Weighted responses

Simulate flaky providers by mixing success and error responses with weights:

```python
mock = MockConfig(
    enabled=True,
    responses=[
        MockResponse(weight=0.8),                                               # 80% succeed
        MockResponse(weight=0.2, error=GenerationFailedError("rate limited")),  # 20% fail
    ],
)
```

---

## Mock + fallback chains

Combine mock with fallback to test your fallback logic without hitting any real API:

```python
config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/veo3.1/fast",
    mock=MockConfig(
        enabled=True,
        responses=[MockResponse(weight=1.0, error=GenerationFailedError("forced fail"))],
    ),
    fallback_configs=[
        VideoGenerationConfig(
            provider="replicate",
            model="google/veo-3.1",
            mock=MockConfig(enabled=True),  # fallback mock succeeds
        ),
    ],
)

response = generate_video(config, request)
assert response.execution_metadata.fallback_triggered is True
```

---

## Image generation mock

`MockConfig` works the same way for image generation:

```python
from tarash.tarash_gateway import generate_image
from tarash.tarash_gateway.models import ImageGenerationConfig, ImageGenerationRequest

config = ImageGenerationConfig(
    provider="openai",
    model="dall-e-3",
    mock=MockConfig(enabled=True),
)
response = generate_image(config, ImageGenerationRequest(prompt="A sunset"))
print(response.images[0])   # → URL to a stock mock image
print(response.is_mock)     # → True
```

---

## API reference

See [Mock Provider API reference](../api-reference/mock.md) for the full `MockConfig`
and `MockResponse` field reference.
