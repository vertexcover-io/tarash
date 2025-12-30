# Tarash Gateway

Unified interface for AI video, image, and audio generation models. Similar to LiteLLM but for generative media.

## Features

- **Unified API**: Same interface across different providers (Fal, OpenAI, Vertex AI, Replicate)
- **Type Safe**: Full Pydantic validation with IDE support
- **Async First**: Built for async/await from the ground up
- **Optional Dependencies**: Install only the providers you need
- **Debuggable**: Access raw provider responses alongside normalized data
- **Progress Tracking**: Real-time updates during generation

## Installation

```bash
# Install with specific provider
pip install tarash-gateway[fal]

# Install with multiple providers
pip install tarash-gateway[fal,openai]

# Install all providers
pip install tarash-gateway[all]
```

## Quick Start

### Simple Video Generation

```python
from tarash.tarash_gateway.video import VideoGenerationConfig, VideoGenerationRequest, generate_video

# Configure provider
config = VideoGenerationConfig(
    model="fal-ai/veo3.1",
    provider="fal",
    api_key="your_fal_key"
)

# Create request
request = VideoGenerationRequest(
    prompt="A cat jumping over a fence in slow motion",
    duration="8s",
    resolution="720p",
    aspect_ratio="16:9",
    model_params={
        "generate_audio": True,
        "seed": 42
    }
)

# Generate video
response = await generate_video(config, request)
print(f"Video URL: {response.video_url}")
```

### With Progress Tracking

```python
def on_progress(update):
    print(f"Status: {update.status}")
    if update.progress_percent:
        print(f"Progress: {update.progress_percent}%")
    if update.status == "completed":
        print(f"Done! Video: {update.result.video_url}")

response = await generate_video(config, request, on_progress=on_progress)
```

### Streaming Updates

```python
from tarash.tarash_gateway.video import generate_video_stream

async for update in generate_video_stream(config, request):
    if update.status == "completed":
        print(f"Video ready: {update.result.video_url}")
        break
    elif update.status == "processing":
        print(f"Processing: {update.message}")
    elif update.status == "failed":
        print(f"Failed: {update.error}")
```

## Supported Providers

### Fal.ai
- Models: `fal-ai/veo3`, `fal-ai/veo3.1`, `fal-ai/veo3.1/fast`
- Install: `pip install tarash-gateway[fal]`

### OpenAI (Coming Soon)
- Models: `openai/sora-2`, `openai/sora-2-pro`
- Install: `pip install tarash-gateway[openai]`

### Google Vertex AI (Coming Soon)
- Models: `google/veo-3.1`
- Install: `pip install tarash-gateway[google]`

### Replicate (Coming Soon)
- Install: `pip install tarash-gateway[replicate]`

## API Reference

### VideoGenerationConfig

```python
class VideoGenerationConfig:
    model: str              # Model identifier
    provider: str           # Provider name
    api_key: str           # API key
    base_url: str | None   # Optional custom endpoint
    timeout: int           # Request timeout (default: 600s)
    max_poll_attempts: int # Max polling attempts (default: 120)
    poll_interval: int     # Seconds between polls (default: 5)
```

### VideoGenerationRequest

```python
class VideoGenerationRequest:
    prompt: str                    # Text prompt
    duration: int | str | None     # Video duration
    resolution: str | None         # Video resolution
    aspect_ratio: str | None       # Aspect ratio
    image_urls: list[str]          # Reference images (I2V)
    video_url: str | None          # Reference video (V2V)
    model_params: dict             # Model-specific parameters
```

### VideoGenerationResponse

```python
class VideoGenerationResponse:
    id: str                   # Unique job ID
    provider_job_id: str      # Provider's job ID
    video_url: str            # Generated video URL
    audio_url: str | None     # Generated audio URL (if separate)
    duration: float | None    # Video duration in seconds
    resolution: str | None    # Video resolution
    aspect_ratio: str | None  # Video aspect ratio
    status: str               # "completed" or "failed"
    raw_response: dict        # Raw provider response
    provider_metadata: dict   # Provider-specific data
```

## Development

```bash
# Clone repository
git clone <repo-url>
cd tarash/packages/tarash-gateway

# Install dev dependencies
pip install -e ".[dev,all]"

# Run tests
pytest
```

## License

MIT
