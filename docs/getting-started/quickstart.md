# Quickstart

Generate your first video in under a minute.

## 1. Install

```bash
pip install tarash-gateway[fal]
```

## 2. Generate a video

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(
    provider="fal",
    api_key="YOUR_FAL_API_KEY",  # or set FAL_KEY env var
    model="fal-ai/veo3",
)

request = VideoGenerationRequest(
    prompt="A golden retriever running on a beach at sunset, cinematic slow motion",
    duration_seconds=4,
    aspect_ratio="16:9",
)

response = generate_video(config, request)
print(response.video)
```

## 3. Generate asynchronously

```python
import asyncio
from tarash.tarash_gateway import generate_video_async
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

async def main():
    config = VideoGenerationConfig(provider="fal", api_key="YOUR_KEY")
    request = VideoGenerationRequest(prompt="A timelapse of a blooming flower")

    response = await generate_video_async(config, request)
    print(response.video)

asyncio.run(main())
```

## 4. Track progress

```python
from tarash.tarash_gateway.models import VideoGenerationUpdate

def on_progress(update: VideoGenerationUpdate) -> None:
    print(f"Status: {update.status} — {update.progress_percent}%")

response = generate_video(config, request, on_progress=on_progress)
```

## Next steps

- [Authentication](authentication.md) — managing API keys securely
- [Video Generation Guide](../guides/video-generation.md) — full parameter reference
- [API Reference](../api-reference/gateway.md) — full function signatures
