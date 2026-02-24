# Custom Providers

Tarash Gateway is extensible. You can plug in your own provider implementation
at runtime and use it exactly like a built-in provider.

## When to use this

- You have a private or internal video/image generation API
- You want to use a Fal model that isn't in the built-in registry
- You want to wrap a provider with custom pre/post-processing logic

## Quick option: add a Fal model

If you just need to add a Fal model that isn't in the registry yet, the fastest path
is the `/add-fal-model` skill in Claude Code:

```
/add-fal-model fal-ai/your-model-id
```

This fetches the model's schema from fal.ai, generates field mappers, registers the model,
and writes tests — all automatically. To contribute the model upstream, [open a GitHub issue](https://github.com/vertexcover-io/tarash/issues).

## Implementing a custom provider

Implement the `ProviderHandler` protocol from `tarash.tarash_gateway.models`:

```python
from tarash.tarash_gateway.models import (
    VideoGenerationConfig,
    VideoGenerationRequest,
    VideoGenerationResponse,
    ImageGenerationConfig,
    ImageGenerationRequest,
    ImageGenerationResponse,
    ProgressCallback,
    ImageProgressCallback,
)
from tarash.tarash_gateway.utils import generate_unique_id


class MyProviderHandler:
    """Custom provider that calls my-provider.example.com."""

    def generate_video(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        import httpx

        response = httpx.post(
            "https://my-provider.example.com/v1/generate",
            json={"prompt": request.prompt},
            headers={"Authorization": f"Bearer {config.api_key}"},
            timeout=config.timeout,
        )
        response.raise_for_status()
        data = response.json()

        return VideoGenerationResponse(
            request_id=generate_unique_id(),
            video=data["video_url"],
            status="completed",
            raw_response=data,
        )

    async def generate_video_async(
        self,
        config: VideoGenerationConfig,
        request: VideoGenerationRequest,
        on_progress: ProgressCallback | None = None,
    ) -> VideoGenerationResponse:
        import httpx

        async with httpx.AsyncClient() as client:
            response = await client.post(
                "https://my-provider.example.com/v1/generate",
                json={"prompt": request.prompt},
                headers={"Authorization": f"Bearer {config.api_key}"},
                timeout=config.timeout,
            )
            response.raise_for_status()
            data = response.json()

        return VideoGenerationResponse(
            request_id=generate_unique_id(),
            video=data["video_url"],
            status="completed",
            raw_response=data,
        )

    def generate_image(self, config, request, on_progress=None):
        raise NotImplementedError("my-provider does not support image generation")

    async def generate_image_async(self, config, request, on_progress=None):
        raise NotImplementedError("my-provider does not support image generation")
```

## Registering your provider

```python
from tarash.tarash_gateway import register_provider

register_provider("my-provider", MyProviderHandler())
```

Call this once at application startup. After that, use it like any built-in provider:

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(
    provider="my-provider",   # matches the name you registered
    model="my-model-v1",
    api_key="...",
)

response = generate_video(config, request)
```

## Registering custom Fal field mappers in code

If your provider is Fal-based but needs specific field mappings for a model not yet
in the registry, register field mappers directly:

```python
from tarash.tarash_gateway import register_provider_field_mapping
from tarash.tarash_gateway.providers.field_mappers import (
    passthrough_field_mapper,
    duration_field_mapper,
)

register_provider_field_mapping(
    provider="fal",
    model_mappings={
        "fal-ai/my-custom-model": {
            "prompt": passthrough_field_mapper("prompt", required=True),
            "duration": duration_field_mapper(
                field_type="str",
                allowed_values=["5s", "10s"],
                provider="fal",
                model="my-custom-model",
            ),
            "seed": passthrough_field_mapper("seed"),
        },
    },
)
```

After registration, `VideoGenerationConfig(provider="fal", model="fal-ai/my-custom-model")`
uses your mappers automatically.

## Notes

- `register_provider()` and `register_provider_field_mapping()` are global — call once at startup
- Registering with the same name as a built-in provider overwrites it
- The `ProviderHandler` protocol requires all four methods (`generate_video`, `generate_video_async`, `generate_image`, `generate_image_async`). Raise `NotImplementedError` for unsupported modalities
