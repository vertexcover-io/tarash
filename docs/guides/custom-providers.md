# Custom Providers

Tarash Gateway is extensible. You can plug in your own provider implementation
at runtime and use it exactly like a built-in provider.

## When to use this

- You have a private or internal video/image generation API
- You want to use a Fal model that isn't in the built-in registry
- You want to wrap a provider with custom pre/post-processing logic

!!! tip "Just need to add a Fal model? There's a skill for that. <span style='background:#1565C0;color:#fff;padding:1px 7px;border-radius:4px;font-size:0.72em;font-weight:700;letter-spacing:0.04em;vertical-align:middle'>BETA</span>"
    If the model is on fal.ai, run this in Claude Code:

    ```
    /add-fal-model fal-ai/your-model-id
    ```

    It fetches the model's schema from fal.ai, generates field mappers, registers the model, and writes tests — all automatically.

## Adding a Fal model in code

If you need to add a Fal model with proper field mappings without opening an issue or running the skill, you can register it at runtime using `register_provider_field_mapping`:

```python
from tarash.tarash_gateway import generate_video, register_provider_field_mapping
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest
from tarash.tarash_gateway.providers.field_mappers import (
    passthrough_field_mapper,
    duration_field_mapper,
    single_image_field_mapper,
)

# Register field mappings for fal-ai/my-model
register_provider_field_mapping("fal", {
    "fal-ai/my-model": {
        "prompt": passthrough_field_mapper("prompt", required=True),
        "duration": duration_field_mapper(
            field_type="str",
            allowed_values=["5s", "10s"],
            provider="fal",
            model="my-model",
        ),
        "image_url": single_image_field_mapper(image_type="reference"),
        "seed": passthrough_field_mapper("seed"),
        "negative_prompt": passthrough_field_mapper("negative_prompt"),
    },
})

# Now use it like any built-in model
config = VideoGenerationConfig(provider="fal", model="fal-ai/my-model", api_key="...")
request = VideoGenerationRequest(prompt="A sunset over the ocean", duration_seconds=5)
response = generate_video(config, request)
```

Call `register_provider_field_mapping` once at startup. The built-in Fal field mapper utilities cover the common cases:

| Utility | Use for |
|---|---|
| `passthrough_field_mapper(field, required)` | Fields that map 1:1 (prompt, seed, negative_prompt) |
| `duration_field_mapper(type, allowed_values)` | Duration with validation and format conversion |
| `single_image_field_mapper(image_type)` | First frame, last frame, or reference image from `image_list` |
| `image_list_field_mapper()` | All images in `image_list` as a URL list |
| `video_url_field_mapper()` | `video` field converted to URL |
| `extra_params_field_mapper(key)` | A key from `extra_params` |

For models not yet in the registry, any parameters passed via `extra_params` are forwarded to the Fal API unchanged — so you can use a model immediately even without registering field mappers.

---

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

## Notes

- `register_provider()` and `register_provider_field_mapping()` are global — call once at startup
- Registering with the same name as a built-in provider overwrites it
- The `ProviderHandler` protocol requires all four methods (`generate_video`, `generate_video_async`, `generate_image`, `generate_image_async`). Raise `NotImplementedError` for unsupported modalities
