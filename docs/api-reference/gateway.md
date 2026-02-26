# Gateway

Public API entry points. Import from `tarash.tarash_gateway`.

```python
from tarash.tarash_gateway import (
    generate_video,
    generate_video_async,
    generate_image,
    generate_image_async,
    register_provider,
    register_provider_field_mapping,
    get_provider_field_mapping,
)
```

| Function | Sync/Async | `on_progress` receives | Returns |
|---|---|---|---|
| `generate_video(config, request, on_progress=None)` | Sync | `VideoGenerationUpdate` | `VideoGenerationResponse` |
| `generate_video_async(config, request, on_progress=None)` | Async | `VideoGenerationUpdate` | `VideoGenerationResponse` |
| `generate_image(config, request, on_progress=None)` | Sync | `ImageGenerationUpdate` | `ImageGenerationResponse` |
| `generate_image_async(config, request, on_progress=None)` | Async | `ImageGenerationUpdate` | `ImageGenerationResponse` |
| `register_provider(name, handler)` | — | — | `None` |
| `register_provider_field_mapping(provider, model_mappings)` | — | — | `None` |
| `get_provider_field_mapping(provider)` | — | — | `dict | None` |

`on_progress` accepts both sync and async callables. See the [Custom Providers guide](../guides/custom-providers.md) for registration usage.

---

::: tarash.tarash_gateway.api
    options:
      show_source: true
      show_signature_annotations: true
      show_symbol_type_heading: true
      members_order: source
      separate_signature: true
