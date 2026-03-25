# Gateway

Public API entry points. Import from `tarash.tarash_gateway`.

```python
from tarash.tarash_gateway import (
    generate_video,
    generate_video_async,
    generate_image,
    generate_image_async,
    generate_tts,
    generate_tts_async,
    generate_sts,
    generate_sts_async,
    health_check,
    health_check_async,
    estimate_cost,
    register_provider,
    register_provider_field_mapping,
    get_provider_field_mapping,
)
```

## Generation Functions

| Function | Sync/Async | `on_progress` receives | Returns |
|---|---|---|---|
| `generate_video(config, request, on_progress=None)` | Sync | `VideoGenerationUpdate` | `VideoGenerationResponse` |
| `generate_video_async(config, request, on_progress=None)` | Async | `VideoGenerationUpdate` | `VideoGenerationResponse` |
| `generate_image(config, request, on_progress=None)` | Sync | `ImageGenerationUpdate` | `ImageGenerationResponse` |
| `generate_image_async(config, request, on_progress=None)` | Async | `ImageGenerationUpdate` | `ImageGenerationResponse` |
| `generate_tts(config, request, on_progress=None)` | Sync | `TTSUpdate` | `TTSResponse` |
| `generate_tts_async(config, request, on_progress=None)` | Async | `TTSUpdate` | `TTSResponse` |
| `generate_sts(config, request, on_progress=None)` | Sync | `STSUpdate` | `STSResponse` |
| `generate_sts_async(config, request, on_progress=None)` | Async | `STSUpdate` | `STSResponse` |

`on_progress` accepts both sync and async callables.

## Health & Cost

| Function | Description | Returns |
|---|---|---|
| `health_check(configs)` | Check provider connectivity (sync) | `dict[str, HealthCheckResult]` |
| `health_check_async(configs)` | Check provider connectivity (async) | `dict[str, HealthCheckResult]` |
| `estimate_cost(config)` | Estimate generation cost for a provider/model | `CostEstimate \| None` |

## Registration

| Function | Description | Returns |
|---|---|---|
| `register_provider(provider, handler)` | Register a custom provider handler | `None` |
| `register_provider_field_mapping(provider, model_mappings)` | Register field mappings for a provider | `None` |
| `get_provider_field_mapping(provider)` | Get registered field mappings | `dict \| None` |

See the [Custom Providers guide](../guides/custom-providers.md) for registration usage.

---

::: tarash.tarash_gateway.api
    options:
      show_source: true
      show_signature_annotations: true
      show_symbol_type_heading: true
      members_order: source
      separate_signature: true
