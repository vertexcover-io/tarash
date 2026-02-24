# Fallback & Routing

Tarash Gateway can automatically retry with a different provider when a request fails.
Configure a fallback chain on any config object and the SDK handles the rest.

## Basic fallback

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/veo3",
    fallback_configs=[
        VideoGenerationConfig(
            provider="replicate",
            model="google/veo-3",
        ),
    ],
)

request = VideoGenerationRequest(
    prompt="A golden retriever running on a beach at sunset",
    duration_seconds=6,
)

response = generate_video(config, request)
# If Fal fails with a retryable error, Replicate is tried automatically
```

## Chaining multiple fallbacks

`fallback_configs` is a list — chain as many providers as you need:

```python
config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/veo3",
    fallback_configs=[
        VideoGenerationConfig(provider="replicate", model="google/veo-3"),
        VideoGenerationConfig(provider="openai", model="openai/sora-2"),
    ],
)
```

The providers are tried in order: Fal → Replicate → OpenAI. The first success wins.

## What triggers a fallback

Not all errors trigger fallbacks. Tarash classifies errors into two categories:

| Error type | Triggers fallback? | Reason |
|---|:---:|---|
| `GenerationFailedError` | ✅ | Transient; different provider may succeed |
| `TimeoutError` | ✅ | Transient timeout |
| `HTTPConnectionError` | ✅ | Network issue |
| `HTTPError` (429, 500, 502, 503, 504) | ✅ | Rate limit or server error |
| `ValidationError` | ❌ | Your request has an invalid field — fix the request |
| `ContentModerationError` | ❌ | Content policy violation — fix the prompt |
| `HTTPError` (400, 401, 403, 404) | ❌ | Auth or request error — fix the config |

**Non-retryable errors propagate immediately** regardless of how many fallbacks are configured.

## Inspecting which provider was used

Every response includes `execution_metadata` with the full fallback history:

```python
response = generate_video(config, request)

meta = response.execution_metadata
print(meta.total_attempts)          # how many providers were tried
print(meta.fallback_triggered)      # True if a fallback ran
print(meta.successful_attempt)      # 1-based index of the winner
print(meta.configs_in_chain)        # total configs in the chain
print(meta.total_elapsed_seconds)   # wall-clock time across all attempts

# Per-attempt details
for attempt in meta.attempts:
    print(attempt.provider)         # "fal", "replicate", etc.
    print(attempt.model)            # model used
    print(attempt.status)           # "success", "failed"
    print(attempt.error_type)       # exception class name if failed
    print(attempt.error_message)    # error message if failed
    print(attempt.elapsed_seconds)  # time spent on this attempt
```

## Example: log the winning provider

```python
response = generate_video(config, request)
meta = response.execution_metadata

if meta.fallback_triggered:
    winner = meta.attempts[meta.successful_attempt - 1]
    print(f"Primary failed. Used fallback: {winner.provider}/{winner.model}")
else:
    print(f"Primary succeeded: {meta.attempts[0].provider}")
```

## Fallback with different models, same provider

Fallback doesn't have to switch providers. Use it to try a cheaper or faster model
before falling back to a higher-quality one:

```python
config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/veo3.1/fast",      # fast, cheaper
    fallback_configs=[
        VideoGenerationConfig(
            provider="fal",
            model="fal-ai/veo3.1",   # higher quality fallback
        ),
    ],
)
```

## Works with image generation too

`ImageGenerationConfig` supports `fallback_configs` with the same behavior:

```python
from tarash.tarash_gateway import generate_image
from tarash.tarash_gateway.models import ImageGenerationConfig

config = ImageGenerationConfig(
    provider="openai",
    model="dall-e-3",
    fallback_configs=[
        ImageGenerationConfig(provider="fal", model="fal-ai/flux-pro"),
    ],
)
```
