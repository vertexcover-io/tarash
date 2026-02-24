# Exceptions

All exceptions inherit from `TarashException`. Every exception exposes:
`message`, `provider`, `model`, `request_id`, `raw_response`.

## Retryability

The orchestrator uses this table to decide whether to try the next fallback:

| Exception | Retryable | Typical cause |
|---|:---:|---|
| `GenerationFailedError` | ✅ | Provider failed, timeout, cancellation |
| `TimeoutError` | ✅ | Polling timed out (exposes `timeout_seconds`) |
| `HTTPConnectionError` | ✅ | Network failure |
| `HTTPError` (429, 500–504) | ✅ | Rate limit or server error (exposes `status_code`) |
| `ValidationError` | ❌ | Invalid request parameters |
| `ContentModerationError` | ❌ | Prompt violates content policy |
| `HTTPError` (400, 401, 403, 404) | ❌ | Bad request or auth error |

Non-retryable errors are raised immediately regardless of fallback configuration.

## Catching errors

```python
from tarash.tarash_gateway.exceptions import (
    TarashException,
    ValidationError,
    ContentModerationError,
    GenerationFailedError,
    TimeoutError,
    HTTPConnectionError,
    HTTPError,
)

try:
    response = generate_video(config, request)
except ContentModerationError as e:
    print(f"Content moderation: {e.message}")
except ValidationError as e:
    print(f"Bad request: {e.message}")
except TimeoutError as e:
    print(f"Timed out after {e.timeout_seconds}s, provider={e.provider}")
except HTTPConnectionError as e:
    print(f"Network failure: {e.message}, provider={e.provider}")
except GenerationFailedError as e:
    print(f"Generation failed: {e.message}, provider={e.provider}")
except TarashException as e:
    print(f"Other error: {e.message}")
```

## Utilities

`is_retryable_error(error)` — returns `True` for errors that trigger fallback. Used internally by the orchestrator; useful when building custom retry logic.

`handle_video_generation_errors` — decorator applied to provider `generate_video` / `generate_video_async` methods. Lets `TarashException` and Pydantic `ValidationError` propagate unchanged; wraps any other exception in `TarashException` with full traceback in `raw_response`. See the [Custom Providers guide](../guides/custom-providers.md).

---

::: tarash.tarash_gateway.exceptions
    options:
      show_source: false
      show_signature_annotations: true
      show_symbol_type_heading: true
      members_order: source
      separate_signature: true
