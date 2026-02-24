# Exceptions

All exceptions inherit from `TarashException`.

## Retryability

The orchestrator uses this table to decide whether to try the next fallback:

| Exception | Retryable | Typical cause |
|---|:---:|---|
| `GenerationFailedError` | ✅ | Provider failed, timeout, cancellation |
| `TimeoutError` | ✅ | Polling timed out |
| `HTTPConnectionError` | ✅ | Network failure |
| `HTTPError` (429, 500–504) | ✅ | Rate limit or server error |
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
    HTTPError,
)

try:
    response = generate_video(config, request)
except ContentModerationError as e:
    print(f"Content moderation: {e.message}")
except ValidationError as e:
    print(f"Bad request: {e.message}")
except GenerationFailedError as e:
    print(f"Generation failed: {e.message}, provider={e.provider}")
except TarashException as e:
    print(f"Other error: {e.message}")
```

All exceptions expose: `message`, `provider`, `model`, `request_id`, `raw_response`.

---

::: tarash.tarash_gateway.exceptions
    options:
      show_source: false
      show_signature_annotations: true
      show_symbol_type_heading: true
      members_order: source
      separate_signature: true
