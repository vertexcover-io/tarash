# Mock Provider

Mock provider for testing without real API calls. Set `VideoGenerationConfig.mock` to a `MockConfig` to intercept generation requests. See the [Mock guide](../guides/mock.md) for usage patterns.

## MockConfig

| Field | Type | Required | Default | Description |
|---|---|:---:|---|---|
| `enabled` | `bool` | ✅ | — | Set to `True` to activate the mock provider |
| `responses` | `list[MockResponse] \| None` | — | single auto-success | Pool of weighted outcomes; one is selected per request |
| `polling` | `MockPollingConfig \| None` | — | `None` | Simulated polling sequence; if `None`, `on_progress` is never called |

## MockResponse

One entry in the `MockConfig.responses` pool. Exactly one of `mock_response`, `output_video`, or `error` should be set; if none are set the mock auto-selects a sample video matching the request.

| Field | Type | Default | Description |
|---|---|---|---|
| `weight` | `float` | `1.0` | Relative selection probability (must be positive) |
| `mock_response` | `VideoGenerationResponse \| None` | `None` | Return this exact response (`request_id` and `is_mock` are overwritten) |
| `output_video` | `MediaType \| None` | `None` | Video URL, base64, or bytes to use as the result |
| `output_video_type` | `"url" \| "content"` | `"url"` | `"content"` downloads the video and returns bytes instead of the URL |
| `error` | `Exception \| None` | `None` | Raise this exception instead of returning a response |

## MockPollingConfig

Controls the simulated polling sequence fired to the `on_progress` callback.

| Field | Type | Default | Description |
|---|---|---|---|
| `enabled` | `bool` | `True` | Enable polling simulation |
| `status_sequence` | `list[StatusType]` | `["queued", "processing", "completed"]` | Ordered statuses emitted to the callback |
| `delay_between_updates` | `float` | `0.5` | Seconds to sleep between each simulated update |
| `progress_percentages` | `list[int] \| None` | `None` | Per-step progress values; length must match `status_sequence` |
| `custom_updates` | `list[dict] \| None` | `None` | Per-step payload merged into each `VideoGenerationUpdate`; length must match `status_sequence` |
