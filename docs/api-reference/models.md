# Models

Core Pydantic models for requests, responses, and configuration.

---

## VideoGenerationConfig

Configuration passed to `generate_video()` and `generate_video_async()`.

| Field | Type | Required | Default | Description |
|---|---|:---:|---|---|
| `provider` | `str` | ✅ | — | Provider ID: `"fal"`, `"openai"`, `"runway"`, `"google"`, `"replicate"`, `"stability"` |
| `model` | `str` | ✅ | — | Model ID, e.g. `"fal-ai/veo3"`, `"openai/sora-2"` |
| `api_key` | `str \| None` | — | `None` | API key; omit to auto-read provider env var |
| `base_url` | `str \| None` | — | `None` | Override provider base URL |
| `api_version` | `str \| None` | — | `None` | API version (required for Azure OpenAI) |
| `timeout` | `int` | — | `600` | Max seconds to wait for completion |
| `max_poll_attempts` | `int` | — | `120` | Max polling iterations |
| `poll_interval` | `int` | — | `5` | Seconds between polls |
| `mock` | `MockConfig \| None` | — | `None` | Enable mock generation |
| `fallback_configs` | `list[VideoGenerationConfig] \| None` | — | `None` | Fallback chain |
| `provider_config` | `dict` | — | `{}` | Extra provider-specific config (e.g. `{"gcp_project": "my-project"}` for Google Vertex AI) |

---

## VideoGenerationRequest

Parameters for a video generation request. Unknown kwargs are automatically captured into `extra_params`.

| Field | Type | Required | Default | Description |
|---|---|:---:|---|---|
| `prompt` | `str` | ✅ | — | Text description of the video |
| `duration_seconds` | `int \| None` | — | `None` | Duration in seconds (provider may round) |
| `resolution` | `"360p" \| "480p" \| "720p" \| "1080p" \| "4k" \| None` | — | `None` | Requested resolution |
| `aspect_ratio` | `"16:9" \| "9:16" \| "1:1" \| "4:3" \| "21:9" \| None` | — | `None` | Requested aspect ratio |
| `generate_audio` | `bool \| None` | — | `None` | Generate audio alongside video |
| `image_list` | `list[ImageType]` | — | `[]` | Input images with semantic roles |
| `video` | `MediaType \| None` | — | `None` | Input video for extend or remix |
| `seed` | `int \| None` | — | `None` | Reproducibility seed |
| `number_of_videos` | `int` | — | `1` | Number of variants to generate |
| `negative_prompt` | `str \| None` | — | `None` | Elements to avoid |
| `enhance_prompt` | `bool \| None` | — | `None` | Allow provider to enhance the prompt |
| `extra_params` | `dict` | — | `{}` | Provider/model-specific parameters |

---

## VideoGenerationResponse

| Field | Type | Description |
|---|---|---|
| `request_id` | `str` | Tarash-assigned unique ID |
| `video` | `MediaType` | Generated video (URL, base64, or bytes) |
| `status` | `"completed" \| "failed"` | Final status |
| `duration` | `float \| None` | Actual video duration in seconds |
| `resolution` | `str \| None` | Actual resolution |
| `aspect_ratio` | `str \| None` | Actual aspect ratio |
| `content_type` | `str \| None` | MIME type (e.g. `"video/mp4"`) |
| `audio_url` | `str \| None` | Generated audio URL if requested |
| `is_mock` | `bool` | True if produced by mock provider |
| `raw_response` | `dict` | Unmodified provider response |
| `provider_metadata` | `dict` | Extra provider fields not in standard interface |
| `execution_metadata` | `ExecutionMetadata \| None` | Timing and fallback details |

---

## ImageGenerationConfig

| Field | Type | Required | Default | Description |
|---|---|:---:|---|---|
| `provider` | `str` | ✅ | — | Provider ID |
| `model` | `str` | ✅ | — | Model ID, e.g. `"dall-e-3"`, `"fal-ai/flux-pro"` |
| `api_key` | `str \| None` | — | `None` | API key; omit to auto-read env var |
| `base_url` | `str \| None` | — | `None` | Override base URL |
| `api_version` | `str \| None` | — | `None` | API version |
| `timeout` | `int` | — | `120` | Max seconds to wait |
| `max_poll_attempts` | `int` | — | `60` | Max polling iterations |
| `poll_interval` | `int` | — | `2` | Seconds between polls |
| `mock` | `MockConfig \| None` | — | `None` | Enable mock |
| `fallback_configs` | `list[ImageGenerationConfig] \| None` | — | `None` | Fallback chain |
| `provider_config` | `dict` | — | `{}` | Extra provider config |

---

## ImageGenerationRequest

| Field | Type | Required | Default | Description |
|---|---|:---:|---|---|
| `prompt` | `str` | ✅ | — | Text description of the image |
| `negative_prompt` | `str \| None` | — | `None` | Elements to avoid |
| `size` | `str \| None` | — | `None` | Output size, e.g. `"1024x1024"` |
| `quality` | `str \| None` | — | `None` | Quality level, e.g. `"standard"`, `"hd"` |
| `style` | `str \| None` | — | `None` | Style, e.g. `"vivid"`, `"natural"` |
| `n` | `int \| None` | — | `None` | Number of images |
| `aspect_ratio` | `AspectRatio \| None` | — | `None` | Alternative to explicit size |
| `image_list` | `list[ImageType]` | — | `[]` | Input images for img2img/inpainting |
| `mask_image` | `MediaType \| None` | — | `None` | Inpainting mask (white = edit area) |
| `seed` | `int \| None` | — | `None` | Reproducibility seed |
| `extra_params` | `dict` | — | `{}` | Provider-specific parameters |

---

## ImageGenerationResponse

| Field | Type | Description |
|---|---|---|
| `request_id` | `str` | Tarash-assigned unique ID |
| `images` | `list[str]` | Generated images (URLs or base64) |
| `status` | `"completed" \| "failed"` | Final status |
| `content_type` | `str \| None` | MIME type (default `"image/png"`) |
| `is_mock` | `bool` | True if produced by mock provider |
| `revised_prompt` | `str \| None` | Prompt as revised by provider (OpenAI) |
| `raw_response` | `dict` | Unmodified provider response |
| `provider_metadata` | `dict` | Extra provider fields |
| `execution_metadata` | `ExecutionMetadata \| None` | Timing and fallback details |

---

## ExecutionMetadata

Attached to every response by the orchestrator. Inspect it to understand which provider
succeeded and how long each attempt took.

| Field | Type | Description |
|---|---|---|
| `total_attempts` | `int` | Number of providers tried |
| `successful_attempt` | `int \| None` | 1-based index of the winning attempt |
| `attempts` | `list[AttemptMetadata]` | Per-attempt details |
| `fallback_triggered` | `bool` | True if any fallback ran |
| `configs_in_chain` | `int` | Total configs in the fallback chain |
| `total_elapsed_seconds` | `float` | Wall-clock time across all attempts (computed property) |

---

## Full auto-generated reference

::: tarash.tarash_gateway.models
    options:
      show_source: false
      show_signature_annotations: true
      show_symbol_type_heading: true
      members_order: source
      separate_signature: true
