# Tarash Gateway Docs Overhaul — Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Overhaul the Tarash Gateway documentation to fix naming, improve the home page, add missing guides (fallback/routing, mock provider, custom providers), fix the authentication section, document raw response access, and improve the API reference.

**Architecture:** All changes are Markdown + TOML. No Python code changes. Changes touch 15 existing files and create 3 new guide files. Each task is a self-contained set of edits to one or two files.

**Tech Stack:** Zensical (MkDocs-based), Material theme, mkdocstrings-python for auto-gen sections. Preview with `uv run zensical serve`.

---

## Before You Start

Verify docs build:
```bash
cd /media/aman/external/vertexcover/tarash
uv run zensical serve
```
Open http://127.0.0.1:8000 to preview. Keep it running while making changes.

---

### Task 1: Fix package name and nav in zensical.toml

**Files:**
- Modify: `zensical.toml`

**What to change:**

1. Change `site_name` from `"Tarash"` to `"Tarash Gateway"`
2. Add 3 new entries to the Guides nav section after `image-to-video.md`:

```toml
{ "Fallback & Routing" = "guides/fallback-and-routing.md" },
{ "Mock Provider" = "guides/mock.md" },
{ "Custom Providers" = "guides/custom-providers.md" },
```

**Full updated nav Guides section:**
```toml
{ "Guides" = [
  { "Video Generation" = "guides/video-generation.md" },
  { "Image Generation" = "guides/image-generation.md" },
  { "Image to Video" = "guides/image-to-video.md" },
  { "Fallback & Routing" = "guides/fallback-and-routing.md" },
  { "Mock Provider" = "guides/mock.md" },
  { "Custom Providers" = "guides/custom-providers.md" },
]},
```

**Verify:** `uv run zensical serve` — nav should show "Tarash Gateway" in browser title.

**Commit:**
```bash
git add zensical.toml
git commit -m "📝 Rename site to Tarash Gateway and add 3 new guide nav entries"
```

---

### Task 2: Fix installation page (pip as primary)

**Files:**
- Modify: `docs/getting-started/installation.md`

**What to change:**

Replace the current file content with:

```markdown
# Installation

## Requirements

- Python 3.12+

## Install

Install with all provider dependencies:

```bash
pip install tarash-gateway[all]
```

Or install a single provider:

```bash
pip install tarash-gateway[fal]
pip install tarash-gateway[openai]
pip install tarash-gateway[runway]
```

## Available extras

| Extra | Providers included |
|---|---|
| `fal` | Fal.ai |
| `openai` | OpenAI, Azure OpenAI |
| `runway` | Runway Gen-3 |
| `veo3` | Google Veo3 (Vertex AI) |
| `replicate` | Replicate |
| `all` | All of the above |

## Verify installation

```python
import tarash.tarash_gateway
print("tarash-gateway installed successfully")
```

## Next steps

- [Quickstart](quickstart.md) — generate your first video in 10 lines
- [Authentication](authentication.md) — set up your API keys
```

**Verify:** Docs show pip as primary, no uv at all.

**Commit:**
```bash
git add docs/getting-started/installation.md
git commit -m "📝 Make pip primary install method"
```

---

### Task 3: Fix authentication page (env vars are auto-read)

**Files:**
- Modify: `docs/getting-started/authentication.md`

**Problem:** Current docs show `api_key=os.environ["FAL_KEY"]` — this is wrong. The underlying provider SDKs auto-read their standard env vars when `api_key=None`.

**Replace file content with:**

```markdown
# Authentication

Each provider requires its own API key. You can pass it directly or rely on the
provider's standard environment variable — Tarash passes `None` to the provider
SDK which then reads the env var automatically.

## Option 1: Pass the key directly

```python
from tarash.tarash_gateway.models import VideoGenerationConfig

config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/veo3",
    api_key="fal-xxxxxxxxxxxxxxxx",
)
```

!!! warning
    Never hardcode API keys in source files. Use environment variables or a secrets manager.

## Option 2: Use environment variables (recommended)

Set the key in your environment and omit `api_key` — the provider SDK reads it automatically:

```bash
export FAL_KEY="fal-xxxxxxxxxxxxxxxx"
export OPENAI_API_KEY="sk-xxxxxxxxxxxxxxxx"
```

```python
config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/veo3",
    # api_key omitted — FAL_KEY is read automatically
)
```

No `os.environ` calls needed. The provider SDK handles it.

## Provider env var reference

| Provider | Env var | Where to get it |
|---|---|---|
| Fal.ai | `FAL_KEY` | [fal.ai/dashboard](https://fal.ai/dashboard) |
| OpenAI | `OPENAI_API_KEY` | [platform.openai.com](https://platform.openai.com/api-keys) |
| Azure OpenAI | `AZURE_OPENAI_API_KEY` | Azure Portal |
| Runway | `RUNWAY_API_KEY` | [app.runwayml.com](https://app.runwayml.com) |
| Google (Veo3) | `GOOGLE_APPLICATION_CREDENTIALS` | GCP Service Account JSON path |
| Replicate | `REPLICATE_API_TOKEN` | [replicate.com/account](https://replicate.com/account) |
| Stability AI | `STABILITY_API_KEY` | [platform.stability.ai](https://platform.stability.ai) |
```

**Commit:**
```bash
git add docs/getting-started/authentication.md
git commit -m "📝 Fix env var authentication — auto-read, no os.environ needed"
```

---

### Task 4: Rewrite home page (index.md)

**Files:**
- Modify: `docs/index.md`

**Replace file content with:**

```markdown
# Tarash Gateway

**One API. Every AI video and image provider.**

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(provider="fal", model="fal-ai/veo3")
request = VideoGenerationRequest(prompt="A cat playing piano, cinematic lighting")
response = generate_video(config, request)
print(response.video)  # → URL to generated video
```

Switch providers by changing two words:

```python
config = VideoGenerationConfig(provider="runway", model="gen3a_turbo")
```

---

## Why Tarash Gateway

Sora, Veo3, Runway, Kling, Imagen — every provider ships a different API with different parameters, auth patterns, and polling logic. Tarash Gateway handles the translation. You write one integration and it runs on all of them.

---

## Features

<div class="grid cards" markdown>

-   **Fallback chains**

    If a provider fails or rate-limits, automatically continue with the next one.
    [Learn more →](guides/fallback-and-routing.md)

    ```python
    config = VideoGenerationConfig(
        provider="fal", model="fal-ai/veo3", api_key="...",
        fallback_configs=[
            VideoGenerationConfig(provider="replicate", model="google/veo-3"),
        ],
    )
    ```

-   **Async + progress callbacks**

    Every method has a sync and async variant. Get real-time status updates during generation.

    ```python
    async def on_progress(update):
        print(f"{update.status} — {update.progress_percent}%")

    response = await generate_video_async(config, request, on_progress=on_progress)
    ```

-   **Mock provider**

    Run your full integration locally without hitting any API or spending credits.
    [Learn more →](guides/mock.md)

    ```python
    from tarash.tarash_gateway.mock import MockConfig

    config = VideoGenerationConfig(
        provider="fal", model="fal-ai/veo3",
        mock=MockConfig(enabled=True),
    )
    ```

-   **Raw response access**

    Every response preserves the original provider payload for debugging and
    accessing provider-specific fields.

    ```python
    response.raw_response       # full provider JSON
    response.provider_metadata  # extra provider fields
    ```

-   **Pydantic v2**

    Every request and response is a typed model. No dict guessing. Full IDE support.

-   **Custom providers**

    Plug in your own provider or extend Fal with any model.
    [Learn more →](guides/custom-providers.md)

</div>

---

## Providers

| Provider | Video | Image | Install extra |
|---|:---:|:---:|---|
| [Fal.ai](providers/fal.md) | ✅ | ✅ | `fal` |
| [OpenAI](providers/openai.md) | ✅ | ✅ | `openai` |
| [Azure OpenAI](providers/azure-openai.md) | ✅ | ✅ | `openai` |
| [Google](providers/google.md) | ✅ | ✅ | `veo3` |
| [Runway](providers/runway.md) | ✅ | — | `runway` |
| [Replicate](providers/replicate.md) | ✅ | — | `replicate` |
| [Stability AI](providers/stability.md) | — | ✅ | — |

---

## Adding a new Fal model

Fal hosts hundreds of models. To add support for one that isn't registered yet,
open a GitHub issue — or run this in Claude Code:

```
/add-fal-model fal-ai/your-model-id
```

The skill fetches the model schema, generates field mappers, and writes tests automatically.

---

## Installation

```bash
pip install tarash-gateway[fal]       # single provider
pip install tarash-gateway[all]       # every provider
```

[:fontawesome-solid-rocket: Get Started](getting-started/installation.md){ .md-button .md-button--primary }
[:fontawesome-brands-github: GitHub](https://github.com/vertexcover-io/tarash){ .md-button }
```

**Verify:** Home page has prominent feature grid cards, fallback/mock/custom-providers all linked, GitHub issue flow mentioned.

**Commit:**
```bash
git add docs/index.md
git commit -m "📝 Overhaul home page — prominent features, provider table, Fal model callout"
```

---

### Task 5: Fix providers/index.md (remove duplicate model tables)

**Files:**
- Modify: `docs/providers/index.md`

**Problem:** The page has a 15-row "Model Quick Reference" table that duplicates every individual provider page. Remove it, keep only the capability matrix and the switching example.

**Replace file content with:**

```markdown
# Providers

Tarash Gateway provides a unified interface to generate videos and images across multiple AI providers.
Change `provider` and `model` in your config — nothing else in your code changes.

## Supported providers

| Provider | Video | Image | Image-to-Video | Async | Install extra |
|---|:---:|:---:|:---:|:---:|---|
| [OpenAI](openai.md) | ✅ | ✅ | ✅ | ✅ | `openai` |
| [Azure OpenAI](azure-openai.md) | ✅ | ✅ | ✅ | ✅ | `openai` |
| [Fal.ai](fal.md) | ✅ | ✅ | ✅ | ✅ | `fal` |
| [Google](google.md) | ✅ | ✅ | ✅ | ✅ | `veo3` |
| [Runway](runway.md) | ✅ | — | ✅ | ✅ | `runway` |
| [Replicate](replicate.md) | ✅ | — | ✅ | ✅ | `replicate` |
| [Stability AI](stability.md) | — | ✅ | — | ✅ | — |

## Switching providers

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

request = VideoGenerationRequest(
    prompt="A cat playing piano, cinematic lighting",
    duration_seconds=5,
    aspect_ratio="16:9",
)

# Use Fal.ai
config = VideoGenerationConfig(provider="fal", model="fal-ai/veo3")
response = generate_video(config, request)

# Switch to Runway — same request, one line change
config = VideoGenerationConfig(provider="runway", model="gen3a_turbo")
response = generate_video(config, request)
```

## Fallback chains

Configure automatic fallback to a backup provider if the primary fails or rate-limits:

```python
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
```

See the [Fallback & Routing guide](../guides/fallback-and-routing.md) for full details on retry behavior, execution metadata, and chaining multiple providers.

## Adding a new Fal model

Fal hosts hundreds of models. To add one that isn't registered:

```
/add-fal-model fal-ai/your-model-id
```

Or [open a GitHub issue](https://github.com/vertexcover-io/tarash/issues).
```

**Commit:**
```bash
git add docs/providers/index.md
git commit -m "📝 Simplify providers overview — remove duplicate model tables"
```

---

### Task 6: Update providers/fal.md (add all registered models)

**Files:**
- Modify: `docs/providers/fal.md`

**What to change:**

1. Remove the `## Installation` snippet (redundant — already on installation page)
2. Replace the `## Video Models` table with a complete table of all models currently in `FAL_MODEL_REGISTRY` (from `packages/tarash-gateway/src/tarash/tarash_gateway/providers/fal.py`)

**Updated `## Video Models` section — replace the existing table with:**

```markdown
## Video Models

Model lookup uses **prefix matching**: `fal-ai/veo3.1/fast` matches the `fal-ai/veo3.1` registry entry,
so any sub-variant automatically inherits the right field mappers.

| Model prefix | Duration | Image-to-Video | Notes |
|---|---|:---:|---|
| `fal-ai/veo3` | 4s, 6s, 8s | ✅ | Audio; first/last frame; extend-video |
| `fal-ai/veo3.1` | 4s, 6s, 7s, 8s | ✅ | Latest Veo; fast variant via `/fast`; extend-video via `/fast/extend-video` |
| `fal-ai/minimax` | 6s, 10s | ✅ | Hailuo series; prompt optimizer support |
| `fal-ai/kling-video/v2.6` | 5s, 10s | ✅ | Motion control, cfg_scale, last-frame pinning |
| `fal-ai/kling-video/o1` | 5s, 10s | ✅ | Reference-to-video, video edit, start/end frame |
| `fal-ai/sora-2` | 4s, 8s, 12s | ✅ | Sora via Fal; remix via `/video-to-video/remix` |
| `wan/v2.6/` | configurable | ✅ | Wan v2.6; text, image, reference-to-video |
| `fal-ai/wan-25-preview/` | configurable | ✅ | Wan v2.5 preview |
| `fal-ai/wan/v2.2-14b/animate/` | — | ✅ | Wan animate: video+image motion control |
| `fal-ai/bytedance/seedance` | 2s–12s | ✅ | ByteDance Seedance v1/v1.5; reference-to-video |
| `fal-ai/pixverse/v5` | 5s, 8s, 10s | ✅ | Pixverse v5; transition, effects, swap |
| `fal-ai/pixverse/v5.5` | 5s, 8s, 10s | ✅ | Pixverse v5.5; same API as v5 |
| `fal-ai/pixverse/swap` | — | ✅ | Pixverse swap variant |
| Any other `fal-ai/*` | — | ✅ | Generic field mappers (prompt, seed, aspect_ratio) |

Any Fal model not in this table gets **generic mappers** (prompt passthrough + common fields). For full support with model-specific parameters, [add the model](https://github.com/vertexcover-io/tarash/issues) or use `/add-fal-model`.
```

**Remove from the file:** The entire `## Installation` section (lines 27-31 in current fal.md).

**Verify:** fal.md shows all 13 model prefixes. No `## Installation` section.

**Commit:**
```bash
git add docs/providers/fal.md
git commit -m "📝 Add all registered Fal models to provider page"
```

---

### Task 7: Remove redundant installation snippets from other provider pages

**Files:**
- Modify: `docs/providers/openai.md`, `docs/providers/azure-openai.md`, `docs/providers/google.md`, `docs/providers/runway.md`, `docs/providers/replicate.md`, `docs/providers/stability.md`

**In each file:** Remove the `## Installation` section that shows `pip install tarash-gateway[x]`. It's already on the installation page. Keep everything else.

**For each file**, find and delete a block like this:
```markdown
## Installation

```bash
pip install tarash-gateway[fal]
# or
uv add tarash-gateway[fal]
```
```

**Commit:**
```bash
git add docs/providers/openai.md docs/providers/azure-openai.md docs/providers/google.md docs/providers/runway.md docs/providers/replicate.md docs/providers/stability.md
git commit -m "📝 Remove redundant installation snippets from provider pages"
```

---

### Task 8: Add config param table + raw response section to video-generation.md

**Files:**
- Modify: `docs/guides/video-generation.md`

**Change 1:** Add a `## Config parameters` section after the existing `## Basic usage` block:

```markdown
## Config parameters

```python
config = VideoGenerationConfig(
    provider="fal",          # required
    model="fal-ai/veo3",     # required
    api_key="...",           # optional — reads FAL_KEY env var if omitted
)
```

| Parameter | Type | Required | Default | Description |
|---|---|:---:|---|---|
| `provider` | `str` | ✅ | — | Provider ID: `"fal"`, `"openai"`, `"runway"`, `"google"`, `"replicate"`, `"stability"` |
| `model` | `str` | ✅ | — | Model ID, e.g. `"fal-ai/veo3"`, `"openai/sora-2"` |
| `api_key` | `str \| None` | — | `None` | API key; omit to use provider's standard env var |
| `base_url` | `str \| None` | — | `None` | Override the provider's base API URL |
| `api_version` | `str \| None` | — | `None` | API version string (required for Azure OpenAI) |
| `timeout` | `int` | — | `600` | Max seconds to wait for generation to complete |
| `max_poll_attempts` | `int` | — | `120` | Max polling iterations before timeout |
| `poll_interval` | `int` | — | `5` | Seconds between status polls |
| `mock` | `MockConfig \| None` | — | `None` | Enable mock generation for testing |
| `fallback_configs` | `list[VideoGenerationConfig] \| None` | — | `None` | Ordered fallback provider chain |
| `provider_config` | `dict` | — | `{}` | Provider-specific config (e.g. GCP project for Google Vertex AI) |
```

**Change 2:** Add a `## Accessing the raw response` section at the end of the file:

```markdown
## Accessing the raw response

Every response preserves the original, unmodified provider payload:

```python
response = generate_video(config, request)

# Full provider JSON — useful for debugging or accessing provider-specific fields
print(response.raw_response)

# Additional provider fields not in the standard interface
print(response.provider_metadata)

# Example: extract Fal-specific fields
fal_request_id = response.raw_response.get("request_id")
```

`raw_response` is always populated, even on failure. Use it when you need a field
that isn't in the standard `VideoGenerationResponse` interface.
```

**Commit:**
```bash
git add docs/guides/video-generation.md
git commit -m "📝 Add VideoGenerationConfig param table and raw response section"
```

---

### Task 9: Add config param table + raw response section to image-generation.md

**Files:**
- Modify: `docs/guides/image-generation.md`

**Change 1:** Add a `## Config parameters` section after `## Basic usage`:

```markdown
## Config parameters

| Parameter | Type | Required | Default | Description |
|---|---|:---:|---|---|
| `provider` | `str` | ✅ | — | Provider ID: `"openai"`, `"fal"`, `"stability"`, `"google"` |
| `model` | `str` | ✅ | — | Model ID, e.g. `"dall-e-3"`, `"fal-ai/flux-pro"` |
| `api_key` | `str \| None` | — | `None` | API key; omit to use provider's standard env var |
| `base_url` | `str \| None` | — | `None` | Override the provider's base API URL |
| `api_version` | `str \| None` | — | `None` | API version string (required for Azure OpenAI) |
| `timeout` | `int` | — | `120` | Max seconds to wait for generation |
| `max_poll_attempts` | `int` | — | `60` | Max polling iterations |
| `poll_interval` | `int` | — | `2` | Seconds between polls |
| `mock` | `MockConfig \| None` | — | `None` | Enable mock generation for testing |
| `fallback_configs` | `list[ImageGenerationConfig] \| None` | — | `None` | Ordered fallback provider chain |
| `provider_config` | `dict` | — | `{}` | Provider-specific config |
```

**Change 2:** Add `## Accessing the raw response` at the end:

```markdown
## Accessing the raw response

```python
response = generate_image(config, request)

# Full original provider response
print(response.raw_response)

# Extra provider fields not in the standard interface
print(response.provider_metadata)
```
```

**Change 3:** Fix the `image_list` row in the existing Request parameters table — current docs say `list[MediaType] | None`, it should be `list[ImageType]` to match the actual model.

**Commit:**
```bash
git add docs/guides/image-generation.md
git commit -m "📝 Add ImageGenerationConfig param table and raw response section"
```

---

### Task 10: Create guides/fallback-and-routing.md

**Files:**
- Create: `docs/guides/fallback-and-routing.md`

**Full file content:**

````markdown
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
````

**Commit:**
```bash
git add docs/guides/fallback-and-routing.md
git commit -m "📝 Add Fallback & Routing guide"
```

---

### Task 11: Create guides/mock.md

**Files:**
- Create: `docs/guides/mock.md`

**Full file content:**

````markdown
# Mock Provider

The mock provider lets you run your full video and image generation pipeline locally
without hitting any real API. No API keys needed, no credits spent.

## Basic usage

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest
from tarash.tarash_gateway.mock import MockConfig

config = VideoGenerationConfig(
    provider="fal",        # provider field still required
    model="fal-ai/veo3",   # model still required
    mock=MockConfig(enabled=True),
)

request = VideoGenerationRequest(prompt="A cat playing piano")

response = generate_video(config, request)
print(response.video)       # → URL to a stock mock video
print(response.is_mock)     # → True
```

When `mock=MockConfig(enabled=True)` is set, Tarash bypasses the real provider entirely
and returns a pre-configured mock response. The `provider` and `model` values are
recorded in the response but no real API call is made.

## Default behavior

By default, `MockConfig(enabled=True)` returns a randomly selected stock video
from a built-in library matched to your request. The response is a valid
`VideoGenerationResponse` with `is_mock=True`.

## Customizing mock responses

Control exactly what the mock returns using `MockResponse`:

```python
from tarash.tarash_gateway.mock import MockConfig, MockResponse

config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/veo3",
    mock=MockConfig(
        enabled=True,
        responses=[
            MockResponse(
                weight=1.0,
                video_url="https://example.com/my-test-video.mp4",
            ),
        ],
    ),
)
```

## Injecting errors for testing

Test your error handling by configuring the mock to raise specific exceptions:

```python
from tarash.tarash_gateway.exceptions import GenerationFailedError
from tarash.tarash_gateway.mock import MockConfig, MockResponse

config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/veo3",
    mock=MockConfig(
        enabled=True,
        responses=[
            MockResponse(weight=1.0, error=GenerationFailedError("Simulated failure")),
        ],
    ),
)

try:
    response = generate_video(config, request)
except GenerationFailedError as e:
    print(f"Caught: {e}")  # → "Caught: Simulated failure"
```

## Weighted responses

Simulate flaky providers by mixing success and error responses with weights:

```python
mock = MockConfig(
    enabled=True,
    responses=[
        MockResponse(weight=0.8),                                              # 80% succeed
        MockResponse(weight=0.2, error=GenerationFailedError("rate limited")), # 20% fail
    ],
)
```

## Mock + fallback chains

Combine mock with fallback to test your fallback logic:

```python
config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/veo3",
    mock=MockConfig(
        enabled=True,
        responses=[MockResponse(weight=1.0, error=GenerationFailedError("forced fail"))],
    ),
    fallback_configs=[
        VideoGenerationConfig(
            provider="replicate",
            model="google/veo-3",
            mock=MockConfig(enabled=True),  # fallback mock succeeds
        ),
    ],
)

response = generate_video(config, request)
assert response.execution_metadata.fallback_triggered is True
```

## Image generation mock

`MockConfig` works the same way for image generation:

```python
from tarash.tarash_gateway import generate_image
from tarash.tarash_gateway.models import ImageGenerationConfig

config = ImageGenerationConfig(
    provider="openai",
    model="dall-e-3",
    mock=MockConfig(enabled=True),
)
```

## API reference

See [Mock Provider API reference](../api-reference/mock.md) for the full `MockConfig`
and `MockResponse` field reference.
````

**Commit:**
```bash
git add docs/guides/mock.md
git commit -m "📝 Add Mock Provider guide"
```

---

### Task 12: Create guides/custom-providers.md

**Files:**
- Create: `docs/guides/custom-providers.md`

**Full file content:**

````markdown
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
and writes tests — all automatically. For contributing the model upstream, [open a GitHub issue](https://github.com/vertexcover-io/tarash/issues).

## Implementing a custom provider

Implement the `ProviderHandler` protocol from `tarash.tarash_gateway.models`:

```python
from tarash.tarash_gateway.models import (
    ProviderHandler,
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

Do this once at application startup. After that, use it like any built-in provider:

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

## Registering custom Fal field mappers

If your provider is Fal-based but needs specific field mappings for a model not yet
in the registry, register field mappers directly:

```python
from tarash.tarash_gateway import register_provider_field_mapping
from tarash.tarash_gateway.providers.field_mappers import (
    FieldMapper,
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
will use your mappers automatically.

## Notes

- `register_provider()` and `register_provider_field_mapping()` are global — call once at startup
- Registering with the same name as a built-in provider overwrites it
- The `ProviderHandler` protocol requires all four methods (`generate_video`, `generate_video_async`, `generate_image`, `generate_image_async`). Raise `NotImplementedError` for unsupported modalities
````

**Commit:**
```bash
git add docs/guides/custom-providers.md
git commit -m "📝 Add Custom Providers guide"
```

---

### Task 13: Improve api-reference/models.md

**Files:**
- Modify: `docs/api-reference/models.md`

**Replace file content with a manual intro section followed by the existing auto-gen directive:**

````markdown
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
| `provider_config` | `dict` | — | `{}` | Extra provider-specific config (e.g. `{"project": "gcp-project"}` for Google) |

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
| `total_elapsed_seconds` | `float` | Wall-clock time across all attempts (property) |

---

## Full auto-generated reference

::: tarash.tarash_gateway.models
    options:
      show_source: false
      show_signature_annotations: true
      show_symbol_type_heading: true
      members_order: source
      separate_signature: true
````

**Commit:**
```bash
git add docs/api-reference/models.md
git commit -m "📝 Expand models API reference with full manual parameter tables"
```

---

### Task 14: Improve api-reference/exceptions.md

**Files:**
- Modify: `docs/api-reference/exceptions.md`

**Replace file content with:**

````markdown
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
````

**Commit:**
```bash
git add docs/api-reference/exceptions.md
git commit -m "📝 Add retryability table and error handling example to exceptions reference"
```

---

### Task 15: Add manual intro to api-reference/gateway.md

**Files:**
- Modify: `docs/api-reference/gateway.md`

**Replace file content with:**

````markdown
# Gateway

Public API entry points. Import from `tarash.tarash_gateway`.

## Function signatures

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

| Function | Sync/Async | Returns |
|---|---|---|
| `generate_video(config, request, on_progress=None)` | Sync | `VideoGenerationResponse` |
| `generate_video_async(config, request, on_progress=None)` | Async | `VideoGenerationResponse` |
| `generate_image(config, request, on_progress=None)` | Sync | `ImageGenerationResponse` |
| `generate_image_async(config, request, on_progress=None)` | Async | `ImageGenerationResponse` |
| `register_provider(name, handler)` | — | `None` |
| `register_provider_field_mapping(provider, model_mappings)` | — | `None` |
| `get_provider_field_mapping(provider)` | — | `dict \| None` |

See the [Custom Providers guide](../guides/custom-providers.md) for registration usage.

---

::: tarash.tarash_gateway.api
    options:
      show_source: true
      show_signature_annotations: true
      show_symbol_type_heading: true
      members_order: source
      separate_signature: true
````

**Commit:**
```bash
git add docs/api-reference/gateway.md
git commit -m "📝 Add manual intro and function summary to gateway API reference"
```

---

## Final verification

```bash
uv run zensical serve
```

Checklist:
- [ ] Site title shows "Tarash Gateway"
- [ ] Home page has feature grid cards (Fallback, Async, Mock, Raw response, Custom providers)
- [ ] Installation page shows pip only (no uv)
- [ ] Authentication shows env var auto-read pattern (no `os.environ`)
- [ ] Providers overview has no model tables (only capability matrix)
- [ ] Fal.ai page lists all 13 model prefixes including Wan, ByteDance Seedance, Pixverse
- [ ] Guides nav has 6 items including Fallback & Routing, Mock Provider, Custom Providers
- [ ] Each new guide loads without 404
- [ ] Models API reference has manual parameter tables for all 4 models
- [ ] Exceptions page has retryability table
- [ ] Gateway page has function summary table
