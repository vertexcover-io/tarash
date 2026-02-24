# Tarash Gateway Docs Overhaul — Design

**Date:** 2026-02-24
**Status:** Approved

---

## Problem

The current Tarash Gateway documentation has several gaps identified by user review:

1. Package referred to as "Tarash" instead of "Tarash Gateway"
2. Home page features are buried and not prominent
3. Installation docs show uv as primary; should be pip
4. Authentication docs incorrectly show manual `os.environ` reads — env vars are auto-read by the SDK
5. Provider pages (especially Fal) are missing many supported models
6. Duplicate model lists between providers/index.md and individual provider pages
7. No dedicated guide for fallback chains / routing
8. No dedicated guide for the mock provider
9. No guide for adding custom providers
10. Raw response access (`raw_response`, `provider_metadata`) is undocumented
11. API reference parameter tables are thin or auto-generated only

---

## Scope

### Files to Modify

| File | Changes |
|---|---|
| `zensical.toml` | Rename site, add 3 new guides to nav |
| `docs/index.md` | Better hero, prominent features grid, FAL model callout, GitHub issue flow |
| `docs/getting-started/installation.md` | pip as primary, uv secondary |
| `docs/getting-started/authentication.md` | Fix env var section: auto-read, not manual `os.environ` |
| `docs/providers/index.md` | Remove model quick-reference tables (reduce duplication), keep capability matrix |
| `docs/providers/fal.md` | Add missing models (Kling v3, more variants), remove redundant installation snippet |
| `docs/providers/*.md` | Remove installation snippets (each provider page has pip install — redundant) |
| `docs/guides/video-generation.md` | Add VideoGenerationConfig param table, add raw response section |
| `docs/guides/image-generation.md` | Add ImageGenerationConfig param table, add raw response section |
| `docs/api-reference/models.md` | Replace pure auto-gen with manual param tables (required vs optional) |
| `docs/api-reference/exceptions.md` | Add retryability table |
| `docs/api-reference/gateway.md` | Add manual intro + parameter summary before auto-gen |

### New Files

| File | Contents |
|---|---|
| `docs/guides/fallback-and-routing.md` | Fallback chains, retry behavior, execution metadata |
| `docs/guides/mock.md` | MockConfig usage, testing patterns |
| `docs/guides/custom-providers.md` | ProviderHandler protocol, register_provider(), custom Fal models in code |

---

## Design Decisions

### 1. Package naming
Replace all occurrences of "Tarash" used as the package name with "Tarash Gateway". The parent monorepo is "tarash" but the user-facing SDK is "tarash-gateway".

### 2. Environment variables
Current docs show:
```python
config = VideoGenerationConfig(provider="fal", api_key=os.environ["FAL_KEY"])
```

Correct behavior: if `api_key` is omitted, the underlying provider SDK reads its standard env var automatically. Docs should show:
```python
# api_key omitted — FAL_KEY env var is read automatically
config = VideoGenerationConfig(provider="fal", model="fal-ai/veo3")
```
The env var table stays, but the code example changes.

### 3. Reduce duplication
`providers/index.md` currently has a 15-row model quick-reference table. Each individual provider page also lists models. Solution: remove the model tables from `providers/index.md`, keep only the capability matrix and switching example. Individual provider pages own their model lists.

### 4. Fallback & Routing guide
This is the biggest missing piece. Needs:
- What makes an error retryable vs non-retryable (table)
- 3-level fallback chain example
- How to inspect `execution_metadata` after a call
- Real-world use cases (rate limits, outages, cost tiers)

### 5. API reference models page
Current `api-reference/models.md` is a single `:::` auto-gen directive. Replace with manual tables for the 4 main models, clearly marking required vs optional fields. The auto-gen can stay as a footer reference.

---

## Nav Structure After Changes

```toml
nav = [
  { "Home" = "index.md" },
  { "Getting Started" = [
    { "Installation" = "getting-started/installation.md" },
    { "Quickstart" = "getting-started/quickstart.md" },
    { "Authentication" = "getting-started/authentication.md" },
  ]},
  { "Providers" = [
    { "Overview" = "providers/index.md" },
    { "OpenAI" = "providers/openai.md" },
    { "Azure OpenAI" = "providers/azure-openai.md" },
    { "Fal.ai" = "providers/fal.md" },
    { "Google" = "providers/google.md" },
    { "Runway" = "providers/runway.md" },
    { "Replicate" = "providers/replicate.md" },
    { "Stability AI" = "providers/stability.md" },
  ]},
  { "Guides" = [
    { "Video Generation" = "guides/video-generation.md" },
    { "Image Generation" = "guides/image-generation.md" },
    { "Image to Video" = "guides/image-to-video.md" },
    { "Fallback & Routing" = "guides/fallback-and-routing.md" },   # NEW
    { "Mock Provider" = "guides/mock.md" },                         # NEW
    { "Custom Providers" = "guides/custom-providers.md" },          # NEW
  ]},
  { "API Reference" = [
    { "Gateway" = "api-reference/gateway.md" },
    { "Models" = "api-reference/models.md" },
    { "Exceptions" = "api-reference/exceptions.md" },
    { "Mock Provider" = "api-reference/mock.md" },
  ]},
]
```

---

## Key Content Details

### VideoGenerationConfig full parameter table (for models.md and video-generation.md)

| Parameter | Type | Required | Default | Description |
|---|---|:---:|---|---|
| `provider` | `str` | ✅ | — | Provider ID: `"fal"`, `"openai"`, `"runway"`, etc. |
| `model` | `str` | ✅ | — | Model ID, e.g. `"fal-ai/veo3"` |
| `api_key` | `str \| None` | — | `None` | API key; omit to use env var |
| `base_url` | `str \| None` | — | `None` | Override provider base URL |
| `api_version` | `str \| None` | — | `None` | Required for Azure OpenAI |
| `timeout` | `int` | — | `600` | Max seconds to wait |
| `max_poll_attempts` | `int` | — | `120` | Max polling iterations |
| `poll_interval` | `int` | — | `5` | Seconds between polls |
| `mock` | `MockConfig \| None` | — | `None` | Enable mock generation |
| `fallback_configs` | `list[VideoGenerationConfig] \| None` | — | `None` | Fallback provider chain |
| `provider_config` | `dict` | — | `{}` | Provider-specific config (e.g. GCP project for Google) |

### Fal.ai missing models to add

The following models are missing from `providers/fal.md`:
- `fal-ai/kling-video/v3` — Kling v3
- Additional MiniMax variants under `fal-ai/minimax` prefix
- Additional veo3.1 sub-variants

---

## Out of Scope

- No changes to Python source code
- No changes to test files
- No changes to provider implementations
- The `/add-fal-model` CLI skill is already documented in fal.md — no changes needed there
