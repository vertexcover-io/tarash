---
name: add-fal-model
description: Add a new model to the Fal provider in tarash-gateway. Fetches model docs from fal.ai, generates field mappers, registers in the model registry, writes unit and e2e tests, and runs them.
disable-model-invocation: true
user-invokable: true
argument-hint: "model-id"
---

# Add Fal Model Skill

You are adding a new model to the Fal provider in the tarash-gateway package.

The user provided model ID: `$ARGUMENTS`

## Key File Paths (from repo root)

- **Fal provider**: `packages/tarash-gateway/src/tarash/tarash_gateway/providers/fal.py`
- **Field mappers framework**: `packages/tarash-gateway/src/tarash/tarash_gateway/providers/field_mappers.py`
- **Models/types**: `packages/tarash-gateway/src/tarash/tarash_gateway/models.py`
- **API entry point**: `packages/tarash-gateway/src/tarash/tarash_gateway/api.py`
- **Unit tests (video)**: `packages/tarash-gateway/tests/unit/video/providers/test_fal.py`
- **Unit tests (image)**: `packages/tarash-gateway/tests/unit/image/providers/test_fal.py`
- **E2E video tests**: `packages/tarash-gateway/tests/e2e/test_fal.py`
- **E2E image tests**: `packages/tarash-gateway/tests/e2e/test_fal_image.py`
- **Test config**: `packages/tarash-gateway/tests/conftest.py`

## CRITICAL: Read Before Writing

**Before writing ANY code or tests**, you MUST read the existing implementations to understand the patterns:

1. **Read existing field mappers in `fal.py`** - Study how VEO3_FIELD_MAPPERS, SORA2_FIELD_MAPPERS, BYTEDANCE_SEEDANCE_FIELD_MAPPERS, KONTEXT_FIELD_MAPPERS are structured as unified mappers
2. **Read existing unit tests** - Study the test patterns in `tests/unit/video/providers/test_fal.py` and `tests/unit/image/providers/test_fal.py`
3. **Read existing e2e tests** - Study `tests/e2e/test_fal.py` and `tests/e2e/test_fal_image.py` to understand how tests maximize feature coverage per test

## Step-by-Step Process

### Step 1: Discover Sub-Models

Go to the fal.ai model page and discover all sub-model variants.

1. Fetch the model page at `https://fal.ai/models/$ARGUMENTS` using WebFetch
2. Extract the list of all sub-model endpoints/variants (e.g., text-to-video, image-to-video, etc.)
3. Note down each variant's full model ID (e.g., `fal-ai/flux-pro/kontext/max/text-to-image`)

### Step 2: Fetch Documentation for Each Sub-Model

For EACH sub-model variant discovered:

1. **Fetch the llms.txt** (model documentation):
   - URL: `https://fal.ai/models/<full-sub-model-id>/llms.txt`
   - This contains human-readable documentation about the model's capabilities and parameters

2. **Fetch the OpenAPI schema**:
   - URL: `https://fal.ai/api/openapi/queue/openapi.json?endpoint_id=<full-sub-model-id>`
   - This contains the exact API schema with all parameter types, constraints, and descriptions

3. Study the schema carefully. Pay attention to:
   - Required vs optional fields
   - Field types (string, integer, boolean, enum values)
   - Default values
   - Allowed values / constraints
   - Image input formats
   - Video input formats
   - Duration format and allowed values
   - Any model-specific parameters

### Step 3: Determine if Video or Image Model

Based on the documentation:
- If the model generates **videos**: add field mappers to `FAL_MODEL_REGISTRY` and use `VideoGenerationRequest`/`VideoGenerationResponse`
- If the model generates **images**: add field mappers to `FAL_IMAGE_MODEL_REGISTRY` and use `ImageGenerationRequest`/`ImageGenerationResponse`
- Some models may support both

### Step 4: Create Field Mappers

Read the existing `fal.py` to understand the pattern, then create field mappers.

**CRITICAL: Always prefer ONE unified field mapper dict per model family.**

Most model families have sub-variants (e.g., text-to-video, image-to-video, video-to-video) that share the same parameters. The correct approach is:

- Create a **single unified field mapper dict** that includes ALL fields across ALL variants
- Mark variant-specific fields as `required=False` — the API enforces which fields are actually required per variant
- Only create separate mapper dicts if variants have **fundamentally different parameter schemas** (different field names, different types — not just different required/optional status)

**Examples of the unified pattern:**
- `VEO3_FIELD_MAPPERS` — one mapper for text-to-video, image-to-video, first-last-frame-to-video, and extend-video
- `SORA2_FIELD_MAPPERS` — one mapper for text-to-video, image-to-video, and video-to-video/remix
- `BYTEDANCE_SEEDANCE_FIELD_MAPPERS` — one mapper for all v1/v1.5 variants
- `KONTEXT_FIELD_MAPPERS` — one mapper for text-to-image, image editing, and multi-image

**Anti-pattern (DO NOT DO THIS):**
```python
# BAD: Creating separate mappers when they only differ by one optional field
MODEL_TEXT_TO_VIDEO_MAPPERS = { "prompt": ..., "seed": ... }
MODEL_IMAGE_TO_VIDEO_MAPPERS = { "prompt": ..., "seed": ..., "image_url": ... }
MODEL_VIDEO_TO_VIDEO_MAPPERS = { "prompt": ..., "seed": ..., "video_url": ... }

# GOOD: One unified mapper with all fields, optional where needed
MODEL_FIELD_MAPPERS = {
    "prompt": ...,
    "seed": ...,
    "image_url": single_image_field_mapper(required=False, ...),  # API enforces when needed
    "video_url": video_url_field_mapper(required=False),           # API enforces when needed
}
```

**Use the appropriate mapper factory functions from `field_mappers.py`:**
  - `passthrough_field_mapper(source_field, required=False)` - direct pass-through
  - `duration_field_mapper(field_type, allowed_values, provider, model, add_suffix)` - duration conversion
  - `single_image_field_mapper(required, image_type, strict)` - single image from image_list
  - `image_list_field_mapper(image_type)` - list of images
  - `video_url_field_mapper(required)` - video URL conversion
  - `extra_params_field_mapper(param_name)` - extract from extra_params dict

**Naming convention for field mapper dicts:**
- Use UPPERCASE with underscores: `MODEL_NAME_FIELD_MAPPERS`
- Examples: `MINIMAX_FIELD_MAPPERS`, `VEO3_FIELD_MAPPERS`, `KONTEXT_FIELD_MAPPERS`

**Field mapping patterns:**
- `prompt` → always use `passthrough_field_mapper("prompt", required=True)`
- `duration` → use `duration_field_mapper()` with correct type and allowed values from schema
- `image_url` → use `single_image_field_mapper(image_type="reference")` for single reference image input
- `first_frame_url` → use `single_image_field_mapper(image_type="first_frame")`
- `last_frame_url` → use `single_image_field_mapper(image_type="last_frame")`
- `video_url` → use `video_url_field_mapper()`
- `aspect_ratio`, `resolution`, `seed`, `negative_prompt` → use `passthrough_field_mapper()`
- Provider-specific params not in the base request model → use `extra_params_field_mapper()`
- For image generation: `image_size`/`size` → use `passthrough_field_mapper("size")`
- For image generation: `num_images`/`n` → use `passthrough_field_mapper("n")`

**IMPORTANT: Check which fields exist on the request model.**
- `VideoGenerationRequest` has `enhance_prompt`, `generate_audio`, etc. as direct fields → use `passthrough_field_mapper()`
- `ImageGenerationRequest` does NOT have these fields → they will be captured into `extra_params` → use `extra_params_field_mapper()`
- When unsure, check the model class definition in `models.py`

### Step 5: Register in Model Registry

Add the field mapper(s) to the appropriate registry in `fal.py`:

**For video models** - add to `FAL_MODEL_REGISTRY`:
```python
FAL_MODEL_REGISTRY: dict[str, dict[str, FieldMapper]] = {
    # ... existing entries ...
    "fal-ai/new-model": NEW_MODEL_FIELD_MAPPERS,
}
```

**For image models** - add to `FAL_IMAGE_MODEL_REGISTRY`:
```python
FAL_IMAGE_MODEL_REGISTRY: dict[str, dict[str, FieldMapper]] = {
    # ... existing entries ...
    "fal-ai/new-model": NEW_MODEL_FIELD_MAPPERS,
}
```

**Registry key rules:**
- Use the **shortest prefix** that uniquely identifies the model family
- Since it's a unified mapper, register the common prefix **once** — all sub-variants will match via prefix lookup
- Only register multiple entries if the model family genuinely needs separate mappers
- Longer prefixes take precedence over shorter ones (longest-match wins)
- Add comments explaining what variants the entry covers

**Example — GOOD (one entry, prefix match covers all variants):**
```python
# Kontext - Unified mapper for all variants (text-to-image, image editing, multi)
"fal-ai/flux-pro/kontext": KONTEXT_FIELD_MAPPERS,
```

**Example — BAD (redundant entries for the same mapper):**
```python
# DON'T DO THIS when all variants use the same mapper
"fal-ai/flux-pro/kontext/max/multi": KONTEXT_FIELD_MAPPERS,
"fal-ai/flux-pro/kontext/multi": KONTEXT_FIELD_MAPPERS,
"fal-ai/flux-pro/kontext/max/text-to-image": KONTEXT_FIELD_MAPPERS,
"fal-ai/flux-pro/kontext/text-to-image": KONTEXT_FIELD_MAPPERS,
"fal-ai/flux-pro/kontext/max": KONTEXT_FIELD_MAPPERS,
"fal-ai/flux-pro/kontext": KONTEXT_FIELD_MAPPERS,
```

### Step 6: Write Unit Tests

Add unit tests to the appropriate test file. Follow these patterns:

**For video models** - add to `tests/unit/video/providers/test_fal.py`
**For image models** - add to `tests/unit/image/providers/test_fal.py`

**CRITICAL: Read existing tests first.** Study how tests are structured for VEO3, Wan, ByteDance Seedance, and Pixverse in the unit test files before writing new ones.

**What to test (each test should be valuable and non-redundant):**
1. **Unified registry lookup** - verify ALL variants resolve to the SAME mapper dict (single test with loop)
2. **Request conversion per variant** - one test per distinct usage pattern (e.g., text-only, with single image, with image list)
3. **Extra params** - verify model-specific params pass through correctly (combine into conversion tests, not separate)

**Test function naming**: `test_<model_name>_<what_is_tested>`
**DO NOT use classes** - use standalone test functions with fixtures.

**Example — registry lookup test (single test covers all variants):**
```python
def test_get_field_mappers_kontext_all_variants():
    """Test unified mapper for all Kontext variants via prefix matching."""
    variants = [
        "fal-ai/flux-pro/kontext",
        "fal-ai/flux-pro/kontext/max",
        "fal-ai/flux-pro/kontext/text-to-image",
        "fal-ai/flux-pro/kontext/max/text-to-image",
        "fal-ai/flux-pro/kontext/multi",
        "fal-ai/flux-pro/kontext/max/multi",
    ]
    for variant in variants:
        mappers = get_image_field_mappers(variant)
        assert mappers is KONTEXT_FIELD_MAPPERS, (
            f"Expected KONTEXT_FIELD_MAPPERS for {variant}"
        )
```

**Example — conversion tests (one per distinct usage pattern, each tests multiple features):**
```python
def test_kontext_text_to_image_conversion(handler):
    """Test Kontext text-to-image request conversion (prompt only, no images)."""
    # Tests: prompt, seed, n, aspect_ratio, guidance_scale, output_format
    # Also verifies no image fields appear in output

def test_kontext_image_editing_conversion(handler):
    """Test Kontext image editing conversion with single reference image."""
    # Tests: image_url from image_list, safety_tolerance, enhance_prompt

def test_kontext_multi_image_conversion(handler):
    """Test Kontext multi-image context conversion with image_urls list."""
    # Tests: image_urls from image_list with multiple images
```

### Step 7: Write E2E Tests

Add e2e tests to the appropriate test file.

**For video models** - add to `tests/e2e/test_fal.py`
**For image models** - add to `tests/e2e/test_fal_image.py`

**CRITICAL: Read existing e2e tests first.** Study how `test_comprehensive_async_video_generation` and `test_sync_veo31_image_to_video` in `tests/e2e/test_fal.py` maximize feature coverage per test.

**E2E test principles — MAXIMIZE COVERAGE PER TEST:**
- **Minimum tests, maximum features per test.** Each test should exercise as many features as possible in one go.
- Each test must cover a **DIFFERENT variant AND a different API path** (async vs sync). Do NOT write two tests that only differ by async/sync on the same variant — combine features into one test instead.
- Maximum 3-4 tests total for the new model (fewer is better)
- Include progress callback tracking in the async test
- Use realistic parameters from the model's documentation
- Always validate: response type, request_id, video/image URL, status

**How to decide test count:**
- 1 test: Model has one variant → one comprehensive test combining async + progress + all params
- 2 tests: Model has text-only + image input → one async test for text-only (with progress, n, extra params), one sync test for image variant
- 3 tests: Model has text-only + single image + multi-image OR pro/max differences → one test per distinct input type + one for a different model tier
- Never more than 4 tests

**Anti-pattern (DO NOT DO THIS):**
```python
# BAD: Two tests that only differ by async/sync on the SAME variant
async def test_model_text_to_video(api_key):      # async, text-to-video
    ...
def test_model_text_to_video_sync(api_key):        # sync, text-to-video (REDUNDANT!)
    ...
```

**Good pattern:**
```python
# GOOD: Each test covers a different variant AND maximizes features tested
async def test_model_text_to_video_async(api_key):
    """Async text-to-video with progress tracking, aspect_ratio, n, guidance_scale, seed."""
    ...

def test_model_image_editing_sync(api_key):
    """Sync image editing with reference image, safety_tolerance, enhance_prompt."""
    ...

async def test_model_max_variant(api_key):
    """Max model variant to verify prefix matching works for higher tier."""
    ...
```

**CRITICAL: Verify all image/video URLs before using them in E2E tests.**

Before adding ANY image or video URL to an E2E test, you MUST verify it is accessible by running a `curl -sI <url>` command and checking for an HTTP 200 response. Only use URLs that return 200. If a URL returns 403, 404, or any other error, find a different URL.

Good public test image URLs that are known to work:
- `https://storage.googleapis.com/falserverless/model_tests/remove_background/elephant.jpg`

When in doubt, use URLs from fal.ai documentation examples or from `storage.googleapis.com/falserverless/` which are maintained by Fal.

**E2E test template:**
```python
@pytest.mark.e2e
@pytest.mark.asyncio
async def test_<model>_<variant>_async(fal_api_key):
    """
    Test <model> <variant> with maximum feature coverage.

    This tests:
    - <variant description>
    - <feature 1: e.g., progress tracking>
    - <feature 2: e.g., aspect_ratio, n, seed>
    - <feature 3: e.g., extra_params like guidance_scale>
    """
    progress_updates = []

    async def progress_callback(update):
        progress_updates.append(update)
        print(f"  Progress: {update.status}")

    config = ImageGenerationConfig(  # or VideoGenerationConfig
        model="<full-model-id>",
        provider="fal",
        api_key=fal_api_key,
        timeout=180,
        max_poll_attempts=90,
        poll_interval=2,
    )

    request = ImageGenerationRequest(  # or VideoGenerationRequest
        prompt="...",
        seed=42,
        aspect_ratio="16:9",
        n=2,                              # Test multiple outputs
        extra_params={
            "guidance_scale": 5.0,         # Test model-specific params
            "output_format": "png",
        },
    )

    response = await api.generate_image_async(
        config, request, on_progress=progress_callback
    )

    # Validate response
    assert isinstance(response, ImageGenerationResponse)
    assert response.request_id is not None
    assert response.images is not None
    assert len(response.images) == 2
    assert response.status == "completed"

    for url in response.images:
        assert isinstance(url, str)
        assert url.startswith("http")

    # Validate progress
    assert len(progress_updates) > 0
    statuses = [u.status for u in progress_updates]
    assert "completed" in statuses

    print(f"✓ Generated: {response.request_id}")
```

### Step 8: Run Unit Tests

Run the unit tests first:

```bash
uv run pytest packages/tarash-gateway/tests/unit/video/providers/test_fal.py -v
```
Or for image models:
```bash
uv run pytest packages/tarash-gateway/tests/unit/image/providers/test_fal.py -v
```

If tests fail:
1. Read the error messages carefully
2. Understand WHY the test is failing
3. Fix the issue (in the test or in the field mapper code)
4. Re-run only the failing tests using `-k "test_name"` to target specific tests
5. Only re-run the full test file if your fix could affect other tests (e.g., changes to shared field mappers, registry, or helper functions). If the fix is isolated to the new model's tests, just re-run those.

### Step 9: Run E2E Tests

Before running e2e tests:

1. Check if FAL_KEY environment variable is set:
   ```bash
   echo $FAL_KEY
   ```

2. If NOT set, load from `.env` file:
   ```bash
   set -a && source .env && set +a
   ```

3. Run the e2e tests:
   ```bash
   uv run pytest packages/tarash-gateway/tests/e2e/test_fal.py -v --e2e -k "<model_name>"
   ```
   Or for image models:
   ```bash
   uv run pytest packages/tarash-gateway/tests/e2e/test_fal_image.py -v --e2e -k "<model_name>"
   ```

4. If tests fail:
   - Read the error output carefully
   - Check if it's a field mapping issue (wrong field name, wrong format)
   - Refer back to the schema/llms.txt documentation
   - Fix and re-run only the failing tests

### Step 10: Final Verification

Once all tests pass:
1. Run the full fal unit test suite to ensure no regressions:
   ```bash
   uv run pytest packages/tarash-gateway/tests/unit/video/providers/test_fal.py -v
   ```
2. Summarize what was added:
   - Model name and variants
   - Field mappers created (should be ONE unified mapper)
   - Registry entries added (should be ONE prefix entry)
   - Tests written and passing

## Important Reminders

- **Use `uv run`** for ALL Python/pytest commands
- **Function-based tests only** - no class-based tests
- **Load env vars if missing** - check with `echo $VAR`, if not set run `set -a && source .env && set +a`
- **Read existing code first** - ALWAYS study existing implementations before writing new code or tests
- **ONE unified field mapper** per model family - do NOT create separate mappers for variants that share parameters
- **ONE registry entry** per model family - use the shortest prefix, let prefix matching handle variants
- **Maximize coverage per E2E test** - each test should combine multiple features, not test one thing
- **Keep field mappers minimal** - only map fields that the model actually uses
- **Add comments** to explain model-specific behavior or constraints
- **Verify all E2E test URLs** - before using any image or video URL in E2E tests, run `curl -sI <url>` to confirm it returns HTTP 200. Never use unverified URLs
