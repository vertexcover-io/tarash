# Image Gateway Migration Plan

## Executive Summary

Migrate from `video/` package to a flat structure at `tarash_gateway/` level that supports multiple media types (video, image) through unified provider handlers with type-specific methods.

---

## Current State

```
src/tarash/tarash_gateway/
├── __init__.py
├── logging.py
├── py.typed
└── video/
    ├── __init__.py
    ├── api.py                    # generate_video(), generate_video_async()
    ├── exceptions.py             # TarashException hierarchy
    ├── mock.py                   # MockProviderHandler
    ├── models.py                 # VideoGenerationConfig, Request, Response
    ├── orchestrator.py           # ExecutionOrchestrator
    ├── registry.py               # get_handler(), register_provider()
    ├── utils.py                  # Utility functions
    └── providers/
        ├── __init__.py
        ├── field_mappers.py      # FieldMapper framework
        ├── fal.py                # FalProviderHandler
        ├── openai.py
        ├── replicate.py
        ├── runway.py
        └── veo3.py
```

---

## Target State

```
src/tarash/tarash_gateway/
├── __init__.py                   # Re-export public API
├── logging.py                    # ✅ Keep as-is
├── py.typed                      # ✅ Keep as-is
├── api.py                        # 🆕 Unified: generate_video(), generate_image(), etc.
├── models.py                     # 🆕 Unified: GenerationConfig + Video/Image Request/Response
├── exceptions.py                 # ♻️ Move from video/, generalize
├── orchestrator.py               # ♻️ Move from video/, generalize
├── registry.py                   # ♻️ Move from video/, generalize
├── utils.py                      # ♻️ Move from video/
├── mock.py                       # ♻️ Move from video/, add generate_image()
└── providers/
    ├── __init__.py
    ├── field_mappers.py          # ♻️ Move from video/, add image mappers
    ├── fal.py                    # ♻️ Add generate_image(), generate_image_async()
    ├── openai.py                 # ♻️ Add generate_image(), generate_image_async()
    ├── replicate.py              # ✅ Keep (video-only for now)
    ├── runway.py                 # ✅ Keep (video-only)
    └── veo3.py                   # ✅ Keep (video-only)
```

---

## Phase 1: Restructure Package

### Wave 1.1: Move Files (Sequential)

**Single task** - Move all files from `video/` to root level. Must complete before any other work.

```
Move: video/exceptions.py → exceptions.py
Move: video/orchestrator.py → orchestrator.py
Move: video/registry.py → registry.py
Move: video/utils.py → utils.py
Move: video/mock.py → mock.py
Move: video/api.py → api.py
Move: video/models.py → models.py
Move: video/providers/ → providers/
Delete: video/
```

### Wave 1.2: Update Source Imports (6 Parallel Agents)

| Agent | Task | Files |
|-------|------|-------|
| **Agent A** | Update core module imports | `api.py`, `orchestrator.py`, `registry.py` |
| **Agent B** | Update model/exception imports | `models.py`, `exceptions.py`, `utils.py`, `mock.py` |
| **Agent C** | Update provider imports | `providers/__init__.py`, `providers/fal.py`, `providers/field_mappers.py` |
| **Agent D** | Update provider imports | `providers/openai.py`, `providers/replicate.py` |
| **Agent E** | Update provider imports | `providers/runway.py`, `providers/veo3.py`, `providers/azure_openai.py` |
| **Agent F** | Update package exports | `__init__.py` |

**Import pattern change:**
```python
# Before
from tarash.tarash_gateway.video.models import VideoGenerationConfig
from tarash.tarash_gateway.video.exceptions import TarashException
from tarash.tarash_gateway.video.providers.fal import FalProviderHandler

# After
from tarash.tarash_gateway.models import VideoGenerationConfig
from tarash.tarash_gateway.exceptions import TarashException
from tarash.tarash_gateway.providers.fal import FalProviderHandler
```

### Wave 1.3: Update Test Imports (4 Parallel Agents)

| Agent | Task | Files |
|-------|------|-------|
| **Agent G** | Update unit test imports | `tests/unit/test_*.py` (root level) |
| **Agent H** | Update & move unit/video tests | `tests/unit/video/test_*.py` → `tests/unit/` |
| **Agent I** | Update & move provider tests | `tests/unit/video/providers/test_*.py` → `tests/unit/providers/` |
| **Agent J** | Update E2E test imports | `tests/e2e/test_*.py` |

### Wave 1.4: Verify Phase 1 (Sequential)

Run full test suite: `uv run pytest tests/`

**Success Criteria:**
- All video unit tests pass
- All video E2E tests pass
- No import errors

---

## Phase 2: Add Multi-Media Support

### Wave 2.1: Core Infrastructure (3 Parallel Agents)

| Agent | Task | Description |
|-------|------|-------------|
| **Agent K** | Add image models to `models.py` | Add `ImageGenerationRequest`, `ImageGenerationResponse`, `ImageGenerationUpdate` |
| **Agent L** | Add image field mappers to `providers/field_mappers.py` | Add `size_field_mapper`, `quality_field_mapper`, `style_field_mapper`, `n_images_field_mapper` |
| **Agent M** | Update `orchestrator.py` | Add `execute_image_async()`, `execute_image_sync()` methods |

**Image Models to Add:**
```python
class ImageGenerationRequest(BaseModel):
    prompt: str
    negative_prompt: str | None = None
    size: str | None = None          # "1024x1024", "1024x1792", etc.
    quality: str | None = None       # "standard", "hd"
    style: str | None = None         # "natural", "vivid"
    n: int | None = None             # Number of images
    image_list: list[ImageType] = Field(default_factory=list)
    mask_image: MediaType | None = None
    seed: int | None = None
    aspect_ratio: str | None = None
    extra_params: dict[str, Any] = Field(default_factory=dict)

class ImageGenerationResponse(BaseModel):
    request_id: str
    images: list[str]  # List of URLs
    status: Literal["completed", "failed"]
    is_mock: bool = False
    raw_response: dict[str, Any]
    revised_prompt: str | None = None
    execution_metadata: ExecutionMetadata | None = None

class ImageGenerationUpdate(BaseModel):
    request_id: str
    status: Literal["queued", "processing", "completed", "failed"]
    progress_percent: int | None = None
    update: dict[str, Any]
```

### Wave 2.2: Add Provider Stubs (5 Parallel Agents)

Add `NotImplementedError` stubs to existing providers:

| Agent | Task | File |
|-------|------|------|
| **Agent N** | Add image stubs | `providers/openai.py` |
| **Agent O** | Add image stubs | `providers/replicate.py` |
| **Agent P** | Add image stubs | `providers/runway.py` |
| **Agent Q** | Add image stubs | `providers/veo3.py` |
| **Agent R** | Add image stubs | `providers/azure_openai.py` |

**Stub pattern:**
```python
async def generate_image_async(self, config, request, on_progress=None):
    raise NotImplementedError(f"{self.__class__.__name__} does not support image generation")

def generate_image(self, config, request, on_progress=None):
    raise NotImplementedError(f"{self.__class__.__name__} does not support image generation")
```

### Wave 2.3: Update API & Mock (2 Parallel Agents)

| Agent | Task | Description |
|-------|------|-------------|
| **Agent S** | Update `api.py` | Add `generate_image()`, `generate_image_async()` |
| **Agent T** | Update `mock.py` | Add `generate_image()`, `generate_image_async()` |

### Wave 2.4: Verify Phase 2 (Sequential)

Run full test suite: `uv run pytest tests/`

**Success Criteria:**
- All existing video tests still pass
- New image API functions exist (raise NotImplementedError for all providers)

---

## Phase 3: Implement Fal Image Generation

### Wave 3.1: Fal Implementation (2 Parallel Agents)

| Agent | Task | Description |
|-------|------|-------------|
| **Agent U** | Add Fal image model registry | Add `FAL_IMAGE_MODEL_REGISTRY` with FLUX, Recraft, etc. field mappers |
| **Agent V** | Add Fal image generation methods | Add `generate_image_async()`, `generate_image()`, `_convert_image_request()`, `_convert_image_response()` |

**Fal Image Models to Support:**
- `fal-ai/flux-pro` (FLUX Pro)
- `fal-ai/flux/schnell` (FLUX Schnell)
- `fal-ai/recraft-v3` (Recraft V3)
- `fal-ai/ideogram/v3` (Ideogram V3)

### Wave 3.2: Tests (2 Parallel Agents)

| Agent | Task | Description |
|-------|------|-------------|
| **Agent W** | Unit tests | Create `tests/unit/test_image_models.py`, `tests/unit/providers/test_fal_image.py` |
| **Agent X** | E2E tests | Create `tests/e2e/test_fal_image.py` |

### Wave 3.3: Final Verification (Sequential)

Run full test suite including new image tests: `uv run pytest tests/`

**Success Criteria:**
- Fal image generation works end-to-end
- All image unit tests pass
- E2E tests pass with real API
- All existing video tests still pass

---

## Dependency Graph

```
Wave 1.1 (Move Files)
    ↓
Wave 1.2 (Update Source Imports) ←─ 6 agents in parallel
    ↓
Wave 1.3 (Update Test Imports) ←─ 4 agents in parallel
    ↓
Wave 1.4 (Verify Phase 1)
    ↓
Wave 2.1 (Image Models + Field Mappers + Orchestrator) ←─ 3 agents in parallel
    ↓
Wave 2.2 (Provider Stubs) ←─ 5 agents in parallel
    ↓
Wave 2.3 (API + Mock) ←─ 2 agents in parallel
    ↓
Wave 2.4 (Verify Phase 2)
    ↓
Wave 3.1 (Fal Image Implementation) ←─ 2 agents in parallel
    ↓
Wave 3.2 (Tests) ←─ 2 agents in parallel
    ↓
Wave 3.3 (Final Verification)
```

---

## Execution Summary

| Phase | Wave | Tasks | Parallelism |
|-------|------|-------|-------------|
| 1 | 1.1 | Move files | Sequential |
| 1 | 1.2 | Update source imports | 6 parallel |
| 1 | 1.3 | Update test imports | 4 parallel |
| 1 | 1.4 | Verify | Sequential |
| 2 | 2.1 | Core infrastructure | 3 parallel |
| 2 | 2.2 | Provider stubs | 5 parallel |
| 2 | 2.3 | API + Mock | 2 parallel |
| 2 | 2.4 | Verify | Sequential |
| 3 | 3.1 | Fal implementation | 2 parallel |
| 3 | 3.2 | Tests | 2 parallel |
| 3 | 3.3 | Final verify | Sequential |

**Total: 25 agent tasks across 11 waves**

---

## Risk Mitigation

1. **Import breakage**: Phase 1 focuses solely on restructuring with no functionality changes. Run all tests after each wave.

2. **Regression**: Each phase ends with full test suite run. No phase proceeds until previous passes.

3. **Provider parity**: Initially only Fal gets image support. Other providers get `NotImplementedError` stubs - clean failure.

---

## File Change Summary

| File | Change Type | Description |
|------|-------------|-------------|
| `video/` | DELETE | Remove entire package |
| `api.py` | MOVE + UPDATE | Add `generate_image()`, `generate_image_async()` |
| `models.py` | MOVE + UPDATE | Add image models |
| `exceptions.py` | MOVE | No content changes |
| `orchestrator.py` | MOVE + UPDATE | Add image execution methods |
| `registry.py` | MOVE | No content changes |
| `utils.py` | MOVE | No content changes |
| `mock.py` | MOVE + UPDATE | Add image generation methods |
| `providers/field_mappers.py` | MOVE + UPDATE | Add image-specific mappers |
| `providers/fal.py` | MOVE + UPDATE | Add image generation + registry |
| `providers/openai.py` | MOVE + STUB | Add `NotImplementedError` stubs |
| `providers/replicate.py` | MOVE + STUB | Add `NotImplementedError` stubs |
| `providers/runway.py` | MOVE + STUB | Add `NotImplementedError` stubs |
| `providers/veo3.py` | MOVE + STUB | Add `NotImplementedError` stubs |
| `providers/azure_openai.py` | MOVE + STUB | Add `NotImplementedError` stubs |
| `tests/unit/video/` | MOVE + UPDATE | Move to `tests/unit/`, update imports |
| `tests/e2e/` | UPDATE | Update imports, add image tests |
