# Phase 1: `SilenceRemovalPreview` Model

> **Status:** pending
> **Depends on:** none

## Overview

Add the `SilenceRemovalPreview` Pydantic model that holds estimated metrics from a dry-run of the silence removal pipeline. After this phase, the model is importable and tested but not yet wired into any API function.

## Implementation

**Files:**
- Modify: `packages/tarash-silence-remover/src/tarash/tarash_silence_remover/models.py` -- add `SilenceRemovalPreview` class
- Modify: `packages/tarash-silence-remover/src/tarash/tarash_silence_remover/__init__.py` -- export `SilenceRemovalPreview`
- Test: `packages/tarash-silence-remover/tests/unit/test_models.py` -- tests for the new model

**Pattern to follow:** `SilenceRemovalResponse` in `models.py` lines 113-129.

**What to test:**
- Model construction with valid fields
- `estimated_removed_duration` computed property returns correct value
- `reduction_percent` computed property returns correct percentage
- `reduction_percent` returns 0.0 when `original_duration` is 0 (division guard)
- Model is frozen (immutable)
- `segments_to_keep` accepts empty list

**What to build:**

```python
class SilenceRemovalPreview(BaseModel):
    """Estimated result of silence removal without processing."""

    original_duration: float = Field(description="Original file duration in seconds.")
    estimated_output_duration: float = Field(description="Estimated output duration in seconds.")
    segments_to_keep: list[SpeechSegment] = Field(description="Speech segments after padding + merge.")
    silence_gaps_to_insert: int = Field(description="Number of shortened silence gaps that would be inserted.")
    detector_used: str = Field(description="Which detector backend was used.")

    model_config: ClassVar[ConfigDict] = ConfigDict(frozen=True)

    @property
    def estimated_removed_duration(self) -> float:
        """Estimated silence removed in seconds."""
        return self.original_duration - self.estimated_output_duration

    @property
    def reduction_percent(self) -> float:
        """Estimated file duration reduction as a percentage."""
        if self.original_duration == 0:
            return 0.0
        return (self.estimated_removed_duration / self.original_duration) * 100
```

Also add `"SilenceRemovalPreview"` to the `__all__` list in `__init__.py` under the Models section, and add the import from `.models`.

**Commit:** `✨ Add SilenceRemovalPreview model`

## Done When

- [ ] `SilenceRemovalPreview` model is defined in `models.py` with all fields and properties
- [ ] Model is exported from `__init__.py`
- [ ] Unit tests pass for construction, computed properties, and edge cases
- [ ] Existing tests still pass: `uv run pytest packages/tarash-silence-remover/tests/unit/`
