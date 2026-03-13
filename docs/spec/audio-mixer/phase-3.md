# Phase 3: Envelope Generation (Volume Filter Builder)

## Covers
- REQ-002 (generate FFmpeg volume filter expression)
- REQ-003 (attack ramps)
- REQ-004 (release ramps)
- REQ-005 (speech padding)
- REQ-006 (base music volume in non-speech)
- REQ-007 (duck level relative to base)
- EDGE-002 (overlapping duck regions merge)
- EDGE-008 (speech at t=0, truncated attack)
- EDGE-009 (speech at end, truncated release)
- EDGE-014 (zero attack/release = step function)

## Dependencies
- Phase 1 (models — SpeechSegment, AudioMixerConfig)

## Files to Modify

### `packages/tarash-audio-mixer/src/tarash/tarash_audio_mixer/processor.py`

Add envelope-related functions. These are pure functions operating on data — no FFmpeg calls.

#### Functions to implement:

- `compute_duck_regions(segments: list[SpeechSegment], config: AudioMixerConfig, total_duration: float) -> list[DuckRegion]`:
  - For each speech segment, compute padded boundaries: `duck_start = max(0, seg.start - speech_padding)`, `duck_end = min(total_duration, seg.end + speech_padding)`
  - Compute ramp boundaries:
    - `attack_start = max(0, duck_start - attack_ms/1000)`
    - `attack_end = duck_start`
    - `release_start = duck_end`
    - `release_end = min(total_duration, duck_end + release_ms/1000)`
  - When attack_ms=0: attack_start = attack_end (step function)
  - When release_ms=0: release_start = release_end (step function)
  - Clamp all times to [0, total_duration]
  - Return list of `DuckRegion` (internal dataclass/namedtuple, not a Pydantic model)

- `merge_duck_regions(regions: list[DuckRegion]) -> list[DuckRegion]`:
  - Sort by attack_start
  - Merge overlapping regions (EDGE-002): when two regions overlap, keep the earliest attack_start and latest release_end, merge the full-duck zone
  - This ensures no volume spikes between adjacent speech segments

- `build_volume_expression(regions: list[DuckRegion], config: AudioMixerConfig, total_duration: float) -> str`:
  - Convert merged duck regions into an FFmpeg volume filter expression
  - Base gain = `10^(base_music_volume_db/20)` (linear)
  - Duck gain = `10^((base_music_volume_db + duck_level_db)/20)` (linear)
  - For each region, generate `if(between(t, ...), ...)` clauses:
    - Attack ramp: linear interpolation from base_gain to duck_gain
    - Full duck: constant duck_gain
    - Release ramp: linear interpolation from duck_gain to base_gain
  - Default (no match): base_gain
  - Expression format: nested `if` or chained `if` with `+` for min-of-overlaps

- `DuckRegion` (internal type, not exported):
  - `attack_start: float`
  - `attack_end: float` (= padded speech start)
  - `full_duck_start: float` (= attack_end)
  - `full_duck_end: float` (= padded speech end)
  - `release_start: float` (= full_duck_end)
  - `release_end: float`

## Tests to Write

### `packages/tarash-audio-mixer/tests/unit/test_processor.py` (envelope-related tests)

All pure-function tests — no FFmpeg mocking needed.

- `test_compute_duck_regions_single_segment` — one speech segment in middle of file, verify attack/release boundaries
- `test_compute_duck_regions_segment_at_start` — speech at t=0, attack ramp truncated to 0 (EDGE-008)
- `test_compute_duck_regions_segment_at_end` — speech at end of file, release truncated (EDGE-009)
- `test_compute_duck_regions_zero_attack` — attack_ms=0, attack_start == attack_end (EDGE-014)
- `test_compute_duck_regions_zero_release` — release_ms=0, release_start == release_end (EDGE-014)
- `test_compute_duck_regions_with_padding` — verify padding extends duck boundaries (REQ-005)
- `test_merge_duck_regions_no_overlap` — two separate regions stay separate
- `test_merge_duck_regions_overlapping` — two close segments merge into one region (EDGE-002)
- `test_merge_duck_regions_empty` — empty input returns empty
- `test_build_volume_expression_single_region` — verify expression contains `between` clause with correct times
- `test_build_volume_expression_no_regions` — no speech, expression is just base gain (EDGE-001 downstream)
- `test_build_volume_expression_base_volume` — verify base gain corresponds to base_music_volume_db (REQ-006)
- `test_build_volume_expression_duck_volume` — verify duck gain = base + duck_level (REQ-007)
- `test_build_volume_expression_ramp_linearity` — verify attack/release use linear interpolation formula (REQ-003, REQ-004)
- `test_build_volume_expression_step_function` — zero attack/release produces no interpolation clause

## Verification
- `uv run pytest tests/unit/test_processor.py -v -k "duck or envelope or volume_expression"`
- All envelope computation tests pass
