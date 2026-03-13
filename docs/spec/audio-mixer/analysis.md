# Self-Q&A Analysis: tarash-audio-mixer

## 1. Assumptions the spec makes

### A1: FFmpeg volume filter with `if`/`between` clauses is the right approach for ducking
- **What if wrong?** FFmpeg's `volume` filter with complex expressions can hit expression length limits for files with many speech segments. The expression string could become enormous.
- **Safer alternative:** Build the expression incrementally and test with a high segment count (e.g., 100+). If FFmpeg has a practical limit, we could fall back to `afade` or use a filter_complex with multiple volume nodes chained. For now, proceed with the single `volume` filter expression since it's the simplest approach and typical podcast/video files have <50 speech segments.

### A2: Silero VAD works reliably on foreground-only audio
- **What if wrong?** The foreground file might already have background music baked in, confusing VAD. But the spec says foreground = speech file, so this is the user's responsibility.
- **Resolution:** Document that foreground should be a clean speech track. VAD threshold is configurable.

### A3: `base_music_volume_db + duck_level_db` is always negative (attenuating)
- **What if wrong?** A user could set `duck_level_db=0` or positive values. `duck_level_db` is relative, so `base_music_volume_db=-6` + `duck_level_db=-12` = `-18 dB` during speech. But `duck_level_db=+6` would mean louder during speech.
- **Resolution:** The spec says "duck level," implying attenuation. Add a validator that `duck_level_db <= 0` to prevent accidental amplification during speech.

### A4: Crossfade at loop boundaries is handled by FFmpeg's `acrossfade` filter
- **What if wrong?** `acrossfade` operates on two separate audio streams. For looping, we need to generate a looped track first, then apply crossfades at join points. This is complex in a single FFmpeg command.
- **Safer alternative:** Use FFmpeg's `aloop` or manual concatenation with `acrossfade` between loop iterations. If crossfade is too complex, fall back to simple concatenation (EDGE-005 already handles this for short backgrounds).

### A5: Background audio can be any format FFmpeg supports
- **What if wrong?** Some formats (e.g., MIDI) won't work with volume filters. But FFmpeg will fail with a clear error, which we catch as `InvalidInputError`.
- **Resolution:** Let FFmpeg validate format compatibility naturally.

### A6: The `SpeechSegment` model from silence-remover can be reused
- **What if wrong?** The audio-mixer spec defines its own `SpeechSegment`. It has the same shape (start/end floats). Importing from silence-remover would add an unwanted dependency.
- **Resolution:** Define our own `SpeechSegment` in audio-mixer's `models.py`, matching the same pattern. No cross-package dependency.

### A7: Single FFmpeg invocation can handle the full mix pipeline
- **What if wrong?** Complex filter graphs (resample + volume envelope + loop + crossfade + mix) might be brittle. If one step fails, debugging is hard.
- **Safer alternative:** Use a multi-step pipeline: (1) probe both files, (2) prepare background (loop/pad/resample), (3) build and apply duck envelope, (4) mix. Each step is a separate FFmpeg call. This is more debuggable and testable.

### A8: `asyncio.to_thread` is sufficient for async FFmpeg operations
- **What if wrong?** `asyncio.create_subprocess_exec` is more efficient for I/O-bound FFmpeg processes.
- **Resolution:** Follow the silence-remover pattern: use `asyncio.create_subprocess_exec` for FFmpeg calls, `asyncio.to_thread` only for CPU-bound Silero VAD inference.

## 2. Top 3 riskiest design decisions

### Risk 1: Complex FFmpeg filter expression for volume envelope
- **Risk:** A single `volume` filter expression with many `if(between(...))` clauses could be fragile, hard to debug, and potentially hit FFmpeg's expression parser limits.
- **Fallback:** If expression-based approach fails, use a two-pass approach: (1) generate a gain automation file, (2) apply it with `volume` filter reading from file. Alternatively, use multiple chained `volume` filters with `enable` expressions. The expression approach is testable in unit tests (just string generation), so we'll know early if it works.

### Risk 2: Background looping with crossfade
- **Risk:** Implementing seamless looping with crossfade in FFmpeg is non-trivial. The `acrossfade` filter needs two separate inputs, so looping N times requires N-1 crossfade operations.
- **Fallback:** For v1, implement simple looping without crossfade as the default, and add crossfade as an enhancement. EDGE-005 already acknowledges that crossfade may be skipped. We can use `aloop` for basic looping and add crossfade only when the background is long enough to make it worthwhile.
- **Practical approach:** Pre-process the background: concatenate copies with crossfade applied between each pair, then trim to foreground duration.

### Risk 3: Overlapping duck regions (EDGE-002)
- **Risk:** When adjacent speech segments have overlapping attack/release ramps, the envelope must take the `min()` of both curves. This is complex in an FFmpeg expression.
- **Fallback:** Pre-merge overlapping duck regions in Python before generating the FFmpeg expression. Instead of computing min() in FFmpeg, compute the merged envelope in Python and emit non-overlapping FFmpeg `between` clauses. This is simpler and fully testable in unit tests.

## 3. Ambiguities and resolutions

### Ambiguity 1: How to compute the FFmpeg volume expression
- **Option A:** Generate a single `volume` filter with nested `if(between(...))` for each segment.
- **Option B:** Pre-compute a list of (time, gain) keyframes in Python and generate the expression from that.
- **Resolution:** Choose Option B. Computing keyframes in Python is easier to test, debug, and reason about. The keyframe list naturally handles overlapping duck regions (take min of overlapping gains). The FFmpeg expression is then a mechanical translation.

### Ambiguity 2: Whether `detector.py` should be a module or a `detectors/` package
- **Option A:** Single `detector.py` file (spec says Silero VAD only, no fallback).
- **Option B:** `detectors/` package like silence-remover.
- **Resolution:** Choose Option A (single `detector.py`). The spec explicitly states "Silero VAD only" and lists "FFmpeg silencedetect fallback detector" as out of scope. A single file is simpler and avoids premature abstraction.

### Ambiguity 3: Whether the processor should use a single complex FFmpeg command or multiple steps
- **Option A:** Single `ffmpeg` command with a complex filter_complex graph.
- **Option B:** Multiple sequential FFmpeg commands (prepare background, build envelope, mix).
- **Resolution:** Choose Option B (multi-step). Each step is independently testable and debuggable. The silence-remover uses multi-step processing successfully. Steps: (1) probe both files, (2) prepare background (loop/pad + resample + channel match), (3) apply duck envelope to background, (4) mix foreground + processed background.
- **Revised after further thought:** Actually, steps 2-4 can be a single `ffmpeg` command using `filter_complex` with multiple filter chains. This avoids intermediate files and is more efficient. The filter_complex string is built in Python (testable) and executed once. The command structure: two inputs, filter_complex applies looping/padding + resampling + volume envelope to input[1], optional gain to input[0], then amix. This is one FFmpeg call but the filter construction is well-separated in Python code.

### Ambiguity 4: What `output_format` means -- codec or container
- **Resolution:** Container format (file extension). FFmpeg infers codec from container. This matches how silence-remover handles output: the suffix determines the format.

### Ambiguity 5: Whether `speech_padding` in audio-mixer serves the same purpose as `padding` in silence-remover
- **Resolution:** Different purpose. In silence-remover, padding extends the kept speech segments. In audio-mixer, `speech_padding` extends the duck region around speech -- it determines how early the background starts ducking before speech and how long after speech it stays ducked. The duck boundaries are `segment.start - speech_padding` and `segment.end + speech_padding`.

### Ambiguity 6: What happens when foreground has no audio stream
- **Resolution:** Raise `InvalidInputError`. The spec says "foreground (speech)" -- it must have audio. Probe foreground and check for audio stream presence.
