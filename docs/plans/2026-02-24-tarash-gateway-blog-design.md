# Blog Design: Tarash Gateway

**Date:** 2026-02-24
**File:** `blogs/blog/2026-02-24-tarash-gateway.mdx`
**Slug:** `unified-ai-video-gateway`
**Tags:** `python`, `ai`, `video-generation`, `open-source`
**Author:** aksdev

---

## Title

**"Stop Writing Video Provider Integrations. We Did It For You."**

---

## Target Audience

Python developers building AI products — engineers integrating AI video/image generation into their apps.

## Primary Goal

Drive adoption: readers should `pip install tarash-gateway` and start using it.

## Tone & Flow Reference

Mirror `2025-11-23-scene-flow.mdx` exactly:
- First-person plural ("we built", "we searched", "we tried")
- Short punchy paragraphs, direct voice
- Specific client story as the hook (not abstract)
- Rhetorical questions to frame intuitions ("Think about it…")
- "We tried X. It didn't work. So we built Y." rhythm
- **Our Fix:** pattern for each challenge
- Same section structure (TLDR → Story → Intuition → Pillars → Challenges → When To Use → See It In Action → How To Use → Conclusion)

---

## Section-by-Section Plan

### TLDR
One-liner in the `<TLDR>` component:
> **Tarash Gateway** is an open-source Python SDK that gives you one consistent API across 8 AI video and image providers — Fal, Runway, Google Veo, Sora, Kling, Replicate, Stability, and Luma. Switch providers with one line. Add fallback chains. Test without API calls.

---

### Section 1: The Story (Hook)
Real client scenario, 3–4 short paragraphs:

- Building a personalized AI video pipeline — avatar videos for outreach campaigns
- Picked Fal.ai with Veo3. Shipped it. Three weeks later, client asks for Kling.
- Write a second integration. Then Sora drops. Now there are three polling loops, three parameter schemas (`duration` vs `duration_seconds`), three error formats, three credential sets.
- Provider goes down at 2am — pipeline goes down too.
- **"There's no good reason every team should solve this separately. So we built Tarash Gateway."**

`{/* truncate */}` goes here.

---

### Section 2: The Core Intuition
Frame as: *"What does a good integration actually look like?"*

- A human switching providers doesn't relearn the problem — they just swap the backend.
- Your prompt, duration, and aspect ratio are the same regardless of which model renders them.
- Every provider should be a plug-in — one unified request in, one unified response out.
- One-liner: **"LiteLLM for video generation."**

---

### Section 3: How Tarash Gateway Implements This — The Four Pillars

**Pillar 1: Unified Models**
- `VideoGenerationConfig` + `VideoGenerationRequest`
- Set `provider` and `model`; everything else is identical
- Code snippet: switching Fal → Runway with a single field change

**Pillar 2: The Field Mapper Registry**
- Most interesting technical piece — declarative parameter translation
- Instead of Python code per provider, you declare a mapper once
- Show `duration_field_mapper("int", [4, 8, 12])` example
- Adding a new model = adding a dict entry, zero new code
- Show architecture diagram: `request → mapper → provider payload`

**Pillar 3: Fallback Chains**
- Show 3-provider fallback config (Fal → Replicate → OpenAI)
- Orchestrator catches retryable errors, falls to next, tracks `execution_metadata`
- "If Fal is down at 2am, Replicate catches it."

**Pillar 4: Mock Provider**
- One line to enable: `mock=MockConfig(enabled=True)`
- No API call, deterministic response
- Makes testing fast and free

---

### Section 4: Real-World Challenges We Faced

**Challenge 1: Async Client Caching**
- Different providers handle event loops differently
- Fal's async client throws if reused across event loops
- **Our Fix:** provider-aware caching strategies — cache sync clients only for Fal, cache both for OpenAI
- One interface, different strategies underneath

**Challenge 2: Model Variant Matching**
- Replicate appends version hashes: `minimax/video-01:abc123...`
- Can't maintain a hash-per-version registry
- **Our Fix:** longest-prefix matching — `"minimax/"` matches any Minimax version automatically

---

### Section 5: When to Use Tarash Gateway
Short scannable list (mirrors SceneFlow's "When to Use SceneFlow"):
- More than one AI video provider in your stack
- Fallback resilience (provider outage protection)
- Swapping providers for cost/quality experiments
- Testing pipelines without burning API credits

---

### Section 6: See It In Action
(If we have code output or demo — otherwise skip and fold into Section 7)

---

### Section 7: How to Use It
Install + 3 focused code snippets:
1. Basic video generation (5 lines)
2. Switching providers (2 field changes)
3. Fallback chain with progress callback

GitHub link: https://github.com/vertexcover-io/tarash

---

### Section 8: Conclusion
Short forward-looking paragraph:
- The AI video provider landscape changes fast. New models every month, pricing shifts, outages.
- Tarash Gateway is the abstraction layer that keeps you flexible.
- Open source, extensible, production-ready.
- CTA: `pip install tarash-gateway` + GitHub link

---

## Key Messages

1. Provider chaos is a solved problem — you shouldn't be solving it again
2. The field mapper registry makes adding providers declarative, not imperative
3. Fallback chains are first-class, not bolted on
4. The mock provider makes testing trivially easy

---

## Files to Create

- `blogs/blog/2026-02-24-tarash-gateway/index.mdx` — main blog post
- Any images referenced in the blog (architecture diagram, etc.) go in `blogs/static/img/`
