# Tarash Gateway Blog Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Write and publish a blog post about Tarash Gateway in `blogs/blog/` that drives adoption of the open-source SDK.

**Architecture:** Single MDX file matching the SceneFlow blog format. Story-first structure: client pipeline story → provider chaos pain → core intuition → four technical pillars → real engineering challenges → when to use → code snippets → conclusion.

**Tech Stack:** MDX, Docusaurus 3.8.1, React components (`TLDR`), bun for build validation.

---

## Reference

- Tone/format model: `blogs/blog/2025-11-23-scene-flow.mdx`
- Design doc: `docs/plans/2026-02-24-tarash-gateway-blog-design.md`
- Blog working directory: `blogs/`
- Build command (run from `blogs/`): `bun run build`

---

### Task 1: Create the MDX file with frontmatter, imports, and TLDR

**Files:**
- Create: `blogs/blog/2026-02-24-tarash-gateway.mdx`

**Step 1: Create the file with frontmatter and TLDR**

```mdx
---
slug: "unified-ai-video-gateway"
title: "Stop Writing Video Provider Integrations. We Did It For You."
date: 2026-02-24T00:00:00+00:00
authors: [aksdev]
tags: [python, ai, video-generation, open-source]
draft: false
---

import TLDR from '@site/src/components/TLDR';

<TLDR>

**Tarash Gateway** is an open-source Python SDK that gives you one consistent API across 8 AI video and image providers — Fal, Runway, Google Veo, Sora, Kling, Replicate, Stability, and Luma. Switch providers with one line. Add fallback chains. Test without API calls.

</TLDR>
```

**Step 2: Verify build passes**

Run from `blogs/` directory:
```bash
bun run build
```
Expected: Build succeeds with no errors.

**Step 3: Commit**
```bash
git add blogs/blog/2026-02-24-tarash-gateway.mdx
git commit -m "📝 Add Tarash Gateway blog scaffold"
```

---

### Task 2: Write Section 1 — The Story (Hook)

**Files:**
- Modify: `blogs/blog/2026-02-24-tarash-gateway.mdx`

**Step 1: Append the hook section after the TLDR**

```mdx
We were building a personalized video pipeline for a client. AI avatar videos — hundreds of them, for personalized sales outreach campaigns. Different scripts, different faces, all generated at scale. The model of choice was Veo3, running through Fal. The integration was straightforward: send a request, poll for completion, grab the URL.

Three weeks in, the client asked for Kling. Their use case involved specific motion styles that Veo3 didn't handle as well. Fair enough. We wrote a second integration. Different SDK, different parameter names (`duration_seconds` on Fal, `duration` on Kling), different polling pattern, different error format. About a day's work.

Then Sora dropped. The client wanted to evaluate it. We wrote a third integration.

Now we had three polling loops, three parameter schemas, three error hierarchies, three sets of credentials to manage. When Fal had a 20-minute outage one morning, the pipeline went down with it. There was no fallback. No graceful degradation. Just a flood of errors and a manual switch to the next provider.

There's no good reason every team building AI video pipelines should solve this independently. So we built **Tarash Gateway**.

{/* truncate */}
```

**Step 2: Verify build**
```bash
bun run build
```

**Step 3: Commit**
```bash
git add blogs/blog/2026-02-24-tarash-gateway.mdx
git commit -m "📝 Add story hook section to Tarash Gateway blog"
```

---

### Task 3: Write Section 2 — The Core Intuition

**Files:**
- Modify: `blogs/blog/2026-02-24-tarash-gateway.mdx`

**Step 1: Append after the truncate comment**

```mdx
## The Core Intuition: Every Provider is a Plug-in

Before writing any code, we asked ourselves a simple question: *What does a good multi-provider integration actually look like?*

Think about it. When a developer switches from one database ORM to another, they don't rewrite their business logic. The abstraction layer handles the translation. When you use Requests vs. httpx, you don't change how you think about HTTP — just which library you call.

The same principle should apply to AI video providers. **Your prompt, your aspect ratio, your duration — these don't change when you switch from Veo3 to Sora.** What changes is the adapter that speaks each provider's language.

Every provider should be a plug-in. One unified request in, one unified response out. This is the same idea behind LiteLLM for language models. It works just as well for video.
```

**Step 2: Verify build**
```bash
bun run build
```

**Step 3: Commit**
```bash
git add blogs/blog/2026-02-24-tarash-gateway.mdx
git commit -m "📝 Add core intuition section to Tarash Gateway blog"
```

---

### Task 4: Write Section 3 — Pillar 1 and 2 (Unified Models + Field Mapper Registry)

**Files:**
- Modify: `blogs/blog/2026-02-24-tarash-gateway.mdx`

**Step 1: Append the first two pillars**

```mdx
## How Tarash Gateway Implements This

Here's the high-level flow: you construct a config (which provider, which model) and a request (your prompt and parameters). The gateway maps your request to the provider's payload, sends it, polls for completion, and returns a unified response. Same API. Every provider.

### The Four Pillars of a Clean Integration

**1. Unified Models**

Every generation call takes the same two objects: a `VideoGenerationConfig` and a `VideoGenerationRequest`. The response is always a `VideoGenerationResponse` with the same fields regardless of which provider rendered it.

Switching providers looks like this:

```python
from tarash.tarash_gateway import generate_video, VideoGenerationConfig, VideoGenerationRequest

request = VideoGenerationRequest(
    prompt="A calm lake at sunrise, mist rising off the water",
    duration_seconds=4,
    aspect_ratio="16:9"
)

# Fal
config = VideoGenerationConfig(provider="fal", model="fal-ai/veo3.1", api_key="...")
response = generate_video(config, request)

# Runway — change two fields, nothing else
config = VideoGenerationConfig(provider="runway", model="gen3a_turbo", api_key="...")
response = generate_video(config, request)

print(response.video)  # Same field. Every provider.
```

**2. The Field Mapper Registry**

This is the most interesting piece. Each provider speaks a different language. Fal wants `duration_seconds`. Runway wants `duration`. Kling wants `duration` as an enum string. You can't just forward the same payload everywhere.

Instead of writing custom translation code per provider, Tarash uses a declarative **field mapper registry**:

```python
FAL_VEO3_MAPPERS = {
    "prompt": passthrough_field_mapper("prompt", required=True),
    "duration": duration_field_mapper("int", [4, 8]),
    "aspect_ratio": passthrough_field_mapper("aspect_ratio"),
    "image_url": single_image_field_mapper(image_type="start_frame"),
}

FAL_MODEL_REGISTRY = {
    "fal-ai/veo3.1": FAL_VEO3_MAPPERS,
    "fal-ai/kling-video/v2.6": KLING_MAPPERS,
    "fal-ai/minimax": MINIMAX_MAPPERS,
}
```

Adding a new model variant? Add a line to the dict. Zero new code. The mapper handles type conversion, required field validation, and parameter forwarding automatically.
```

**Step 2: Verify build**
```bash
bun run build
```

**Step 3: Commit**
```bash
git add blogs/blog/2026-02-24-tarash-gateway.mdx
git commit -m "📝 Add unified models and field mapper registry sections"
```

---

### Task 5: Write Section 3 — Pillar 3 and 4 (Fallback Chains + Mock Provider)

**Files:**
- Modify: `blogs/blog/2026-02-24-tarash-gateway.mdx`

**Step 1: Append the last two pillars**

```mdx
**3. Fallback Chains**

Provider resilience is a first-class feature, not an afterthought. You configure a primary provider and a list of fallbacks:

```python
config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/veo3.1",
    api_key=FAL_KEY,
    fallback_configs=[
        VideoGenerationConfig(provider="replicate", model="google/veo-3", api_key=REPLICATE_KEY),
        VideoGenerationConfig(provider="openai", model="sora-2-turbo", api_key=OPENAI_KEY),
    ]
)

response = generate_video(config, request)
# Fal → Replicate → OpenAI. First success wins.
```

The orchestrator tries Fal first. If it gets a retryable error — a timeout, a 5xx, a rate limit — it falls over to Replicate. Then OpenAI. The full attempt history is captured in `execution_metadata`. When Fal has a 2am outage, your pipeline keeps running.

**4. Mock Provider**

Testing a video pipeline without the mock provider means burning API credits and waiting 30–60 seconds per test. One line fixes this:

```python
config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/veo3.1",
    api_key="...",
    mock=MockConfig(enabled=True)
)

response = generate_video(config, request)
assert response.is_mock is True  # No API call. Instant.
```

Your test suite runs in milliseconds. Swap `enabled=False` in production.
```

**Step 2: Verify build**
```bash
bun run build
```

**Step 3: Commit**
```bash
git add blogs/blog/2026-02-24-tarash-gateway.mdx
git commit -m "📝 Add fallback chains and mock provider sections"
```

---

### Task 6: Write Section 4 — Real-World Challenges

**Files:**
- Modify: `blogs/blog/2026-02-24-tarash-gateway.mdx`

**Step 1: Append the challenges section**

```mdx
## Real-World Challenges We Faced

Building Tarash Gateway wasn't just about wiring up SDKs. We hit some genuinely hard problems.

### The Async Client Caching Problem

Different provider SDKs have different opinions about async client reuse. Fal's async client throws if you reuse it across different asyncio event loops — a problem that surfaces in test suites and frameworks like FastAPI that manage their own loops. OpenAI's async client is explicitly designed for reuse.

**Our Fix:** Provider-aware caching strategies. Fal caches only sync clients and creates async clients fresh per call. OpenAI caches both. The interface is identical on the outside — one `generate_video_async()` call — but the strategy underneath is tuned per provider. You never think about event loops. The gateway does.

### The Model Variant Matching Problem

Replicate appends version hashes to model names: `minimax/video-01:abc123def456...`. The hash changes with every model update. A registry keyed on exact model names means updating it with every Replicate release — a maintenance burden that compounds with every new model.

**Our Fix:** Longest-prefix matching. Register `"minimax/"` as the key and it matches any Minimax model string, regardless of version hash. Add a new Replicate model? Register the prefix once. No hash maintenance, no registry drift.
```

**Step 2: Verify build**
```bash
bun run build
```

**Step 3: Commit**
```bash
git add blogs/blog/2026-02-24-tarash-gateway.mdx
git commit -m "📝 Add real-world challenges section"
```

---

### Task 7: Write Section 5 and 6 — When to Use + How to Use It

**Files:**
- Modify: `blogs/blog/2026-02-24-tarash-gateway.mdx`

**Step 1: Append the when-to-use and usage sections**

```mdx
## When to Use Tarash Gateway

Tarash Gateway is for teams building serious AI video pipelines. It excels when:

*   **You use more than one provider.** Even if you're on Fal today, having the abstraction means switching costs zero.
*   **You need resilience.** A single-provider pipeline is a single point of failure. Fallback chains protect you.
*   **You're running cost or quality experiments.** Swap providers with one config change. A/B test at scale.
*   **You need a fast test loop.** The mock provider makes your test suite instant and free.

## How to Use It

Tarash Gateway is available on PyPI. Check out the project on GitHub: [https://github.com/vertexcover-io/tarash](https://github.com/vertexcover-io/tarash)

### Installation

```bash
pip install tarash-gateway

# Install with your provider's SDK
pip install tarash-gateway[fal]
pip install tarash-gateway[openai]
pip install tarash-gateway[replicate]
```

### Basic Usage

```python
from tarash.tarash_gateway import generate_video, VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/veo3.1",
    api_key="your-fal-api-key"
)

request = VideoGenerationRequest(
    prompt="A calm lake at sunrise, mist rising off the water",
    duration_seconds=4,
    aspect_ratio="16:9"
)

response = generate_video(config, request)
print(response.video)  # Direct URL to the generated video
```

### With Progress Tracking

```python
def on_progress(update):
    if update.progress_percent:
        print(f"[{update.status}] {update.progress_percent:.0f}%")

response = generate_video(config, request, on_progress=on_progress)
```

### With a Fallback Chain

```python
config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/veo3.1",
    api_key=FAL_KEY,
    fallback_configs=[
        VideoGenerationConfig(provider="replicate", model="google/veo-3", api_key=REPLICATE_KEY),
    ]
)

response = generate_video(config, request)
# Fal first. Replicate if Fal fails. You don't have to think about it.
```
```

**Step 2: Verify build**
```bash
bun run build
```

**Step 3: Commit**
```bash
git add blogs/blog/2026-02-24-tarash-gateway.mdx
git commit -m "📝 Add when-to-use and usage sections"
```

---

### Task 8: Write Section 7 — Conclusion

**Files:**
- Modify: `blogs/blog/2026-02-24-tarash-gateway.mdx`

**Step 1: Append the conclusion**

```mdx
## Conclusion

The AI video provider landscape moves fast. New models drop every few weeks. Pricing shifts. Providers go down. Today's best model might not be tomorrow's.

Tarash Gateway is the abstraction layer that keeps you flexible. Write your integration once. Switch providers when you need to. Add fallbacks for resilience. Test for free with the mock provider.

The code is open source. Give it a try and stop rewriting integrations.

```bash
pip install tarash-gateway
```
```

**Step 2: Verify the full build**
```bash
bun run build
```
Expected: Clean build, no broken links, no MDX errors.

**Step 3: Commit**
```bash
git add blogs/blog/2026-02-24-tarash-gateway.mdx
git commit -m "📝 Add conclusion to Tarash Gateway blog"
```

---

### Task 9: Final review pass

**Files:**
- Read: `blogs/blog/2026-02-24-tarash-gateway.mdx` (full file)
- Read: `blogs/blog/2025-11-23-scene-flow.mdx` (for tone comparison)

**Step 1: Check against SceneFlow tone**

Read both files and verify:
- [ ] First-person plural throughout ("we built", "we tried", "we searched")
- [ ] Short paragraphs (3–5 sentences max)
- [ ] "Our Fix:" pattern used in challenges section
- [ ] No marketing fluff — specific and concrete throughout
- [ ] `{/* truncate */}` is present after the hook
- [ ] TLDR component renders correctly
- [ ] All code blocks have language tags (` ```python `, ` ```bash `)
- [ ] Blog starts with a story, not a definition

**Step 2: Final build**
```bash
bun run build
```

**Step 3: Commit any corrections**
```bash
git add blogs/blog/2026-02-24-tarash-gateway.mdx
git commit -m "📝 Final review pass on Tarash Gateway blog"
```
