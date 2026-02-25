# Minimax (via Fal.ai)

Hailuo series video models hosted on Fal.ai.

## Quick Example

```python
from tarash.tarash_gateway import generate_video
from tarash.tarash_gateway.models import VideoGenerationConfig, VideoGenerationRequest

config = VideoGenerationConfig(
    provider="fal",
    model="fal-ai/minimax",
    api_key="YOUR_FAL_KEY",
)

request = VideoGenerationRequest(
    prompt="A sunrise over mountain peaks, cinematic",
    duration_seconds=6,
)

response = generate_video(config, request)
print(response.video)
```

---

## Supported Models

| Model prefix | Duration | Image-to-Video | Notes |
|---|---|:---:|---|
| `fal-ai/minimax` | `6s`, `10s` | ✅ | Hailuo series; `prompt_optimizer` support |

---

## Parameters

| Parameter | Required | Supported | Notes |
|---|:---:|:---:|---|
| `prompt` | ✅ | ✅ | |
| `duration_seconds` | — | ✅ | `6` or `10` |
| `image_list` (reference) | — | ✅ | Image-to-video |
| `enhance_prompt` | — | ✅ | Mapped to `prompt_optimizer` |

---

## Prompt Optimizer

```python
# Enable prompt optimizer
request = VideoGenerationRequest(
    prompt="A serene ocean at dusk",
    duration_seconds=6,
    enhance_prompt=True,        # mapped to prompt_optimizer
)

# Or pass directly via extra_params
request = VideoGenerationRequest(
    prompt="A serene ocean at dusk",
    duration_seconds=6,
    extra_params={"prompt_optimizer": True},
)
```

---