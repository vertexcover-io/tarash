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
