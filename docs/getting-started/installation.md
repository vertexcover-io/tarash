# Installation

## Requirements

- Python 3.12+
- `uv` (recommended) or `pip`

## Install

Install with all provider dependencies:

=== "uv"

    ```bash
    uv add tarash-gateway[all]
    ```

=== "pip"

    ```bash
    pip install tarash-gateway[all]
    ```

## Install a single provider

If you only need one provider, install just that extra:

=== "uv"

    ```bash
    uv add tarash-gateway[fal]
    uv add tarash-gateway[openai]
    uv add tarash-gateway[runway]
    ```

=== "pip"

    ```bash
    pip install tarash-gateway[fal]
    pip install tarash-gateway[openai]
    pip install tarash-gateway[runway]
    ```

## Available extras

| Extra | Providers included |
|---|---|
| `fal` | Fal.ai |
| `openai` | OpenAI Sora |
| `runway` | Runway Gen-3 |
| `veo3` | Google Veo3 (Vertex AI) |
| `google` | Google Cloud AI Platform |
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
