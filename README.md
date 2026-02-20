# Tarash

A monorepo of Python SDKs for AI-powered media generation.

## Packages

| Package | Description |
|---------|-------------|
| [`tarash-gateway`](packages/tarash-gateway/) | Unified SDK for AI video and image generation across multiple providers |
| [`tarash-captions`](packages/tarash-captions/) | Caption generation SDK with multi-provider support |

## Development

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
git clone <repository-url>
cd tarash

# Install all workspace dependencies
uv sync
```

### Adding Workspace Packages

Create new packages in the `packages/` directory. Each package must have its own `pyproject.toml`.

To use a workspace package as a dependency:

```toml
dependencies = ["package-name"]

[tool.uv.sources]
package-name = { workspace = true }
```

## License

MIT
