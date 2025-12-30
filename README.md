# Tarash

A unified wrapper library for various AI code editing tools, organized as a monorepo.

## Overview

Tarash provides a consistent interface for interacting with multiple AI-powered code editing tools, making it easier to integrate and switch between different AI assistants in your development workflow.

## Structure

This project uses a workspace-based monorepo structure:

```
tarash/
├── packages/          # Individual tool wrappers
└── pyproject.toml     # Workspace configuration
```

## Development

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager

### Setup

```bash
# Clone the repository
git clone <repository-url>
cd tarash

# Install dependencies
uv sync

# Install development dependencies
uv sync --group dev
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

[Add your license here]
