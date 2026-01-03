# Claude Code Guidelines

This document contains guidelines for Claude Code when working on this project.

## Python Environment

**CRITICAL: Always use `uv` for Python commands**

This project uses `uv` as the package manager. ALL Python commands must be run through `uv`:

- ✅ **Correct**: `uv run pytest tests/`
- ✅ **Correct**: `uv run python -m pytest tests/`
- ✅ **Correct**: `uv run python script.py`
- ❌ **Wrong**: `python -m pytest tests/`
- ❌ **Wrong**: `pytest tests/`
- ❌ **Wrong**: `python script.py`

**Why this is critical:**
- Ensures correct virtual environment activation
- Uses project-specific dependencies
- Avoids conflicts with system Python
- Maintains reproducible builds

## Git Commit Guidelines

### Commit Message Format

- **Use GitHub emoji** at the start of commit messages to categorize changes:
  - ✨ `:sparkles:` New features
  - 🐛 `:bug:` Bug fixes
  - 📝 `:memo:` Documentation updates
  - ♻️ `:recycle:` Refactoring
  - ⚡ `:zap:` Performance improvements
  - ✅ `:white_check_mark:` Tests
  - 🔧 `:wrench:` Configuration changes
  - 🎨 `:art:` Code style/formatting
  - 🔥 `:fire:` Code removal
  - 🚀 `:rocket:` Deployment/release
  - 🔒 `:lock:` Security fixes

### Commit Message Style

- **Be concise**: Focus on the primary change in the commit
- **No AI attribution**: Do not mention "Claude Code" or AI assistance in commit messages
- **Primary change only**: Describe the main feature/bug/change, not every file modification
- **Imperative mood**: Use "Add feature" not "Added feature"

### Examples

Good:
```
✨ Add workspace configuration for monorepo
🐛 Fix dependency resolution in package loader
📝 Update installation instructions
♻️ Restructure error handling
```

Bad:
```
✨ Add workspace configuration, update pyproject.toml, modify README.md, and add CLAUDE.md

🤖 Generated with Claude Code

Co-Authored-By: Claude Sonnet 4.5 <noreply@anthropic.com>

Updated multiple files including workspace setup
```

### Commit Workflow

**CRITICAL: Always confirm before committing**

When creating any commit, you MUST follow this process:

1. **Show the user what will be committed**:
   - List all files that will be included in the commit
   - Display the proposed commit message
   - Show relevant diffs if helpful for context
   - **Do NOT include untracked files** unless user explicitly requests them

2. **Wait for explicit user confirmation**:
   - Do NOT execute `git commit` until the user approves
   - User must explicitly confirm (e.g., "yes", "ok", "commit it")

3. **Only then create the commit**:
   - Use the approved message exactly as shown
   - Keep commits atomic - one logical change per commit

4. **Focus on impact**: What does this change accomplish, not how it was done

## Python Coding Guidelines

### Type Hints

- **Use specific types over `Any`**: Prefer concrete types from third-party libraries
- **Import types from SDK/library**: Use actual type definitions when available
- **Fallback to `Any` only when necessary**: When SDK doesn't export types or types are truly dynamic
- **Document `Any` usage**: Add comment explaining why `Any` is needed

**Examples:**

Good:
```python
from runwayml import RunwayML, AsyncRunwayML
from runwayml.lib.polling import NewTaskCreatedResponse, AsyncNewTaskCreatedResponse

def _get_client(self, config: Config, client_type: str) -> RunwayML | AsyncRunwayML:
    ...

def _call_endpoint(self, client: RunwayML | AsyncRunwayML, ...) -> NewTaskCreatedResponse | AsyncNewTaskCreatedResponse:
    ...
```

Bad:
```python
from typing import Any

def _get_client(self, config: Config, client_type: str) -> Any:  # Too vague
    ...

def _call_endpoint(self, client: Any, ...) -> Any:  # Lost type information
    ...
```

When `Any` is appropriate:
```python
# SDK doesn't export Task type, must use Any
def _poll_task(self, task: Any) -> VideoResponse:  # Any is acceptable here
    """Poll task until completion.

    Args:
        task: Runway task object (type not exported by SDK)
    """
    ...
```

### SDK Import Handling

**CRITICAL: No placeholder exception types for missing SDKs**

When importing third-party SDK types and exceptions:

- **DO NOT** create placeholder/fallback exception types when imports fail
- **Let import errors propagate naturally** - if SDK isn't installed, provider shouldn't work
- Provider initialization will fail with clear ImportError, which is the desired behavior
- Tests should handle ImportError with `pytest.skip()` if SDK is optional

**Examples:**

Good:
```python
try:
    from openai import AsyncOpenAI, OpenAI
    from openai import exceptions as openai_exceptions

    APIStatusError = openai_exceptions.APIStatusError
    APIConnectionError = openai_exceptions.APIConnectionError
    APITimeoutError = openai_exceptions.APITimeoutError
except ImportError:
    pass  # Provider won't work without SDK - that's OK
```

Bad:
```python
try:
    from openai import exceptions as openai_exceptions
    APIStatusError = openai_exceptions.APIStatusError
except ImportError:
    # ❌ DON'T DO THIS - creates confusing behavior
    APIStatusError = type("APIStatusError", (Exception,), {})
```

**Rationale:**
- Provider code using these types won't run if `__init__` raises ImportError
- Placeholder types mask the real problem and create confusing errors
- Better to fail fast with clear error than fail later with cryptic behavior
