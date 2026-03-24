# Learning: Pydantic incompatibility with Python 3.14 RC2 requires --noconftest workaround

**Date:** 2026-03-24
**Task:** refactor-execute-methods
**Category:** pipeline-friction

## Problem

The project's `uv` environment resolves to Python 3.14.0rc2, but Pydantic (used heavily in models.py and throughout the codebase) is not yet compatible with Python 3.14 RC releases. This causes `ImportError` or `TypeError` crashes at collection time when pytest loads `conftest.py` or any module that transitively imports Pydantic models.

## Workaround used

- Run tests with `--noconftest` to prevent pytest from loading the project-wide conftest (which imports Pydantic models).
- Write new tests that avoid importing from `models.py` or any module that triggers Pydantic model compilation. Use plain `@dataclass` mocks and duck-typed stand-ins instead.
- This keeps new test files self-contained and runnable even when the broader test suite is broken.

## Command pattern

```bash
uv run python -m pytest tests/path/to/new_test.py --noconftest -v
```

## When this stops being relevant

Once Pydantic ships a release with full Python 3.14 support (or the project pins to Python <=3.13), this workaround is unnecessary and tests should run normally without `--noconftest`.

## Broader takeaway

When testing refactored internals that don't depend on Pydantic models at runtime, duck-typed dataclass mocks are a resilient approach — they decouple tests from the model layer entirely and survive dependency breakages like this one.
