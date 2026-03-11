"""Shared test configuration and fixtures."""

import subprocess
from unittest.mock import AsyncMock

import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--e2e",
        action="store_true",
        default=False,
        help="Run end-to-end tests (requires FFmpeg)",
    )


def pytest_collection_modifyitems(config, items):
    if not config.getoption("--e2e"):
        skip_e2e = pytest.mark.skip(reason="Need --e2e flag to run")
        for item in items:
            if "e2e" in item.keywords:
                item.add_marker(skip_e2e)


@pytest.fixture(scope="session")
def ffmpeg_available():
    """Check FFmpeg is available, skip if not."""
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"], capture_output=True, check=False
        )
        if result.returncode != 0:
            pytest.skip("FFmpeg not available")
    except FileNotFoundError:
        pytest.skip("FFmpeg not installed")


def make_async_proc(returncode=0, stdout=b"", stderr=b""):
    """Create a mock async subprocess process."""
    proc = AsyncMock()
    proc.returncode = returncode
    proc.communicate.return_value = (stdout, stderr)
    return proc
