"""Shared pytest configuration and fixtures for all tests."""

import os
import warnings

import pytest


def pytest_addoption(parser):
    """Add custom command-line options."""
    parser.addoption(
        "--e2e",
        action="store_true",
        default=False,
        help="Run only e2e tests (default: run only unit tests)",
    )


def pytest_configure(config):
    """Configure pytest with custom markers and settings."""
    config.addinivalue_line(
        "markers",
        "unit: Unit tests that use mocks and don't make real API calls",
    )
    config.addinivalue_line(
        "markers",
        "e2e: End-to-end tests that make real API calls",
    )
    config.addinivalue_line(
        "markers",
        "slow: Tests that take a long time to run",
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers and handle skips."""
    # Check for API keys
    fal_key_available = bool(os.getenv("FAL_KEY"))
    openai_key_available = bool(os.getenv("OPENAI_API_KEY"))

    # Get --e2e flag value
    run_e2e = config.getoption("--e2e")

    for item in items:
        # Automatically mark tests based on their directory location
        if "/e2e/" in item.nodeid:
            item.add_marker(pytest.mark.e2e)
        elif "/unit/" in item.nodeid:
            item.add_marker(pytest.mark.unit)

        # By default, skip e2e tests unless --e2e flag is provided
        if not run_e2e and "e2e" in item.keywords:
            item.add_marker(
                pytest.mark.skip(
                    reason="E2E tests skipped by default. Use --e2e to run them."
                )
            )

        # When --e2e flag is provided, skip unit tests
        if run_e2e and "unit" in item.keywords:
            item.add_marker(
                pytest.mark.skip(reason="Unit tests skipped when --e2e flag is used.")
            )

        # Skip e2e tests if API keys not available (only when running e2e tests)
        if run_e2e and "e2e" in item.keywords:
            if "fal" in item.nodeid.lower() and not fal_key_available:
                item.add_marker(
                    pytest.mark.skip(reason="FAL_KEY environment variable not set")
                )
            if "openai" in item.nodeid.lower() and not openai_key_available:
                item.add_marker(
                    pytest.mark.skip(
                        reason="OPENAI_API_KEY environment variable not set"
                    )
                )


@pytest.fixture(scope="session")
def anyio_backend():
    """Configure anyio backend for async tests."""
    return "asyncio"


@pytest.fixture(autouse=True)
def suppress_warnings():
    """Suppress specific warnings during tests."""
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    warnings.filterwarnings("ignore", category=PendingDeprecationWarning)
