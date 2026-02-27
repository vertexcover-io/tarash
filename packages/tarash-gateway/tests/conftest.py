"""Shared pytest configuration and fixtures for all tests."""

import logging
import os
import warnings

import pytest

# Import MockConfig to trigger model_rebuild for config models
# This is needed because config models use MockConfig as a forward reference
from tarash.tarash_gateway.mock import MockConfig  # noqa: F401
from tarash.tarash_gateway.models import (
    AudioGenerationConfig,
    ImageGenerationConfig,
    VideoGenerationConfig,
)

# Rebuild models to resolve forward references
VideoGenerationConfig.model_rebuild()
ImageGenerationConfig.model_rebuild()
AudioGenerationConfig.model_rebuild()


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
    replicate_key_available = bool(os.getenv("REPLICATE_API_KEY"))
    elevenlabs_key_available = bool(os.getenv("ELEVENLABS_API_KEY"))

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
            if "replicate" in item.nodeid.lower() and not replicate_key_available:
                item.add_marker(
                    pytest.mark.skip(
                        reason="REPLICATE_API_KEY environment variable not set"
                    )
                )
            if "elevenlabs" in item.nodeid.lower() and not elevenlabs_key_available:
                item.add_marker(
                    pytest.mark.skip(
                        reason="ELEVENLABS_API_KEY environment variable not set"
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


@pytest.fixture(scope="session", autouse=True)
def configure_logging(request):
    """Configure logging to show only tarash_gateway logs, suppress third-party libraries."""
    # Get the configured log level from pytest config (CLI overrides config file)
    # Try CLI option first (--log-cli-level=DEBUG)
    config_level = request.config.getoption("--log-cli-level", default=None)

    # If not set via CLI, fall back to pyproject.toml value
    if config_level is None:
        config_level = request.config.getini("log_cli_level")

    # Convert string level to logging constant (e.g., "INFO" -> logging.INFO)
    if config_level:
        level = getattr(logging, str(config_level).upper(), logging.INFO)
    else:
        level = logging.INFO

    # Set root logger to ERROR to suppress third-party libraries
    logging.getLogger().setLevel(logging.ERROR)

    # Suppress all existing loggers except tarash.tarash_gateway
    for logger_name in logging.root.manager.loggerDict:
        if not logger_name.startswith("tarash.tarash_gateway"):
            logging.getLogger(logger_name).setLevel(logging.ERROR)

    # Enable configured level for tarash_gateway specifically
    logging.getLogger("tarash.tarash_gateway").setLevel(level)
