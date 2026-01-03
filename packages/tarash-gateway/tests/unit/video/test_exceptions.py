"""Tests for exception utilities and error classification."""

from tarash.tarash_gateway.video.exceptions import (
    ContentModerationError,
    GenerationFailedError,
    HTTPConnectionError,
    HTTPError,
    TimeoutError,
    ValidationError,
    is_retryable_error,
)


def test_is_retryable_error_validation_error_false():
    """ValidationError should not be retryable."""
    error = ValidationError("Invalid prompt", provider="fal", model="veo3.1")
    assert is_retryable_error(error) is False


def test_is_retryable_error_content_moderation_false():
    """ContentModerationError should not be retryable."""
    error = ContentModerationError(
        "Content policy violation", provider="fal", model="veo3.1"
    )
    assert is_retryable_error(error) is False


def test_is_retryable_error_generation_failed_true():
    """GenerationFailedError should be retryable."""
    error = GenerationFailedError("Video generation failed", provider="fal")
    assert is_retryable_error(error) is True


def test_is_retryable_error_timeout_true():
    """TimeoutError should be retryable."""
    error = TimeoutError("Request timed out", provider="fal", timeout_seconds=600)
    assert is_retryable_error(error) is True


def test_is_retryable_error_connection_error_true():
    """HTTPConnectionError should be retryable."""
    error = HTTPConnectionError("Network error", provider="fal")
    assert is_retryable_error(error) is True


def test_is_retryable_error_http_400_false():
    """HTTP 400 should not be retryable."""
    error = HTTPError("Bad request", provider="fal", model="veo3.1", status_code=400)
    assert is_retryable_error(error) is False


def test_is_retryable_error_http_401_false():
    """HTTP 401 should not be retryable."""
    error = HTTPError("Unauthorized", provider="fal", model="veo3.1", status_code=401)
    assert is_retryable_error(error) is False


def test_is_retryable_error_http_403_false():
    """HTTP 403 should not be retryable."""
    error = HTTPError("Forbidden", provider="fal", model="veo3.1", status_code=403)
    assert is_retryable_error(error) is False


def test_is_retryable_error_http_404_false():
    """HTTP 404 should not be retryable."""
    error = HTTPError("Not found", provider="fal", model="veo3.1", status_code=404)
    assert is_retryable_error(error) is False


def test_is_retryable_error_http_429_true():
    """HTTP 429 (rate limit) should be retryable."""
    error = HTTPError(
        "Rate limit exceeded", provider="fal", model="veo3.1", status_code=429
    )
    assert is_retryable_error(error) is True


def test_is_retryable_error_http_500_true():
    """HTTP 500 should be retryable."""
    error = HTTPError(
        "Internal server error", provider="fal", model="veo3.1", status_code=500
    )
    assert is_retryable_error(error) is True


def test_is_retryable_error_http_502_true():
    """HTTP 502 should be retryable."""
    error = HTTPError("Bad gateway", provider="fal", model="veo3.1", status_code=502)
    assert is_retryable_error(error) is True


def test_is_retryable_error_http_503_true():
    """HTTP 503 should be retryable."""
    error = HTTPError(
        "Service unavailable", provider="fal", model="veo3.1", status_code=503
    )
    assert is_retryable_error(error) is True


def test_is_retryable_error_http_504_true():
    """HTTP 504 should be retryable."""
    error = HTTPError(
        "Gateway timeout", provider="fal", model="veo3.1", status_code=504
    )
    assert is_retryable_error(error) is True


def test_is_retryable_error_unknown_exception_false():
    """Unknown exceptions should not be retryable."""
    error = ValueError("Some other error")
    assert is_retryable_error(error) is False
