"""Unit tests for video utility functions."""

import base64
import pytest
from unittest.mock import Mock, patch
from typing_extensions import TypedDict

import httpx

from tarash.tarash_gateway.utils import (
    convert_to_data_url,
    download_media_from_url,
    get_filename_from_url,
    validate_duration,
    validate_model_params,
)
from tarash.tarash_gateway.exceptions import (
    HTTPError,
    TarashException,
    ValidationError,
)


class TestValidateModelParams:
    """Tests for validate_model_params function."""

    def test_validate_valid_params(self):
        """Test validation with valid parameters."""

        class SampleSchema(TypedDict):
            prompt: str
            duration: int

        data = {"prompt": "test prompt", "duration": 5}
        result = validate_model_params(
            schema=SampleSchema,
            data=data,
            provider="test-provider",
            model="test-model",
        )

        assert result == {"prompt": "test prompt", "duration": 5}

    def test_validate_params_removes_none_values(self):
        """Test that None values are removed from validated dict."""

        class SampleSchema(TypedDict, total=False):
            prompt: str
            duration: int | None

        data = {"prompt": "test prompt", "duration": None}
        result = validate_model_params(
            schema=SampleSchema,
            data=data,
            provider="test-provider",
            model="test-model",
        )

        assert result == {"prompt": "test prompt"}
        assert "duration" not in result

    def test_validate_params_invalid_data(self):
        """Test validation with invalid data raises ValidationError."""

        class SampleSchema(TypedDict):
            prompt: str
            duration: int

        data = {"prompt": "test prompt", "duration": "invalid"}  # Should be int

        with pytest.raises(ValidationError) as exc_info:
            validate_model_params(
                schema=SampleSchema,
                data=data,
                provider="test-provider",
                model="test-model",
            )

        assert "Invalid model_params for test-model" in str(exc_info.value)
        assert exc_info.value.provider == "test-provider"


class TestConvertToDataUrl:
    """Tests for convert_to_data_url function."""

    def test_convert_image_bytes_to_data_url(self):
        """Test converting image bytes to data URL."""
        image_bytes = b"fake image data"
        media_content = {
            "content": image_bytes,
            "content_type": "image/jpeg",
        }

        result = convert_to_data_url(media_content)

        expected_base64 = base64.b64encode(image_bytes).decode("utf-8")
        expected_url = f"data:image/jpeg;base64,{expected_base64}"

        assert result == expected_url

    def test_convert_video_bytes_to_data_url(self):
        """Test converting video bytes to data URL."""
        video_bytes = b"fake video data"
        media_content = {
            "content": video_bytes,
            "content_type": "video/mp4",
        }

        result = convert_to_data_url(media_content)

        expected_base64 = base64.b64encode(video_bytes).decode("utf-8")
        expected_url = f"data:video/mp4;base64,{expected_base64}"

        assert result == expected_url

    def test_convert_empty_bytes_to_data_url(self):
        """Test converting empty bytes to data URL."""
        media_content = {
            "content": b"",
            "content_type": "image/png",
        }

        result = convert_to_data_url(media_content)

        # Empty bytes should produce empty base64
        assert result == "data:image/png;base64,"


class TestDownloadMediaFromUrl:
    """Tests for download_media_from_url function."""

    def test_download_media_success(self):
        """Test successful media download."""
        test_url = "https://example.com/image.jpg"
        test_content = b"fake image content"
        test_content_type = "image/jpeg"

        mock_response = Mock()
        mock_response.content = test_content
        mock_response.headers = {"content-type": test_content_type}
        mock_response.raise_for_status.return_value = mock_response

        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=None)
            mock_client.get = Mock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            content, content_type = download_media_from_url(test_url, "test-provider")

            assert content == test_content
            assert content_type == test_content_type
            mock_client.get.assert_called_once_with(test_url)

    def test_download_media_default_content_type(self):
        """Test download with missing content-type header."""
        test_url = "https://example.com/media"
        test_content = b"fake media content"

        mock_response = Mock()
        mock_response.content = test_content
        mock_response.headers = {}  # No content-type header
        mock_response.raise_for_status.return_value = mock_response

        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=None)
            mock_client.get = Mock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            content, content_type = download_media_from_url(test_url, "test-provider")

            assert content == test_content
            assert content_type == "application/octet-stream"

    def test_download_media_http_error(self):
        """Test download with HTTP error status."""
        test_url = "https://example.com/notfound.jpg"

        mock_response = Mock()
        mock_response.status_code = 404
        mock_response.content = b"Not Found"

        http_error = httpx.HTTPStatusError(
            "404 Not Found",
            request=Mock(),
            response=mock_response,
        )

        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=None)
            mock_client.get = Mock(side_effect=http_error)
            mock_client_class.return_value = mock_client

            with pytest.raises(HTTPError) as exc_info:
                download_media_from_url(test_url, "test-provider")

            assert exc_info.value.provider == "test-provider"
            assert exc_info.value.status_code == 404
            assert "Failed to download media from URL" in str(exc_info.value)

    def test_download_media_http_500_error(self):
        """Test download with HTTP 500 server error."""
        test_url = "https://example.com/error.jpg"

        mock_response = Mock()
        mock_response.status_code = 500
        mock_response.content = b"Internal Server Error"

        http_error = httpx.HTTPStatusError(
            "500 Internal Server Error",
            request=Mock(),
            response=mock_response,
        )

        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=None)
            mock_client.get = Mock(side_effect=http_error)
            mock_client_class.return_value = mock_client

            with pytest.raises(HTTPError) as exc_info:
                download_media_from_url(test_url, "test-provider")

            assert exc_info.value.status_code == 500

    def test_download_media_network_error(self):
        """Test download with network error (non-HTTP exception)."""
        test_url = "https://example.com/image.jpg"

        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=None)
            mock_client.get = Mock(side_effect=Exception("Network error"))
            mock_client_class.return_value = mock_client

            with pytest.raises(TarashException) as exc_info:
                download_media_from_url(test_url, "test-provider")

            assert exc_info.value.provider == "test-provider"
            assert "Failed to download media from URL" in str(exc_info.value)

    def test_download_media_timeout(self):
        """Test download with timeout configuration."""
        test_url = "https://example.com/image.jpg"
        test_content = b"fake content"

        mock_response = Mock()
        mock_response.content = test_content
        mock_response.headers = {"content-type": "image/jpeg"}
        mock_response.raise_for_status.return_value = mock_response

        with patch("httpx.Client") as mock_client_class:
            mock_client = Mock()
            mock_client.__enter__ = Mock(return_value=mock_client)
            mock_client.__exit__ = Mock(return_value=None)
            mock_client.get = Mock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            download_media_from_url(test_url, "test-provider")

            # Verify client was created with 30 second timeout
            mock_client_class.assert_called_once_with(timeout=30.0)


class TestGetFilenameFromUrl:
    """Tests for get_filename_from_url function."""

    def test_get_filename_from_simple_url(self):
        """Test extracting filename from simple URL."""
        url = "https://example.com/path/to/video.mp4"
        result = get_filename_from_url(url)
        assert result == "video.mp4"

    def test_get_filename_from_url_with_query_params(self):
        """Test extracting filename from URL with query parameters."""
        url = "https://example.com/videos/output.mp4?token=abc123"
        result = get_filename_from_url(url)
        assert result == "output.mp4"

    def test_get_filename_from_url_no_extension(self):
        """Test extracting filename without extension."""
        url = "https://example.com/media/download"
        result = get_filename_from_url(url)
        assert result == "download"

    def test_get_filename_from_url_no_filename(self):
        """Test URL with trailing slash (no filename) returns default."""
        url = "https://example.com/path/"
        result = get_filename_from_url(url)
        assert result == "media"

    def test_get_filename_from_url_root_path(self):
        """Test URL with root path returns default."""
        url = "https://example.com/"
        result = get_filename_from_url(url)
        assert result == "media"

    def test_get_filename_from_url_no_path(self):
        """Test URL with no path returns default."""
        url = "https://example.com"
        result = get_filename_from_url(url)
        assert result == "media"

    def test_get_filename_from_url_invalid_url(self):
        """Test invalid URL returns default filename."""
        url = "not-a-valid-url"
        result = get_filename_from_url(url)
        # Should not crash, should return default
        assert result == "media" or result == "not-a-valid-url"

    def test_get_filename_from_url_complex_path(self):
        """Test complex path with multiple segments."""
        url = "https://cdn.example.com/users/123/videos/final_output.mp4"
        result = get_filename_from_url(url)
        assert result == "final_output.mp4"


class TestValidateDuration:
    """Tests for validate_duration function."""

    def test_validate_duration_valid_value(self):
        """Test validation with valid duration value."""
        result = validate_duration(5, [4, 5, 10], "test-provider", "test-model")
        assert result == 5

    def test_validate_duration_none_value(self):
        """Test validation with None value returns None."""
        result = validate_duration(None, [4, 5, 10], "test-provider", "test-model")
        assert result is None

    def test_validate_duration_invalid_value(self):
        """Test validation with invalid duration raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            validate_duration(7, [4, 5, 10], "test-provider", "test-model")

        assert "Invalid duration for test-provider (test-model): 7 seconds" in str(
            exc_info.value
        )
        assert "Allowed values: 4, 5, 10" in str(exc_info.value)
        assert exc_info.value.provider == "test-provider"

    def test_validate_duration_no_model_name(self):
        """Test validation error message without model name."""
        with pytest.raises(ValidationError) as exc_info:
            validate_duration(3, [5, 10], "openai")

        assert "Invalid duration for openai: 3 seconds" in str(exc_info.value)
        assert "Allowed values: 5, 10" in str(exc_info.value)

    def test_validate_duration_first_allowed_value(self):
        """Test validation with first allowed value."""
        result = validate_duration(2, [2, 4, 8], "fal", "minimax")
        assert result == 2

    def test_validate_duration_last_allowed_value(self):
        """Test validation with last allowed value."""
        result = validate_duration(12, [4, 8, 12], "fal", "minimax")
        assert result == 12

    def test_validate_duration_single_allowed_value(self):
        """Test validation with single allowed value."""
        result = validate_duration(10, [10], "provider")
        assert result == 10

        with pytest.raises(ValidationError):
            validate_duration(5, [10], "provider")
