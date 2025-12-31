"""Unit tests for FieldMapper, apply_field_mappers, and duration_field_mapper."""

import pytest

from tarash.tarash_gateway.video.exceptions import ValidationError
from tarash.tarash_gateway.video.models import VideoGenerationRequest
from tarash.tarash_gateway.video.providers.fal import (
    FieldMapper,
    apply_field_mappers,
    duration_field_mapper,
)


# ==================== FieldMapper Tests ====================


def test_field_mapper_initialization():
    """Test FieldMapper dataclass initialization with default values."""

    def dummy_converter(request, value):
        return value

    mapper = FieldMapper(source_field="prompt", converter=dummy_converter)

    assert mapper.source_field == "prompt"
    assert mapper.converter is dummy_converter
    assert mapper.required is False  # Default value


def test_field_mapper_with_required_field():
    """Test FieldMapper initialization with required=True."""

    def dummy_converter(request, value):
        return value

    mapper = FieldMapper(
        source_field="prompt", converter=dummy_converter, required=True
    )

    assert mapper.source_field == "prompt"
    assert mapper.converter is dummy_converter
    assert mapper.required is True


# ==================== apply_field_mappers Tests ====================


def test_apply_field_mappers_with_all_fields_present():
    """Test applying mappers when all fields are present."""
    request = VideoGenerationRequest(
        prompt="Test prompt", duration_seconds=5, aspect_ratio="16:9"
    )

    field_mappers = {
        "prompt": FieldMapper(
            source_field="prompt", converter=lambda req, val: val.upper()
        ),
        "duration": FieldMapper(
            source_field="duration_seconds", converter=lambda req, val: f"{val}s"
        ),
        "aspect_ratio": FieldMapper(
            source_field="aspect_ratio", converter=lambda req, val: val
        ),
    }

    result = apply_field_mappers(field_mappers, request)

    assert result == {
        "prompt": "TEST PROMPT",
        "duration": "5s",
        "aspect_ratio": "16:9",
    }


def test_apply_field_mappers_with_optional_fields_missing():
    """Test that missing optional fields are excluded from result."""
    request = VideoGenerationRequest(prompt="Test prompt")

    field_mappers = {
        "prompt": FieldMapper(
            source_field="prompt", converter=lambda req, val: val, required=True
        ),
        "duration": FieldMapper(
            source_field="duration_seconds", converter=lambda req, val: val
        ),  # Optional
        "aspect_ratio": FieldMapper(
            source_field="aspect_ratio", converter=lambda req, val: val
        ),  # Optional
    }

    result = apply_field_mappers(field_mappers, request)

    assert result == {"prompt": "Test prompt"}
    assert "duration" not in result
    assert "aspect_ratio" not in result


def test_apply_field_mappers_required_field_missing_raises_error():
    """Test that missing required field raises ValueError."""
    request = VideoGenerationRequest(prompt="Test prompt")  # duration_seconds is None

    field_mappers = {
        "duration": FieldMapper(
            source_field="duration_seconds",
            converter=lambda req, val: val,
            required=True,
        ),
    }

    with pytest.raises(
        ValueError, match="Required field 'duration_seconds' is missing or None"
    ):
        apply_field_mappers(field_mappers, request)


def test_apply_field_mappers_converter_returns_none_for_required_field():
    """Test that converter returning None for required field raises ValueError."""
    request = VideoGenerationRequest(prompt="Test prompt", duration_seconds=5)

    field_mappers = {
        "duration": FieldMapper(
            source_field="duration_seconds",
            converter=lambda req, val: None,  # Always returns None
            required=True,
        ),
    }

    with pytest.raises(
        ValueError, match="Required field 'duration' cannot be None after conversion"
    ):
        apply_field_mappers(field_mappers, request)


def test_apply_field_mappers_excludes_none_values():
    """Test that None values from converters are excluded from result."""
    request = VideoGenerationRequest(
        prompt="Test prompt", duration_seconds=5, aspect_ratio="16:9"
    )

    field_mappers = {
        "prompt": FieldMapper(source_field="prompt", converter=lambda req, val: val),
        "duration": FieldMapper(
            source_field="duration_seconds",
            converter=lambda req, val: None,  # Returns None
        ),
        "aspect_ratio": FieldMapper(
            source_field="aspect_ratio", converter=lambda req, val: val
        ),
    }

    result = apply_field_mappers(field_mappers, request)

    assert result == {"prompt": "Test prompt", "aspect_ratio": "16:9"}
    assert "duration" not in result


def test_apply_field_mappers_excludes_empty_lists():
    """Test that empty lists are excluded from result."""
    request = VideoGenerationRequest(prompt="Test prompt", image_list=[])

    field_mappers = {
        "image_urls": FieldMapper(
            source_field="image_list", converter=lambda req, val: val
        ),
    }

    result = apply_field_mappers(field_mappers, request)

    assert result == {}
    assert "image_urls" not in result


def test_apply_field_mappers_excludes_empty_dicts():
    """Test that empty dicts are excluded from result."""
    request = VideoGenerationRequest(prompt="Test prompt", extra_params={})

    field_mappers = {
        "params": FieldMapper(
            source_field="extra_params", converter=lambda req, val: val
        ),
    }

    result = apply_field_mappers(field_mappers, request)

    assert result == {}
    assert "params" not in result


def test_apply_field_mappers_with_complex_converter():
    """Test field mapper with complex converter logic that accesses request."""
    request = VideoGenerationRequest(
        prompt="Test prompt", duration_seconds=5, aspect_ratio="16:9"
    )

    def complex_converter(req, val):
        # Converter can access other request fields
        if req.aspect_ratio == "16:9":
            return f"{val}s_widescreen"
        return f"{val}s"

    field_mappers = {
        "duration": FieldMapper(
            source_field="duration_seconds", converter=complex_converter
        ),
    }

    result = apply_field_mappers(field_mappers, request)

    assert result == {"duration": "5s_widescreen"}


# ==================== duration_field_mapper Tests ====================


def test_duration_field_mapper_string_format_without_validation():
    """Test duration_field_mapper creates mapper for string format without validation."""
    mapper = duration_field_mapper("str")

    assert mapper.source_field == "duration_seconds"
    assert mapper.required is False

    # Test converter
    request = VideoGenerationRequest(prompt="test", duration_seconds=5)
    result = mapper.converter(request, 5)
    assert result == "5s"


def test_duration_field_mapper_int_format_without_validation():
    """Test duration_field_mapper creates mapper for int format without validation."""
    mapper = duration_field_mapper("int")

    assert mapper.source_field == "duration_seconds"
    assert mapper.required is False

    # Test converter
    request = VideoGenerationRequest(prompt="test", duration_seconds=10)
    result = mapper.converter(request, 10)
    assert result == 10


def test_duration_field_mapper_string_format_with_validation_success():
    """Test duration_field_mapper with string format and allowed values (valid)."""
    mapper = duration_field_mapper("str", allowed_values=["6s", "10s"])

    request = VideoGenerationRequest(prompt="test", duration_seconds=6)

    # Valid value
    result = mapper.converter(request, 6)
    assert result == "6s"

    # Another valid value
    result = mapper.converter(request, 10)
    assert result == "10s"


def test_duration_field_mapper_string_format_with_validation_failure():
    """Test duration_field_mapper with string format and allowed values (invalid)."""
    mapper = duration_field_mapper("str", allowed_values=["6s", "10s"])

    request = VideoGenerationRequest(prompt="test", duration_seconds=5)

    with pytest.raises(ValidationError) as exc_info:
        mapper.converter(request, 5)

    assert "Invalid duration" in str(exc_info.value)
    assert "5 seconds" in str(exc_info.value)
    assert "6, 10" in str(exc_info.value)


def test_duration_field_mapper_int_format_with_validation_success():
    """Test duration_field_mapper with int format and allowed values (valid)."""
    mapper = duration_field_mapper("int", allowed_values=[5, 10, 15])

    request = VideoGenerationRequest(prompt="test", duration_seconds=5)

    result = mapper.converter(request, 5)
    assert result == 5

    result = mapper.converter(request, 10)
    assert result == 10


def test_duration_field_mapper_int_format_with_validation_failure():
    """Test duration_field_mapper with int format and allowed values (invalid)."""
    mapper = duration_field_mapper("int", allowed_values=[5, 10, 15])

    request = VideoGenerationRequest(prompt="test", duration_seconds=7)

    with pytest.raises(ValidationError) as exc_info:
        mapper.converter(request, 7)

    assert "Invalid duration" in str(exc_info.value)
    assert "7 seconds" in str(exc_info.value)
    assert "5, 10, 15" in str(exc_info.value)


def test_duration_field_mapper_handles_none_value():
    """Test duration_field_mapper converter returns None when value is None."""
    mapper_str = duration_field_mapper("str")
    mapper_int = duration_field_mapper("int")

    request = VideoGenerationRequest(prompt="test")

    assert mapper_str.converter(request, None) is None
    assert mapper_int.converter(request, None) is None


def test_duration_field_mapper_none_bypasses_validation():
    """Test that None value bypasses validation even with allowed_values."""
    mapper = duration_field_mapper("str", allowed_values=["6s", "10s"])

    request = VideoGenerationRequest(prompt="test")

    # None should not raise validation error
    result = mapper.converter(request, None)
    assert result is None


# ==================== Integration Tests ====================


def test_duration_field_mapper_integration_with_apply_field_mappers():
    """Test duration_field_mapper works correctly with apply_field_mappers."""
    request = VideoGenerationRequest(prompt="Test", duration_seconds=6)

    field_mappers = {
        "prompt": FieldMapper(source_field="prompt", converter=lambda req, val: val),
        "duration": duration_field_mapper("str", allowed_values=["6s", "10s"]),
    }

    result = apply_field_mappers(field_mappers, request)

    assert result == {"prompt": "Test", "duration": "6s"}


def test_duration_field_mapper_integration_validation_error_propagates():
    """Test that validation error from duration_field_mapper propagates through apply_field_mappers."""
    request = VideoGenerationRequest(prompt="Test", duration_seconds=5)

    field_mappers = {
        "duration": duration_field_mapper("str", allowed_values=["6s", "10s"]),
    }

    with pytest.raises(ValidationError) as exc_info:
        apply_field_mappers(field_mappers, request)

    assert "Invalid duration" in str(exc_info.value)
    assert "5 seconds" in str(exc_info.value)


def test_multiple_field_mappers_with_different_types():
    """Test apply_field_mappers with various field types and converters."""
    request = VideoGenerationRequest(
        prompt="Test video",
        duration_seconds=10,
        aspect_ratio="16:9",
        image_list=[{"image": "http://example.com/img.jpg", "type": "reference"}],
        seed=42,
    )

    field_mappers = {
        "prompt": FieldMapper(
            source_field="prompt", converter=lambda req, val: val, required=True
        ),
        "duration": duration_field_mapper("str"),
        "aspect_ratio": FieldMapper(
            source_field="aspect_ratio", converter=lambda req, val: val
        ),
        "image_urls": FieldMapper(
            source_field="image_list",
            converter=lambda req, val: [img["image"] for img in val],
        ),
        "seed": FieldMapper(source_field="seed", converter=lambda req, val: val),
    }

    result = apply_field_mappers(field_mappers, request)

    assert result == {
        "prompt": "Test video",
        "duration": "10s",
        "aspect_ratio": "16:9",
        "image_urls": ["http://example.com/img.jpg"],
        "seed": 42,
    }
