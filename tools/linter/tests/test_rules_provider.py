"""Tests for TRH1xx provider method rules."""

from tarash_linter.models import ProviderInfo
from tarash_linter.rules import RuleContext
from tarash_linter.rules.provider import (
    TRH101,
    TRH102,
    TRH103,
    TRH104,
    TRH105,
    TRH106,
)


def _empty_context() -> RuleContext:
    return RuleContext(
        project_root=".",
        registry_mapping={},
        pricing_providers=set(),
        unit_test_files={},
        unit_test_functions={},
        e2e_test_files={},
    )


def _make_provider(name: str, methods: set[str]) -> ProviderInfo:
    return ProviderInfo(
        name=name,
        file=f"providers/{name}.py",
        class_name=f"{name.title()}ProviderHandler",
        class_line=10,
        methods=frozenset(methods),
    )


# --- TRH101: _get_client ---


def test_trh101_passes_with_get_client():
    rule = TRH101()
    provider = _make_provider(
        "fal",
        {"_get_client", "_handle_error", "generate_video", "generate_video_async"},
    )
    assert rule.check(provider, _empty_context()) == []


def test_trh101_fails_without_get_client():
    rule = TRH101()
    provider = _make_provider(
        "fal", {"_handle_error", "generate_video", "generate_video_async"}
    )
    violations = rule.check(provider, _empty_context())
    assert len(violations) == 1
    assert violations[0].code == "TRH101"
    assert "_get_client" in violations[0].message


# --- TRH102: _handle_error ---


def test_trh102_passes_with_handle_error():
    rule = TRH102()
    provider = _make_provider(
        "fal",
        {"_get_client", "_handle_error", "generate_video", "generate_video_async"},
    )
    assert rule.check(provider, _empty_context()) == []


def test_trh102_fails_without_handle_error():
    rule = TRH102()
    provider = _make_provider(
        "fal", {"_get_client", "generate_video", "generate_video_async"}
    )
    violations = rule.check(provider, _empty_context())
    assert len(violations) == 1
    assert violations[0].code == "TRH102"


# --- TRH103: generation method pair ---


def test_trh103_passes_with_valid_pair():
    rule = TRH103()
    provider = _make_provider(
        "fal", {"_get_client", "_handle_error", "generate_tts", "generate_tts_async"}
    )
    assert rule.check(provider, _empty_context()) == []


def test_trh103_fails_with_no_pair():
    rule = TRH103()
    provider = _make_provider("fal", {"_get_client", "_handle_error"})
    violations = rule.check(provider, _empty_context())
    assert len(violations) == 1
    assert violations[0].code == "TRH103"


def test_trh103_fails_with_sync_only():
    """Having only sync without async is not a valid pair."""
    rule = TRH103()
    provider = _make_provider("fal", {"_get_client", "_handle_error", "generate_video"})
    violations = rule.check(provider, _empty_context())
    assert len(violations) == 1
    assert violations[0].code == "TRH103"


# --- TRH104: _convert_request (video only) ---


def test_trh104_passes_for_video_provider_with_convert_request():
    rule = TRH104()
    provider = _make_provider(
        "fal",
        {
            "_get_client",
            "_handle_error",
            "_convert_request",
            "generate_video",
            "generate_video_async",
        },
    )
    assert rule.check(provider, _empty_context()) == []


def test_trh104_passes_for_video_provider_with_variant_convert_request():
    """Variant like _convert_video_request also satisfies TRH104."""
    rule = TRH104()
    provider = _make_provider(
        "xai",
        {
            "_get_client",
            "_handle_error",
            "_convert_video_request",
            "generate_video",
            "generate_video_async",
        },
    )
    assert rule.check(provider, _empty_context()) == []


def test_trh104_fails_for_video_provider_without_convert_request():
    rule = TRH104()
    provider = _make_provider(
        "fal",
        {
            "_get_client",
            "_handle_error",
            "generate_video",
            "generate_video_async",
        },
    )
    violations = rule.check(provider, _empty_context())
    assert len(violations) == 1
    assert violations[0].code == "TRH104"


def test_trh104_skips_non_video_provider():
    """TRH104 does not apply to audio-only providers."""
    rule = TRH104()
    provider = _make_provider(
        "cartesia",
        {
            "_get_client",
            "_handle_error",
            "generate_tts",
            "generate_tts_async",
        },
    )
    assert rule.check(provider, _empty_context()) == []


# --- TRH105: _convert_response (video only) ---


def test_trh105_fails_for_video_provider_without_convert_response():
    rule = TRH105()
    provider = _make_provider(
        "fal",
        {
            "_get_client",
            "_handle_error",
            "generate_video",
            "generate_video_async",
        },
    )
    violations = rule.check(provider, _empty_context())
    assert len(violations) == 1
    assert violations[0].code == "TRH105"


def test_trh105_skips_non_video_provider():
    rule = TRH105()
    provider = _make_provider(
        "cartesia",
        {
            "_get_client",
            "_handle_error",
            "generate_tts",
            "generate_tts_async",
        },
    )
    assert rule.check(provider, _empty_context()) == []


# --- TRH106: _validate_params (video only) ---


def test_trh106_fails_for_video_provider_without_validate_params():
    rule = TRH106()
    provider = _make_provider(
        "fal",
        {
            "_get_client",
            "_handle_error",
            "generate_video",
            "generate_video_async",
        },
    )
    violations = rule.check(provider, _empty_context())
    assert len(violations) == 1
    assert violations[0].code == "TRH106"


def test_trh106_skips_non_video_provider():
    rule = TRH106()
    provider = _make_provider(
        "cartesia",
        {
            "_get_client",
            "_handle_error",
            "generate_tts",
            "generate_tts_async",
        },
    )
    assert rule.check(provider, _empty_context()) == []
