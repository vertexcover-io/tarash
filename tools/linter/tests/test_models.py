"""Tests for core data models."""

from tarash_linter.models import LintConfig, ProviderInfo, Violation


def test_provider_info_is_video_provider_with_generate_video():
    """Provider with generate_video is a video provider."""
    info = ProviderInfo(
        name="fal",
        file="providers/fal.py",
        class_name="FalProviderHandler",
        class_line=100,
        methods=frozenset(
            {"_get_client", "_handle_error", "generate_video", "generate_video_async"}
        ),
    )
    assert info.is_video_provider is True


def test_provider_info_is_video_provider_without_generate_video():
    """Provider without generate_video is not a video provider."""
    info = ProviderInfo(
        name="cartesia",
        file="providers/cartesia.py",
        class_name="CartesiaProviderHandler",
        class_line=50,
        methods=frozenset(
            {"_get_client", "_handle_error", "generate_tts", "generate_tts_async"}
        ),
    )
    assert info.is_video_provider is False


def test_provider_info_generation_method_pairs():
    """generation_method_pairs returns matched sync+async pairs."""
    info = ProviderInfo(
        name="fal",
        file="providers/fal.py",
        class_name="FalProviderHandler",
        class_line=100,
        methods=frozenset(
            {
                "generate_video",
                "generate_video_async",
                "generate_image",
                "generate_image_async",
            }
        ),
    )
    pairs = info.generation_method_pairs
    assert ("generate_video", "generate_video_async") in pairs
    assert ("generate_image", "generate_image_async") in pairs


def test_provider_info_no_generation_pairs():
    """Provider with no generation methods has empty pairs."""
    info = ProviderInfo(
        name="broken",
        file="providers/broken.py",
        class_name="BrokenProviderHandler",
        class_line=1,
        methods=frozenset({"_get_client"}),
    )
    assert info.generation_method_pairs == []


def test_violation_format_text():
    """Violation formats to standard linter output."""
    v = Violation(
        code="TRH101",
        file="providers/fal.py",
        line=100,
        col=0,
        message="missing '_get_client'",
    )
    assert v.format_text() == "providers/fal.py:100:0: TRH101 missing '_get_client'"


def test_violation_format_json():
    """Violation serializes to dict."""
    v = Violation(
        code="TRH101", file="providers/fal.py", line=100, col=0, message="test"
    )
    d = v.to_dict()
    assert d["code"] == "TRH101"
    assert d["file"] == "providers/fal.py"
    assert d["line"] == 100
    assert d["severity"] == "error"


def test_lint_config_is_rule_selected_with_prefix():
    """Rule selection works with prefix matching."""
    config = LintConfig(select=["TRH1", "TRH3"], ignore=[], exclude_providers=[])
    assert config.is_rule_selected("TRH101") is True
    assert config.is_rule_selected("TRH301") is True
    assert config.is_rule_selected("TRH201") is False


def test_lint_config_ignore_takes_precedence():
    """Ignore overrides select."""
    config = LintConfig(select=["TRH1"], ignore=["TRH101"], exclude_providers=[])
    assert config.is_rule_selected("TRH101") is False
    assert config.is_rule_selected("TRH102") is True


def test_lint_config_default_selects_all():
    """Empty select means all rules selected."""
    config = LintConfig(select=[], ignore=[], exclude_providers=[])
    assert config.is_rule_selected("TRH101") is True
    assert config.is_rule_selected("TRH501") is True
