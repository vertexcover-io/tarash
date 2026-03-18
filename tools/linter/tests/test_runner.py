"""Tests for the lint runner — context building and orchestration."""

from pathlib import Path

from tarash_linter.runner import (
    parse_pricing_providers,
    parse_registry_mapping,
    run_lint,
    scan_test_files,
    scan_test_functions,
)
from tarash_linter.models import LintConfig


# --- AST parsing helpers ---


def test_parse_registry_mapping_extracts_if_elif():
    """Parse provider == 'name' -> HandlerClass() from if/elif chain."""
    source = """
def get_handler(config):
    provider = config.provider
    if provider == "fal":
        _HANDLER_INSTANCES[provider] = cast(ProviderHandler, FalProviderHandler())
    elif provider == "openai":
        _HANDLER_INSTANCES[provider] = cast(ProviderHandler, OpenAIProviderHandler())
"""
    mapping = parse_registry_mapping(source)
    assert mapping == {"fal": "FalProviderHandler", "openai": "OpenAIProviderHandler"}


def test_parse_registry_mapping_empty():
    """Returns empty dict for source with no if/elif pattern."""
    mapping = parse_registry_mapping("x = 1")
    assert mapping == {}


def test_parse_pricing_providers_extracts_tuple_keys():
    """Extract provider names from PRICING_TABLE (provider, model) tuples."""
    source = """
PRICING_TABLE = {
    ("fal", "fal-ai/veo3"): PricingEntry(usd_per_unit=Decimal("0.40"), unit="seconds"),
    ("fal", "fal-ai/flux/dev"): PricingEntry(usd_per_unit=Decimal("0.025"), unit="megapixels"),
    ("openai", "sora"): PricingEntry(usd_per_unit=Decimal("0.10"), unit="seconds"),
}
"""
    providers = parse_pricing_providers(source)
    assert providers == {"fal", "openai"}


def test_parse_pricing_providers_empty():
    """Returns empty set for source with no PRICING_TABLE."""
    providers = parse_pricing_providers("x = 1")
    assert providers == set()


def test_scan_test_files(tmp_path: Path):
    """Finds test files matching test_{provider}*.py pattern."""
    test_dir = tmp_path / "tests" / "unit" / "video" / "providers"
    test_dir.mkdir(parents=True)
    (test_dir / "test_fal.py").write_text("def test_foo(): pass")
    (test_dir / "test_fal_cost.py").write_text("def test_bar(): pass")
    (test_dir / "test_openai.py").write_text("def test_baz(): pass")
    (test_dir / "conftest.py").write_text("")  # not a test file

    result = scan_test_files(tmp_path / "tests" / "unit", ["fal", "openai", "google"])
    assert "fal" in result
    assert len(result["fal"]) == 2
    assert "openai" in result
    assert len(result["openai"]) == 1
    assert "google" not in result


def test_scan_test_functions(tmp_path: Path):
    """Extracts test function names from test files."""
    test_dir = tmp_path / "tests" / "unit" / "video" / "providers"
    test_dir.mkdir(parents=True)
    (test_dir / "test_fal.py").write_text(
        "def test_convert_request_basic(): pass\ndef test_handle_error_timeout(): pass\n"
    )

    files = {"fal": [str(test_dir / "test_fal.py")]}
    functions = scan_test_functions(files)
    assert "test_convert_request_basic" in functions["fal"]
    assert "test_handle_error_timeout" in functions["fal"]


# --- Integration ---


def test_run_lint_clean_codebase(tmp_gateway: Path):
    """A properly structured codebase produces zero violations."""
    # Add registry.py
    gw_src = (
        tmp_gateway
        / "packages"
        / "tarash-gateway"
        / "src"
        / "tarash"
        / "tarash_gateway"
    )
    (gw_src / "registry.py").write_text("""
def get_handler(config):
    provider = config.provider
    if provider == "fakevideo":
        _HANDLER_INSTANCES[provider] = cast(ProviderHandler, FakevideoProviderHandler())
    elif provider == "fakeaudio":
        _HANDLER_INSTANCES[provider] = cast(ProviderHandler, FakeaudioProviderHandler())
""")

    # Add pricing.py
    (gw_src / "pricing.py").write_text("""
PRICING_TABLE = {
    ("fakevideo", "fakevideo/model1"): "entry",
    ("fakeaudio", "fakeaudio/model1"): "entry",
}
""")

    # Add unit test files
    gw_root = tmp_gateway / "packages" / "tarash-gateway"
    unit_dir = gw_root / "tests" / "unit" / "video" / "providers"
    unit_dir.mkdir(parents=True)
    (unit_dir / "test_fakevideo.py").write_text(
        "def test_convert_request(): pass\ndef test_handle_error(): pass\n"
    )
    audio_unit = gw_root / "tests" / "unit" / "audio" / "providers"
    audio_unit.mkdir(parents=True)
    (audio_unit / "test_fakeaudio.py").write_text(
        "def test_convert_request(): pass\ndef test_handle_error(): pass\n"
    )

    # Add e2e test files
    e2e_dir = gw_root / "tests" / "e2e"
    e2e_dir.mkdir(parents=True)
    (e2e_dir / "test_fakevideo.py").write_text("def test_e2e(): pass\n")
    (e2e_dir / "test_fakeaudio.py").write_text("def test_e2e(): pass\n")

    config = LintConfig()
    violations = run_lint(tmp_gateway, config)
    assert violations == [], (
        f"Expected 0 violations, got: {[v.format_text() for v in violations]}"
    )


def test_run_lint_excludes_provider(tmp_gateway: Path):
    """Excluded providers produce no violations even if incomplete."""
    gw_src = (
        tmp_gateway
        / "packages"
        / "tarash-gateway"
        / "src"
        / "tarash"
        / "tarash_gateway"
    )
    (gw_src / "registry.py").write_text("x = 1")
    (gw_src / "pricing.py").write_text("PRICING_TABLE = {}")

    config = LintConfig(exclude_providers=["fakevideo", "fakeaudio"])
    violations = run_lint(tmp_gateway, config)
    assert violations == []
