"""Tests for the lint runner — context building and orchestration."""

from pathlib import Path

from tarash_linter.runner import (
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


def test_parse_registry_mapping_syntax_error():
    """Syntax errors in registry source return empty mapping."""
    mapping = parse_registry_mapping("def broken(\n")
    assert mapping == {}


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


def test_scan_test_functions_skips_syntax_error(tmp_path: Path):
    """scan_test_functions skips test files with syntax errors without crashing."""
    test_dir = tmp_path / "tests" / "unit" / "video" / "providers"
    test_dir.mkdir(parents=True)
    (test_dir / "test_broken.py").write_text("def test_foo(\n")  # invalid syntax

    files = {"broken": [str(test_dir / "test_broken.py")]}
    functions = scan_test_functions(files)
    # Broken file is skipped; no functions collected means key absent
    assert "broken" not in functions


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

    config = LintConfig(exclude_providers=["fakevideo", "fakeaudio"])
    violations = run_lint(tmp_gateway, config)
    assert violations == []


def test_run_lint_respects_noqa(tmp_gateway: Path):
    """Violations on lines with # noqa are suppressed."""
    # Overwrite the fakevideo provider to be missing _get_client but with # noqa
    providers_dir = (
        tmp_gateway
        / "packages"
        / "tarash-gateway"
        / "src"
        / "tarash"
        / "tarash_gateway"
        / "providers"
    )
    (providers_dir / "fakevideo.py").write_text(
        '''"""Fake video provider."""

class FakevideoProviderHandler:  # noqa: TRH101, TRH102
    def _convert_request(self, config, request): ...
    def _convert_response(self, config, request, request_id, response): ...
    async def generate_video_async(self, config, request, on_progress=None): ...
    def generate_video(self, config, request, on_progress=None): ...
'''
    )

    # Add minimal supporting files
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

    gw_root = tmp_gateway / "packages" / "tarash-gateway"
    unit_dir = gw_root / "tests" / "unit" / "video" / "providers"
    unit_dir.mkdir(parents=True)
    (unit_dir / "test_fakevideo.py").write_text(
        "def test_convert_request(): pass\ndef test_handle_error(): pass\n"
    )
    audio_unit = gw_root / "tests" / "unit" / "audio" / "providers"
    audio_unit.mkdir(parents=True)
    (audio_unit / "test_fakeaudio.py").write_text("def test_handle_error(): pass\n")
    e2e_dir = gw_root / "tests" / "e2e"
    e2e_dir.mkdir(parents=True)
    (e2e_dir / "test_fakevideo.py").write_text("def test_e2e(): pass\n")
    (e2e_dir / "test_fakeaudio.py").write_text("def test_e2e(): pass\n")

    config = LintConfig()
    violations = run_lint(tmp_gateway, config)

    # TRH101 and TRH102 should be suppressed by # noqa on the class line
    codes = {v.code for v in violations if v.file.endswith("fakevideo.py")}
    assert "TRH101" not in codes, "TRH101 should be suppressed by # noqa"
    assert "TRH102" not in codes, "TRH102 should be suppressed by # noqa"


def test_run_lint_provider_filter(tmp_gateway: Path):
    """--provider filter restricts violations to only the specified provider."""
    config = LintConfig()
    violations = run_lint(tmp_gateway, config, provider_filter="fakevideo")

    # All violations must reference fakevideo, not fakeaudio
    assert all("fakevideo" in v.file for v in violations), (
        f"All violations must be from fakevideo, got: {[v.file for v in violations]}"
    )
    assert not any("fakeaudio" in v.file for v in violations), (
        "fakeaudio violations should not appear when filtering for fakevideo"
    )


def test_scan_test_files_overlapping_provider_names(tmp_path: Path):
    """test_fal_ai.py must not be claimed by 'fal' when 'fal_ai' is also a provider."""
    test_dir = tmp_path / "tests" / "unit" / "video" / "providers"
    test_dir.mkdir(parents=True)
    (test_dir / "test_fal.py").write_text("def test_foo(): pass")
    (test_dir / "test_fal_ai.py").write_text("def test_bar(): pass")

    result = scan_test_files(tmp_path / "tests" / "unit", ["fal", "fal_ai"])

    assert "fal_ai" in result
    assert any("test_fal_ai.py" in f for f in result["fal_ai"]), (
        "test_fal_ai.py should be claimed by fal_ai"
    )
    # test_fal_ai.py must NOT appear under fal
    fal_files = result.get("fal", [])
    assert not any("test_fal_ai.py" in f for f in fal_files), (
        "test_fal_ai.py must not be claimed by fal"
    )
