"""Tests for TRH3xx unit test rules."""

from tarash_linter.models import ProviderInfo
from tarash_linter.rules import RuleContext
from tarash_linter.rules.unit_tests import TRH301, TRH302, TRH303


def _make_provider(name: str) -> ProviderInfo:
    return ProviderInfo(
        name=name,
        file=f"providers/{name}.py",
        class_name=f"{name.title()}ProviderHandler",
        class_line=10,
        methods=frozenset(
            {"_get_client", "_handle_error", "generate_video", "generate_video_async"}
        ),
    )


def _context_with_tests(
    files: dict[str, list[str]] | None = None,
    functions: dict[str, set[str]] | None = None,
) -> RuleContext:
    return RuleContext(
        project_root=".",
        registry_mapping={},
        unit_test_files=files or {},
        unit_test_functions=functions or {},
        e2e_test_files={},
    )


def test_trh301_passes_when_test_file_exists():
    rule = TRH301()
    provider = _make_provider("fal")
    ctx = _context_with_tests(files={"fal": ["tests/unit/video/providers/test_fal.py"]})
    assert rule.check(provider, ctx) == []


def test_trh301_fails_when_no_test_file():
    rule = TRH301()
    provider = _make_provider("newprovider")
    ctx = _context_with_tests(files={})
    violations = rule.check(provider, ctx)
    assert len(violations) == 1
    assert violations[0].code == "TRH301"


def test_trh302_passes_with_convert_request_test():
    rule = TRH302()
    provider = _make_provider("fal")
    ctx = _context_with_tests(
        files={"fal": ["test_fal.py"]},
        functions={"fal": {"test_convert_request_basic", "test_handle_error_timeout"}},
    )
    assert rule.check(provider, ctx) == []


def test_trh302_fails_without_convert_request_test():
    rule = TRH302()
    provider = _make_provider("fal")
    ctx = _context_with_tests(
        files={"fal": ["test_fal.py"]},
        functions={"fal": {"test_handle_error_timeout"}},
    )
    violations = rule.check(provider, ctx)
    assert len(violations) == 1
    assert violations[0].code == "TRH302"


def test_trh302_skips_when_no_test_file():
    """TRH302 doesn't fire if there's no test file -- TRH301 covers that."""
    rule = TRH302()
    provider = _make_provider("fal")
    ctx = _context_with_tests(files={}, functions={})
    assert rule.check(provider, ctx) == []


def test_trh302_skips_non_video_provider():
    """TRH302 doesn't apply to audio-only providers."""
    rule = TRH302()
    provider = ProviderInfo(
        name="cartesia",
        file="providers/cartesia.py",
        class_name="CartesiaProviderHandler",
        class_line=10,
        methods=frozenset(
            {"_get_client", "_handle_error", "generate_tts", "generate_tts_async"}
        ),
    )
    ctx = _context_with_tests(
        files={"cartesia": ["test_cartesia.py"]},
        functions={"cartesia": {"test_handle_error_timeout"}},
    )
    assert rule.check(provider, ctx) == []


def test_trh303_passes_with_handle_error_test():
    rule = TRH303()
    provider = _make_provider("fal")
    ctx = _context_with_tests(
        files={"fal": ["test_fal.py"]},
        functions={"fal": {"test_handle_error_maps_400"}},
    )
    assert rule.check(provider, ctx) == []


def test_trh303_fails_without_handle_error_test():
    rule = TRH303()
    provider = _make_provider("fal")
    ctx = _context_with_tests(
        files={"fal": ["test_fal.py"]},
        functions={"fal": {"test_convert_request_basic"}},
    )
    violations = rule.check(provider, ctx)
    assert len(violations) == 1
    assert violations[0].code == "TRH303"
