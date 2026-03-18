"""Tests for TRH5xx e2e test rules."""

from tarash_linter.models import ProviderInfo
from tarash_linter.rules import RuleContext
from tarash_linter.rules.e2e_tests import TRH501


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


def _context_with_e2e(files: dict[str, list[str]]) -> RuleContext:
    return RuleContext(
        project_root=".",
        registry_mapping={},
        pricing_providers=set(),
        unit_test_files={},
        unit_test_functions={},
        e2e_test_files=files,
    )


def test_trh501_passes_when_e2e_test_exists():
    rule = TRH501()
    provider = _make_provider("fal")
    ctx = _context_with_e2e({"fal": ["tests/e2e/test_fal.py"]})
    assert rule.check(provider, ctx) == []


def test_trh501_fails_when_no_e2e_test():
    rule = TRH501()
    provider = _make_provider("newprovider")
    ctx = _context_with_e2e({})
    violations = rule.check(provider, ctx)
    assert len(violations) == 1
    assert violations[0].code == "TRH501"
