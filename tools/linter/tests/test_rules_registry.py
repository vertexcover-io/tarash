"""Tests for TRH2xx registry rules."""

from tarash_linter.models import ProviderInfo
from tarash_linter.rules import RuleContext
from tarash_linter.rules.registry import TRH201


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


def _context_with_registry(mapping: dict[str, str]) -> RuleContext:
    return RuleContext(
        project_root=".",
        registry_mapping=mapping,
        pricing_providers=set(),
        unit_test_files={},
        unit_test_functions={},
        e2e_test_files={},
    )


def test_trh201_passes_when_registered():
    rule = TRH201()
    provider = _make_provider("fal")
    ctx = _context_with_registry({"fal": "FalProviderHandler"})
    assert rule.check(provider, ctx) == []


def test_trh201_fails_when_not_registered():
    rule = TRH201()
    provider = _make_provider("newprovider")
    ctx = _context_with_registry({"fal": "FalProviderHandler"})
    violations = rule.check(provider, ctx)
    assert len(violations) == 1
    assert violations[0].code == "TRH201"
    assert "newprovider" in violations[0].message
    assert "registry.py" in violations[0].message
