"""Tests for TRH3xx pricing rules."""

from tarash_linter.models import ProviderInfo
from tarash_linter.rules import RuleContext
from tarash_linter.rules.pricing import TRH301


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


def _context_with_pricing(providers: set[str]) -> RuleContext:
    return RuleContext(
        project_root=".",
        registry_mapping={},
        pricing_providers=providers,
        unit_test_files={},
        unit_test_functions={},
        e2e_test_files={},
    )


def test_trh301_passes_when_pricing_exists():
    rule = TRH301()
    provider = _make_provider("fal")
    ctx = _context_with_pricing({"fal"})
    assert rule.check(provider, ctx) == []


def test_trh301_fails_when_no_pricing():
    rule = TRH301()
    provider = _make_provider("newprovider")
    ctx = _context_with_pricing({"fal", "openai"})
    violations = rule.check(provider, ctx)
    assert len(violations) == 1
    assert violations[0].code == "TRH301"
    assert "PRICING_TABLE" in violations[0].message
