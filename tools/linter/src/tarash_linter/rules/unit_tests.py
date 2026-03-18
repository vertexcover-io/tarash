"""TRH4xx: Unit test rules."""

from __future__ import annotations

from tarash_linter.models import ProviderInfo, Violation
from tarash_linter.rules import Rule, RuleContext, register_rule


class TRH401(Rule):
    """Provider must have at least one unit test file."""

    code = "TRH401"

    def check(self, provider: ProviderInfo, context: RuleContext) -> list[Violation]:
        if context.unit_test_files.get(provider.name):
            return []
        return [
            Violation(
                code=self.code,
                file=provider.file,
                line=provider.class_line,
                col=0,
                message=(
                    f"No unit test file found matching "
                    f"tests/unit/**/test_{provider.name}*.py"
                ),
            )
        ]


class TRH402(Rule):
    """Unit tests must include a test for _convert_request (video providers only)."""

    code = "TRH402"

    def check(self, provider: ProviderInfo, context: RuleContext) -> list[Violation]:
        # Only applies to video providers (who must have _convert_request per TRH104)
        if not provider.is_video_provider:
            return []
        # Skip if no test files at all — TRH401 covers that
        if not context.unit_test_files.get(provider.name):
            return []
        functions = context.unit_test_functions.get(provider.name, set())
        # Accept test_convert_request, test_convert_video_request, test_convert_image_request, etc.
        if any("convert" in fn and "request" in fn for fn in functions):
            return []
        return [
            Violation(
                code=self.code,
                file=provider.file,
                line=provider.class_line,
                col=0,
                message=(
                    f"No test function matching 'test_*convert*request*' found "
                    f"in unit tests for provider '{provider.name}'"
                ),
            )
        ]


class TRH403(Rule):
    """Unit tests must include a test for _handle_error."""

    code = "TRH403"

    def check(self, provider: ProviderInfo, context: RuleContext) -> list[Violation]:
        if not context.unit_test_files.get(provider.name):
            return []
        functions = context.unit_test_functions.get(provider.name, set())
        # Accept test_handle_error_*, test_*raises*, or test_*error* patterns
        if any(
            "handle_error" in fn or "raises" in fn or "error" in fn for fn in functions
        ):
            return []
        return [
            Violation(
                code=self.code,
                file=provider.file,
                line=provider.class_line,
                col=0,
                message=(
                    f"No test function matching 'test_*handle_error*' or "
                    f"'test_*raises*' found in unit tests for provider '{provider.name}'"
                ),
            )
        ]


register_rule(TRH401())
register_rule(TRH402())
register_rule(TRH403())
