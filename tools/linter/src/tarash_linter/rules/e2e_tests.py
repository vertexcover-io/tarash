"""TRH5xx: E2E test rules."""

from __future__ import annotations

from tarash_linter.models import ProviderInfo, Violation
from tarash_linter.rules import Rule, RuleContext, register_rule


class TRH401(Rule):
    """Provider must have at least one e2e test file."""

    code = "TRH401"

    def check(self, provider: ProviderInfo, context: RuleContext) -> list[Violation]:
        if context.e2e_test_files.get(provider.name):
            return []
        return [
            Violation(
                code=self.code,
                file=provider.file,
                line=provider.class_line,
                col=0,
                message=(
                    f"No e2e test file found matching "
                    f"tests/e2e/test_{provider.name}*.py"
                ),
            )
        ]


register_rule(TRH401())
