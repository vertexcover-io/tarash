"""TRH3xx: Pricing rules."""

from __future__ import annotations

from tarash_linter.models import ProviderInfo, Violation
from tarash_linter.rules import Rule, RuleContext, register_rule


class TRH301(Rule):
    """Provider must have at least one entry in PRICING_TABLE."""

    code = "TRH301"

    def check(self, provider: ProviderInfo, context: RuleContext) -> list[Violation]:
        if provider.name in context.pricing_providers:
            return []
        return [
            Violation(
                code=self.code,
                file=provider.file,
                line=provider.class_line,
                col=0,
                message=(
                    f"Provider '{provider.name}' has zero entries in PRICING_TABLE "
                    f"in pricing.py"
                ),
            )
        ]


register_rule(TRH301())
