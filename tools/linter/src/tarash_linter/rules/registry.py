"""TRH2xx: Registry rules."""

from __future__ import annotations

from tarash_linter.models import ProviderInfo, Violation
from tarash_linter.rules import Rule, RuleContext, register_rule


class TRH201(Rule):
    """Provider handler must be registered in registry.py."""

    code = "TRH201"

    def check(self, provider: ProviderInfo, context: RuleContext) -> list[Violation]:
        # Check if any registry entry maps to this handler class
        for _prov_name, handler_class in context.registry_mapping.items():
            if handler_class == provider.class_name:
                return []
        return [
            Violation(
                code=self.code,
                file=provider.file,
                line=provider.class_line,
                col=0,
                message=(
                    f"Provider '{provider.name}' ({provider.class_name}) "
                    f"not registered in registry.py"
                ),
            )
        ]


register_rule(TRH201())
