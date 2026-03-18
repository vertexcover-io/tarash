"""TRH1xx: Provider method rules."""

from __future__ import annotations

from tarash_linter.models import ProviderInfo, Violation
from tarash_linter.rules import Rule, RuleContext, register_rule


class _RequiredMethodRule(Rule):
    """Base for rules that check a single required method exists."""

    code: str = ""
    required_method: str = ""
    video_only: bool = False

    def check(self, provider: ProviderInfo, context: RuleContext) -> list[Violation]:
        if self.video_only and not provider.is_video_provider:
            return []
        if self.required_method not in provider.methods:
            return [
                Violation(
                    code=self.code,
                    file=provider.file,
                    line=provider.class_line,
                    col=0,
                    message=f"Provider '{provider.name}' missing '{self.required_method}' method",
                )
            ]
        return []


class TRH101(_RequiredMethodRule):
    """Provider must implement _get_client."""

    code = "TRH101"
    required_method = "_get_client"
    video_only = False


class TRH102(_RequiredMethodRule):
    """Provider must implement _handle_error."""

    code = "TRH102"
    required_method = "_handle_error"
    video_only = False


class TRH103(Rule):
    """Provider must have at least one public generation method pair (sync + async)."""

    code = "TRH103"

    def check(self, provider: ProviderInfo, context: RuleContext) -> list[Violation]:
        if provider.generation_method_pairs:
            return []
        return [
            Violation(
                code=self.code,
                file=provider.file,
                line=provider.class_line,
                col=0,
                message=(
                    f"Provider '{provider.name}' has no public generation method pair. "
                    f"Expected sync + async pair (e.g., generate_video + generate_video_async)"
                ),
            )
        ]


class _ConvertMethodRule(Rule):
    """Check for _convert*request or _convert*response patterns."""

    code: str = ""
    prefix: str = "_convert"
    suffix: str = ""
    video_only: bool = True

    def check(self, provider: ProviderInfo, context: RuleContext) -> list[Violation]:
        if self.video_only and not provider.is_video_provider:
            return []
        if any(
            m.startswith(self.prefix) and m.endswith(self.suffix)
            for m in provider.methods
        ):
            return []
        return [
            Violation(
                code=self.code,
                file=provider.file,
                line=provider.class_line,
                col=0,
                message=(
                    f"Provider '{provider.name}' missing method matching "
                    f"'{self.prefix}*{self.suffix}'"
                ),
            )
        ]


class TRH104(_ConvertMethodRule):
    """Video provider must implement a convert request method."""

    code = "TRH104"
    suffix = "request"


class TRH105(_ConvertMethodRule):
    """Video provider must implement a convert response method."""

    code = "TRH105"
    suffix = "response"


class TRH106(_RequiredMethodRule):
    """Video provider must implement _validate_params."""

    code = "TRH106"
    required_method = "_validate_params"
    video_only = True


# Register all rules
register_rule(TRH101())
register_rule(TRH102())
register_rule(TRH103())
register_rule(TRH104())
register_rule(TRH105())
register_rule(TRH106())
