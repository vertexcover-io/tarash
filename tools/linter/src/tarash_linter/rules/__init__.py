"""Rule base class and rule registry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tarash_linter.models import ProviderInfo, Violation


class Rule(ABC):
    """Base class for all lint rules."""

    code: str = ""
    message_template: str = ""

    @abstractmethod
    def check(self, provider: ProviderInfo, context: RuleContext) -> list[Violation]:
        """Run this rule against a provider. Return violations found."""
        ...


class RuleContext:
    """Shared context passed to all rules during a lint run.

    Pre-computed data that multiple rules need (registry mappings,
    test file lists) so each rule doesn't re-parse files.
    """

    def __init__(
        self,
        project_root: str,
        registry_mapping: dict[str, str],
        unit_test_files: dict[str, list[str]],
        unit_test_functions: dict[str, set[str]],
        e2e_test_files: dict[str, list[str]],
    ) -> None:
        self.project_root = project_root
        # provider_name -> handler_class_name from registry.py
        self.registry_mapping = registry_mapping
        # provider_name -> list of unit test file paths
        self.unit_test_files = unit_test_files
        # provider_name -> set of test function names across all unit test files
        self.unit_test_functions = unit_test_functions
        # provider_name -> list of e2e test file paths
        self.e2e_test_files = e2e_test_files


# All registered rules. Import rule modules to populate.
RULES: list[Rule] = []


def register_rule(rule: Rule) -> Rule:
    """Register a rule instance."""
    RULES.append(rule)
    return rule


def load_all_rules() -> None:
    """Auto-discover and import all rule modules in the rules package.

    Each module's top-level register_rule() calls populate the RULES list.
    Idempotent — safe to call multiple times (modules only import once).
    """
    import importlib
    import pkgutil

    package_path = Path(__file__).parent
    for module_info in pkgutil.iter_modules([str(package_path)]):
        importlib.import_module(f"tarash_linter.rules.{module_info.name}")
