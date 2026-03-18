"""Core data models for tarash-lint."""

from __future__ import annotations

from dataclasses import dataclass, field

# All valid public generation method base names.
# Each must have a sync + async pair (e.g., generate_video + generate_video_async).
GENERATION_METHOD_BASES: list[str] = [
    "generate_video",
    "generate_image",
    "text_to_speech",
    "speech_to_text",
    "generate_tts",
    "generate_sts",
    "generate_compound",
]


@dataclass(frozen=True)
class ProviderInfo:
    """Information about a discovered provider handler class."""

    name: str
    file: str
    class_name: str
    class_line: int
    methods: frozenset[str] = field(default_factory=frozenset)

    @property
    def is_video_provider(self) -> bool:
        """True if provider has generate_video or generate_video_async."""
        return bool({"generate_video", "generate_video_async"} & self.methods)

    @property
    def generation_method_pairs(self) -> list[tuple[str, str]]:
        """Return matched (sync, async) generation method pairs found on this provider."""
        pairs = []
        for base in GENERATION_METHOD_BASES:
            sync_name = base
            async_name = f"{base}_async"
            if sync_name in self.methods and async_name in self.methods:
                pairs.append((sync_name, async_name))
        return pairs


@dataclass(frozen=True)
class Violation:
    """A single lint violation."""

    code: str
    file: str
    line: int
    col: int
    message: str
    severity: str = "error"

    def format_text(self) -> str:
        """Format as standard linter output: file:line:col: CODE message."""
        return f"{self.file}:{self.line}:{self.col}: {self.code} {self.message}"

    def to_dict(self) -> dict[str, str | int]:
        """Serialize to dict for JSON output."""
        return {
            "file": self.file,
            "line": self.line,
            "col": self.col,
            "code": self.code,
            "message": self.message,
            "severity": self.severity,
        }


@dataclass(frozen=True)
class LintConfig:
    """Configuration for a lint run."""

    select: list[str] = field(default_factory=list)
    ignore: list[str] = field(default_factory=list)
    exclude_providers: list[str] = field(default_factory=list)

    def is_rule_selected(self, code: str) -> bool:
        """Check if a rule code is selected and not ignored."""
        # Ignore always takes precedence
        for pattern in self.ignore:
            if code == pattern or code.startswith(pattern):
                return False
        # Empty select means all rules
        if not self.select:
            return True
        # Check if any select pattern matches
        return any(code == pat or code.startswith(pat) for pat in self.select)
