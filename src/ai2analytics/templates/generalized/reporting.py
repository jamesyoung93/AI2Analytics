"""Small reporting helper for the generalized template."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GeneralizedReporter:
    """Collect and print run assumptions, warnings, and progress messages."""

    verbose: bool = True
    warn_on_defaults: bool = True
    warnings: list[str] = field(default_factory=list)

    def progress(self, message: str) -> None:
        if self.verbose:
            print(f"[generalized] {message}")

    def warn(self, message: str) -> None:
        self.warnings.append(message)
        if self.warn_on_defaults:
            print(f"[generalized warning] {message}")

    def print_summary(self) -> None:
        if not self.verbose or not self.warnings:
            return
        print("\n[generalized] Assumptions and default behavior used in this run:")
        for idx, warning in enumerate(self.warnings, start=1):
            print(f"  {idx}. {warning}")
