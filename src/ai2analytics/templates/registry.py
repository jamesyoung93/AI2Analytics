"""Template registry — discovers and loads available pipeline templates."""

from __future__ import annotations

from ai2analytics.templates.base import BaseTemplate

_REGISTRY: dict[str, type[BaseTemplate]] = {}


def register(template_cls: type[BaseTemplate]) -> type[BaseTemplate]:
    """Register a template class. Use as a decorator."""
    _REGISTRY[template_cls.name] = template_cls
    return template_cls


def get_template(name: str) -> type[BaseTemplate]:
    """Get a registered template by name."""
    if name not in _REGISTRY:
        available = ", ".join(sorted(_REGISTRY.keys())) or "(none)"
        raise KeyError(f"Unknown template '{name}'. Available: {available}")
    return _REGISTRY[name]


def list_templates() -> dict[str, str]:
    """Return {name: description} for all registered templates."""
    return {name: cls.description for name, cls in sorted(_REGISTRY.items())}


def find_template(prompt: str) -> list[tuple[str, type[BaseTemplate]]]:
    """Simple keyword match to suggest templates for a user prompt."""
    words = set(prompt.lower().split())
    scored = []
    for name, cls in _REGISTRY.items():
        keywords = set(cls.description.lower().split()) | set(name.lower().split("_"))
        overlap = len(words & keywords)
        if overlap > 0:
            scored.append((overlap, name, cls))
    scored.sort(key=lambda x: -x[0])
    return [(name, cls) for _, name, cls in scored]
