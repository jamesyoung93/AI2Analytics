"""Base template class that all pipeline templates inherit from."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ColumnRequirement:
    """Describes a column a template needs."""
    name: str
    dtype: str  # "int", "float", "date", "string", "numeric"
    description: str
    aliases: list[str] = field(default_factory=list)


@dataclass
class TableRequirement:
    """Describes a data table a template needs."""
    key: str  # internal key used in config (e.g. "hcp_weekly")
    description: str
    required_columns: list[ColumnRequirement] = field(default_factory=list)
    optional_columns: list[ColumnRequirement] = field(default_factory=list)
    source_type: str = "spark_table"  # "spark_table", "csv", "delta"


class BaseTemplate:
    """Abstract base for pipeline templates.

    Subclasses must define:
        name: str — unique identifier
        description: str — human-readable description
        required_tables: list[TableRequirement] — data inputs needed
        config_class: type — the dataclass config for this pipeline

    And implement:
        run(config, spark=None) -> results
    """

    name: str = ""
    description: str = ""
    required_tables: list[TableRequirement] = []
    config_class: type = None

    @classmethod
    def get_schema_summary(cls) -> str:
        """Return a human-readable summary of required data inputs."""
        lines = [f"Template: {cls.name}", f"  {cls.description}", "", "Required data inputs:"]
        for table in cls.required_tables:
            lines.append(f"\n  [{table.key}] ({table.source_type})")
            lines.append(f"    {table.description}")
            if table.required_columns:
                lines.append("    Required columns:")
                for col in table.required_columns:
                    aliases = f" (aliases: {', '.join(col.aliases)})" if col.aliases else ""
                    lines.append(f"      - {col.name} ({col.dtype}): {col.description}{aliases}")
            if table.optional_columns:
                lines.append("    Optional columns:")
                for col in table.optional_columns:
                    lines.append(f"      - {col.name} ({col.dtype}): {col.description}")
        return "\n".join(lines)

    def run(self, config: Any, spark: Any = None) -> Any:
        raise NotImplementedError("Subclasses must implement run()")
