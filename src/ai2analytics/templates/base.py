"""Base template class that all pipeline templates inherit from."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ColumnRequirement:
    """Describes a column a template needs."""
    name: str
    dtype: str  # "int", "float", "date", "string", "numeric"
    description: str
    aliases: list[str] = field(default_factory=list)
    config_field: str = ""  # maps to this config field (e.g. "col_npi")


@dataclass
class TableRequirement:
    """Describes a data table a template needs."""
    key: str  # internal key used in config (e.g. "hcp_weekly")
    description: str
    required_columns: list[ColumnRequirement] = field(default_factory=list)
    optional_columns: list[ColumnRequirement] = field(default_factory=list)
    source_type: str = "spark_table"  # "spark_table", "csv", "delta"
    config_field: str = ""  # maps to this config field (e.g. "hcp_weekly_table")


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
            cfg_hint = f" -> cfg.{table.config_field}" if table.config_field else ""
            lines.append(f"\n  [{table.key}] ({table.source_type}){cfg_hint}")
            lines.append(f"    {table.description}")
            if table.required_columns:
                lines.append("    Required columns:")
                for col in table.required_columns:
                    aliases = f" (aliases: {', '.join(col.aliases)})" if col.aliases else ""
                    cfg = f" -> cfg.{col.config_field}" if col.config_field else ""
                    lines.append(
                        f"      - {col.name} ({col.dtype}): {col.description}{aliases}{cfg}"
                    )
            if table.optional_columns:
                lines.append("    Optional columns:")
                for col in table.optional_columns:
                    cfg = f" -> cfg.{col.config_field}" if col.config_field else ""
                    lines.append(f"      - {col.name} ({col.dtype}): {col.description}{cfg}")
        return "\n".join(lines)

    @classmethod
    def get_config_summary(cls) -> str:
        """Return a summary of config fields with types and defaults.

        This gives the LLM the exact field names it needs to auto-fill.
        """
        if cls.config_class is None or not dataclasses.is_dataclass(cls.config_class):
            return ""

        lines = [f"Config class: {cls.config_class.__name__}", ""]
        for f in dataclasses.fields(cls.config_class):
            # Determine effective default
            if f.default is not dataclasses.MISSING:
                default = f.default
            elif f.default_factory is not dataclasses.MISSING:
                default = f.default_factory()
            else:
                default = "REQUIRED"

            # Flag empty-string defaults as needing user input
            if default == "":
                tag = "  <-- MUST BE SET"
            elif default == "REQUIRED":
                tag = "  <-- REQUIRED"
            else:
                tag = ""

            # Type hint
            type_str = f.type if isinstance(f.type, str) else getattr(f.type, "__name__", str(f.type))

            lines.append(f"  {f.name} ({type_str}) = {repr(default)}{tag}")

        return "\n".join(lines)

    def run(self, config: Any, spark: Any = None) -> Any:
        raise NotImplementedError("Subclasses must implement run()")
