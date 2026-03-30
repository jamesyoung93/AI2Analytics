"""Table surveyor — discovers and profiles available data in Spark catalogs."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ai2analytics.llm import LLMClient


@dataclass
class ColumnProfile:
    """Profile of a single column."""
    name: str
    dtype: str
    null_pct: float = 0.0
    n_distinct: int = 0
    sample_values: list = field(default_factory=list)
    min_val: Any = None
    max_val: Any = None
    is_time_series: bool = False


@dataclass
class TableProfile:
    """Profile of a single table."""
    full_name: str
    row_count: int = 0
    columns: list[ColumnProfile] = field(default_factory=list)
    time_columns: list[str] = field(default_factory=list)
    id_columns: list[str] = field(default_factory=list)
    has_time_series: bool = False
    grain: str = ""  # e.g. "NPI x WEEK", "NPI x MONTH"
    completeness_notes: str = ""


@dataclass
class DataSurvey:
    """Full survey of available data."""
    tables: list[TableProfile] = field(default_factory=list)
    catalogs_scanned: list[str] = field(default_factory=list)
    schemas_scanned: list[str] = field(default_factory=list)
    summary: str = ""


def survey_tables(
    spark: Any,
    schemas: list[str],
    catalog: str = "hive_metastore",
    sample_rows: int = 100,
) -> DataSurvey:
    """Survey all tables in the given schemas, profiling columns and structure.

    Args:
        spark: PySpark SparkSession.
        schemas: List of schema names to scan.
        catalog: Catalog name (default hive_metastore).
        sample_rows: Number of rows to sample for profiling.
    """
    survey = DataSurvey(
        catalogs_scanned=[catalog],
        schemas_scanned=schemas,
    )

    for schema in schemas:
        try:
            tables_df = spark.sql(f"SHOW TABLES IN {catalog}.{schema}").toPandas()
        except Exception as e:
            print(f"  WARNING: Could not list tables in {catalog}.{schema}: {e}")
            continue

        table_col = "tableName" if "tableName" in tables_df.columns else tables_df.columns[-1]

        for _, row in tables_df.iterrows():
            table_name = row[table_col]
            full_name = f"{catalog}.{schema}.{table_name}"
            print(f"  Profiling {full_name}...", end=" ", flush=True)

            try:
                profile = _profile_table(spark, full_name, sample_rows)
                survey.tables.append(profile)
                print(f"{profile.row_count:,} rows, {len(profile.columns)} cols")
            except Exception as e:
                print(f"SKIP ({e})")

    survey.summary = _build_survey_summary(survey)
    return survey


def _profile_table(spark: Any, full_name: str, sample_rows: int) -> TableProfile:
    """Profile a single Spark table."""
    # Get row count
    count_df = spark.sql(f"SELECT COUNT(*) as cnt FROM {full_name}").toPandas()
    row_count = int(count_df["cnt"].iloc[0])

    # Get schema
    desc_df = spark.sql(f"DESCRIBE {full_name}").toPandas()
    col_names = desc_df["col_name"].tolist()
    col_types = desc_df["data_type"].tolist()

    # Sample data
    sample = spark.sql(
        f"SELECT * FROM {full_name} LIMIT {sample_rows}"
    ).toPandas()

    columns = []
    time_columns = []
    id_columns = []

    for col_name, col_type in zip(col_names, col_types):
        if col_name.startswith("#") or col_name.strip() == "":
            continue

        cp = ColumnProfile(name=col_name, dtype=col_type)

        if col_name in sample.columns:
            col_data = sample[col_name]
            cp.null_pct = round(col_data.isna().mean(), 4)
            cp.n_distinct = int(col_data.nunique())
            cp.sample_values = col_data.dropna().head(5).tolist()

            if col_data.dropna().shape[0] > 0:
                try:
                    cp.min_val = col_data.min()
                    cp.max_val = col_data.max()
                except (TypeError, ValueError):
                    pass

        # Detect time columns
        if any(kw in col_name.upper() for kw in ["DATE", "WEEK", "MONTH", "YEAR", "TIME"]):
            cp.is_time_series = True
            time_columns.append(col_name)
        elif col_type in ("date", "timestamp"):
            cp.is_time_series = True
            time_columns.append(col_name)

        # Detect ID columns
        if any(kw in col_name.upper() for kw in ["NPI", "ID", "KEY", "CODE"]):
            id_columns.append(col_name)

        columns.append(cp)

    profile = TableProfile(
        full_name=full_name,
        row_count=row_count,
        columns=columns,
        time_columns=time_columns,
        id_columns=id_columns,
        has_time_series=len(time_columns) > 0,
    )

    # Determine grain
    if time_columns and id_columns:
        profile.grain = f"{id_columns[0]} x {time_columns[0]}"

    return profile


def _build_survey_summary(survey: DataSurvey) -> str:
    """Build a human-readable summary of the survey."""
    lines = [f"Data Survey: {len(survey.tables)} tables found\n"]
    for t in survey.tables:
        ts_label = " [TIME SERIES]" if t.has_time_series else ""
        lines.append(f"  {t.full_name}: {t.row_count:,} rows, {len(t.columns)} cols{ts_label}")
        if t.grain:
            lines.append(f"    Grain: {t.grain}")
        if t.time_columns:
            lines.append(f"    Time cols: {', '.join(t.time_columns)}")
        if t.id_columns:
            lines.append(f"    ID cols: {', '.join(t.id_columns)}")
    return "\n".join(lines)


def profile_for_llm(survey: DataSurvey) -> str:
    """Format survey into a compact string suitable for LLM context."""
    lines = []
    for t in survey.tables:
        lines.append(f"\nTABLE: {t.full_name} ({t.row_count:,} rows)")
        for c in t.columns:
            ts_tag = " [TIME]" if c.is_time_series else ""
            null_tag = f" [{c.null_pct:.0%} null]" if c.null_pct > 0.05 else ""
            distinct = f" ({c.n_distinct} distinct)" if c.n_distinct < 50 else ""
            samples = ""
            if c.sample_values:
                sv = [str(v)[:30] for v in c.sample_values[:3]]
                samples = f" e.g. {', '.join(sv)}"
            lines.append(f"  {c.name} ({c.dtype}){ts_tag}{null_tag}{distinct}{samples}")
    return "\n".join(lines)
