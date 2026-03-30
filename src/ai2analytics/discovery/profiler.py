"""Data profiler — deep analysis of specific tables for template matching."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ai2analytics.discovery.surveyor import TableProfile


@dataclass
class TimeSeriesProfile:
    """Detailed time series characteristics of a table."""
    time_column: str = ""
    frequency: str = ""  # "weekly", "monthly", "daily", "irregular"
    min_date: str = ""
    max_date: str = ""
    n_periods: int = 0
    has_gaps: bool = False
    gap_pct: float = 0.0
    entities_per_period: float = 0.0  # avg observations per time period
    categories_per_entity: float = 0.0


@dataclass
class DeepProfile:
    """Deep profile combining table profile with time series and structure analysis."""
    table: TableProfile
    time_series: TimeSeriesProfile | None = None
    is_panel_data: bool = False  # entity x time
    entity_column: str = ""
    n_entities: int = 0
    categorical_columns: list[str] = field(default_factory=list)
    numeric_columns: list[str] = field(default_factory=list)
    flag_columns: list[str] = field(default_factory=list)
    stagnant_columns: list[str] = field(default_factory=list)  # no time variation


def deep_profile(
    spark: Any,
    full_table_name: str,
    entity_col: str | None = None,
    time_col: str | None = None,
    sample_size: int = 10000,
) -> DeepProfile:
    """Run a deep profile on a specific table.

    Analyzes time series characteristics, completeness, variable types,
    and whether variables change over time or are static per entity.

    Args:
        spark: PySpark SparkSession.
        full_table_name: Fully qualified table name.
        entity_col: Column identifying entities (e.g. NPI). Auto-detected if None.
        time_col: Time column. Auto-detected if None.
        sample_size: Rows to sample for analysis.
    """
    # Get a sample
    sample = spark.sql(
        f"SELECT * FROM {full_table_name} LIMIT {sample_size}"
    ).toPandas()

    row_count = spark.sql(
        f"SELECT COUNT(*) as cnt FROM {full_table_name}"
    ).toPandas()["cnt"].iloc[0]

    # Build basic table profile
    from ai2analytics.discovery.surveyor import ColumnProfile
    columns = []
    for col in sample.columns:
        cp = ColumnProfile(
            name=col,
            dtype=str(sample[col].dtype),
            null_pct=round(sample[col].isna().mean(), 4),
            n_distinct=int(sample[col].nunique()),
            sample_values=sample[col].dropna().head(3).tolist(),
        )
        columns.append(cp)

    table_profile = TableProfile(
        full_name=full_table_name,
        row_count=int(row_count),
        columns=columns,
    )

    result = DeepProfile(table=table_profile)

    # Auto-detect entity and time columns
    if entity_col is None:
        entity_col = _detect_entity_col(sample)
    if time_col is None:
        time_col = _detect_time_col(sample)

    result.entity_column = entity_col or ""
    result.n_entities = int(sample[entity_col].nunique()) if entity_col else 0

    # Classify columns
    for col in sample.columns:
        if col in (entity_col, time_col):
            continue
        if sample[col].dtype == "object":
            if sample[col].nunique() <= 5:
                result.flag_columns.append(col)
            else:
                result.categorical_columns.append(col)
        elif pd.api.types.is_numeric_dtype(sample[col]):
            result.numeric_columns.append(col)

    # Time series analysis
    if time_col and entity_col:
        result.is_panel_data = True
        result.time_series = _analyze_time_series(sample, entity_col, time_col)

        # Check which numeric columns are stagnant (don't vary within entity)
        for col in result.numeric_columns:
            within_var = sample.groupby(entity_col)[col].std().mean()
            if within_var is not None and within_var < 0.001:
                result.stagnant_columns.append(col)

    return result


def _detect_entity_col(df: pd.DataFrame) -> str | None:
    """Auto-detect the entity/ID column."""
    candidates = []
    for col in df.columns:
        upper = col.upper()
        if "NPI" in upper:
            candidates.append((0, col))  # highest priority
        elif "ID" in upper or "KEY" in upper:
            candidates.append((1, col))
        elif "CODE" in upper:
            candidates.append((2, col))

    if candidates:
        candidates.sort()
        return candidates[0][1]

    # Fallback: column with high cardinality relative to row count
    for col in df.columns:
        if df[col].nunique() > len(df) * 0.1 and df[col].nunique() > 10:
            return col
    return None


def _detect_time_col(df: pd.DataFrame) -> str | None:
    """Auto-detect the time column."""
    for col in df.columns:
        upper = col.upper()
        if any(kw in upper for kw in ["WEEK_END", "WEEK_ENDING", "DATE", "PERIOD"]):
            return col
    # Check dtypes
    for col in df.columns:
        if pd.api.types.is_datetime64_any_dtype(df[col]):
            return col
        # Try parsing
        try:
            parsed = pd.to_datetime(df[col].head(10), errors="coerce")
            if parsed.notna().mean() > 0.8:
                return col
        except (TypeError, ValueError):
            pass
    return None


def _analyze_time_series(
    df: pd.DataFrame, entity_col: str, time_col: str
) -> TimeSeriesProfile:
    """Analyze time series characteristics."""
    ts = TimeSeriesProfile(time_column=time_col)

    try:
        df[time_col] = pd.to_datetime(df[time_col])
    except (TypeError, ValueError):
        return ts

    ts.min_date = str(df[time_col].min())
    ts.max_date = str(df[time_col].max())

    # Detect frequency
    sorted_dates = df[time_col].sort_values().unique()
    if len(sorted_dates) > 1:
        diffs = pd.Series(sorted_dates).diff().dropna()
        median_diff = diffs.median()
        ts.n_periods = len(sorted_dates)

        if pd.Timedelta(days=5) <= median_diff <= pd.Timedelta(days=9):
            ts.frequency = "weekly"
        elif pd.Timedelta(days=25) <= median_diff <= pd.Timedelta(days=35):
            ts.frequency = "monthly"
        elif median_diff <= pd.Timedelta(days=2):
            ts.frequency = "daily"
        else:
            ts.frequency = "irregular"

    # Completeness: check for gaps
    entities = df[entity_col].unique()
    periods = df[time_col].unique()
    expected = len(entities) * len(periods)
    actual = len(df.drop_duplicates([entity_col, time_col]))
    if expected > 0:
        ts.gap_pct = round(1 - actual / expected, 4)
        ts.has_gaps = ts.gap_pct > 0.05

    # Average observations per period
    ts.entities_per_period = df.groupby(time_col)[entity_col].nunique().mean()

    # Categories per entity per period (if there are multiple rows per entity x period)
    multi = df.groupby([entity_col, time_col]).size()
    ts.categories_per_entity = multi.mean()

    return ts


def format_deep_profile(profile: DeepProfile) -> str:
    """Format a deep profile as a readable string."""
    lines = [f"Deep Profile: {profile.table.full_name}"]
    lines.append(f"  Rows: {profile.table.row_count:,}")
    lines.append(f"  Entity column: {profile.entity_column} ({profile.n_entities:,} unique)")

    if profile.is_panel_data and profile.time_series:
        ts = profile.time_series
        lines.append(f"\n  Time Series:")
        lines.append(f"    Column: {ts.time_column}")
        lines.append(f"    Frequency: {ts.frequency}")
        lines.append(f"    Range: {ts.min_date} to {ts.max_date} ({ts.n_periods} periods)")
        lines.append(f"    Completeness: {1 - ts.gap_pct:.1%} {'(has gaps)' if ts.has_gaps else '(complete)'}")
        lines.append(f"    Avg entities/period: {ts.entities_per_period:.0f}")
        if ts.categories_per_entity > 1.1:
            lines.append(f"    Avg rows/entity/period: {ts.categories_per_entity:.1f} (multiple categories)")

    lines.append(f"\n  Numeric columns ({len(profile.numeric_columns)}): {', '.join(profile.numeric_columns[:10])}")
    lines.append(f"  Categorical columns ({len(profile.categorical_columns)}): {', '.join(profile.categorical_columns[:10])}")
    lines.append(f"  Flag columns ({len(profile.flag_columns)}): {', '.join(profile.flag_columns[:10])}")

    if profile.stagnant_columns:
        lines.append(f"  Stagnant columns (no time variation): {', '.join(profile.stagnant_columns[:10])}")

    return "\n".join(lines)
