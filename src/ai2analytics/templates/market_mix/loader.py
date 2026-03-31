"""Stage 1: Data loading — loads the time series table for market mix modeling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ai2analytics.templates.market_mix.config import MarketMixConfig


@dataclass
class MarketMixData:
    """Container for all data loaded by the data loading stage."""
    time_series: pd.DataFrame = field(default_factory=pd.DataFrame)
    date_col: str = ""
    response_col: str = ""
    media_cols: list[str] = field(default_factory=list)
    control_cols: list[str] = field(default_factory=list)
    n_periods: int = 0


def load_data(
    cfg: MarketMixConfig,
    spark: Any = None,
    dataframes: dict[str, pd.DataFrame] | None = None,
) -> MarketMixData:
    """Load the time series data per config. Returns a MarketMixData container.

    Args:
        cfg: Pipeline configuration.
        spark: PySpark SparkSession (required for Spark table reads).
        dataframes: Optional dict of in-memory pandas DataFrames to use
                    instead of reading from files/tables. Keys:
                    "time_series".
                    If a key is present, the corresponding table config
                    is ignored and the DataFrame is used directly.
    """
    dfs = dataframes or {}
    data = MarketMixData()
    print("=" * 70)
    print("STAGE 1: Loading data")
    print("=" * 70)

    # Load time series table
    if "time_series" in dfs:
        data.time_series = dfs["time_series"].copy()
        print(f"  Time series:   {len(data.time_series):,} rows (in-memory)")
    elif cfg.time_series_source_type == "csv":
        data.time_series = pd.read_csv(cfg.time_series_table)
        print(f"  Time series:   {len(data.time_series):,} rows (CSV)")
    else:
        if spark is None:
            raise RuntimeError("spark session is required for loading tables")
        data.time_series = spark.table(cfg.time_series_table).toPandas()
        print(f"  Time series:   {len(data.time_series):,} rows")

    # Validate required columns exist
    ts = data.time_series
    missing = []
    if cfg.col_date not in ts.columns:
        missing.append(f"date column '{cfg.col_date}'")
    if cfg.col_response not in ts.columns:
        missing.append(f"response column '{cfg.col_response}'")
    for mc in cfg.media_columns:
        if mc not in ts.columns:
            missing.append(f"media column '{mc}'")
    for cc in cfg.control_columns:
        if cc not in ts.columns:
            missing.append(f"control column '{cc}'")
    if missing:
        raise ValueError(
            f"Missing columns in time_series table: {', '.join(missing)}"
        )

    # Parse and sort by date
    ts[cfg.col_date] = pd.to_datetime(ts[cfg.col_date])
    ts = ts.sort_values(cfg.col_date).reset_index(drop=True)
    data.time_series = ts

    # Store column references
    data.date_col = cfg.col_date
    data.response_col = cfg.col_response
    data.media_cols = list(cfg.media_columns)
    data.control_cols = list(cfg.control_columns)
    data.n_periods = len(ts)

    print(f"  Date range:    {ts[cfg.col_date].min()} to {ts[cfg.col_date].max()}")
    print(f"  Periods:       {data.n_periods}")
    print(f"  Media cols:    {data.media_cols}")
    print(f"  Control cols:  {data.control_cols}")
    print("  Done.\n")
    return data
