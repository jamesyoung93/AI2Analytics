"""Stage C: Data loading — loads all required tables and reference files."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ai2analytics.templates.detail_optimization.config import DetailOptimizationConfig
from ai2analytics.utils import clean_npi, resolve_col, require_columns, yn_binary


@dataclass
class LoadedData:
    """Container for all data loaded by the data loading stage."""
    hcp_weekly: pd.DataFrame = field(default_factory=pd.DataFrame)
    calls: pd.DataFrame = field(default_factory=pd.DataFrame)
    team_a_align: pd.DataFrame = field(default_factory=pd.DataFrame)
    team_b_align: pd.DataFrame = field(default_factory=pd.DataFrame)
    portfolio_decile: pd.DataFrame = field(default_factory=pd.DataFrame)
    priority_targets: pd.DataFrame | None = None
    hcp_reference: pd.DataFrame = field(default_factory=pd.DataFrame)


def load_data(
    cfg: DetailOptimizationConfig,
    spark: Any = None,
    dataframes: dict[str, pd.DataFrame] | None = None,
) -> LoadedData:
    """Load all data sources per config. Returns a LoadedData container.

    Args:
        cfg: Pipeline configuration.
        spark: PySpark SparkSession (required for Spark table reads).
        dataframes: Optional dict of in-memory DataFrames to use instead of
                    reading from files/tables. Keys can be any of:
                    "hcp_weekly", "calls", "team_a_align", "team_b_align",
                    "portfolio_decile", "priority_targets", "hcp_reference".
                    If a key is present, the corresponding path/table config
                    is ignored and the DataFrame is used directly.
    """
    dfs = dataframes or {}
    data = LoadedData()
    print("=" * 70)
    print("STAGE C: Loading data")
    print("=" * 70)

    # C1. HCP weekly table
    if "hcp_weekly" in dfs:
        data.hcp_weekly = dfs["hcp_weekly"].copy()
        print(f"  HCP weekly:       {len(data.hcp_weekly):,} rows (in-memory)")
    else:
        if spark is None:
            raise RuntimeError("spark session is required for loading tables")
        query = (
            f"SELECT * FROM {cfg.hcp_weekly_table} "
            f"WHERE {cfg.hcp_filter_col} LIKE '{cfg.hcp_filter_val}'"
        )
        data.hcp_weekly = spark.sql(query).toPandas()
        print(f"  HCP weekly:       {len(data.hcp_weekly):,} rows")

    # C2. Calls table
    if "calls" in dfs:
        data.calls = dfs["calls"].copy()
        print(f"  Calls:            {len(data.calls):,} rows (in-memory)")
    else:
        if spark is None:
            raise RuntimeError("spark session is required for loading tables")
        data.calls = spark.table(cfg.calls_table).toPandas()
        print(f"  Calls:            {len(data.calls):,} rows")

    # C3. Team A alignment
    if "team_a_align" in dfs:
        raw_a = dfs["team_a_align"].copy()
    else:
        raw_a = pd.read_csv(cfg.team_a_align_path)
    raw_a = raw_a.rename(columns={
        cfg.team_a_npi_col: cfg.col_npi,
        cfg.team_a_territory_col: cfg.col_team_a_territory,
    })
    data.team_a_align = clean_npi(raw_a, cfg.col_npi)
    data.team_a_align[cfg.col_team_a_territory] = (
        data.team_a_align[cfg.col_team_a_territory].astype(int)
    )
    print(f"  Team A align:     {data.team_a_align[cfg.col_npi].nunique():,} NPIs")

    # C4. Team B alignment
    if "team_b_align" in dfs:
        raw_b = dfs["team_b_align"].copy()
    else:
        raw_b = pd.read_csv(cfg.team_b_align_path)
    raw_b = raw_b.rename(columns={
        cfg.team_b_npi_col: cfg.col_npi,
        cfg.team_b_territory_col: cfg.col_team_b_territory,
    })
    data.team_b_align = clean_npi(raw_b, cfg.col_npi)
    data.team_b_align[cfg.col_team_b_territory] = (
        data.team_b_align[cfg.col_team_b_territory].astype(int)
    )
    print(f"  Team B align:     {data.team_b_align[cfg.col_npi].nunique():,} NPIs")

    # C5. Portfolio-drug decile
    if "portfolio_decile" in dfs:
        data.portfolio_decile = clean_npi(dfs["portfolio_decile"].copy(), cfg.col_npi)
        data.portfolio_decile = (
            data.portfolio_decile
            .groupby(cfg.col_npi, as_index=False)
            .agg({cfg.col_portfolio_decile: "max"})
        )
        print(f"  Portfolio decile: {len(data.portfolio_decile):,} NPIs (in-memory)")
    elif isinstance(cfg.portfolio_decile_path, str) and cfg.portfolio_decile_path:
        data.portfolio_decile = clean_npi(
            pd.read_csv(cfg.portfolio_decile_path), cfg.col_npi
        )
        data.portfolio_decile = (
            data.portfolio_decile
            .groupby(cfg.col_npi, as_index=False)
            .agg({cfg.col_portfolio_decile: "max"})
        )
        print(f"  Portfolio decile: {len(data.portfolio_decile):,} NPIs")

    # C6. Priority targets
    if "priority_targets" in dfs:
        pt_raw = dfs["priority_targets"].copy()
    elif isinstance(cfg.priority_target_path, str) and cfg.priority_target_path:
        pt_raw = pd.read_csv(cfg.priority_target_path)
    else:
        pt_raw = None

    if pt_raw is not None:
        pt_npi = resolve_col(pt_raw, [cfg.col_npi, "npi_number", "NPI"])
        if pt_npi and pt_npi != cfg.col_npi:
            pt_raw = pt_raw.rename(columns={pt_npi: cfg.col_npi})

        pt_flag = resolve_col(pt_raw, [
            cfg.col_priority_flag, "PRIORITY_TARGET_FLAG",
            "PRIORITY_TARGET", "PRIORITY_TARGET_FLAG_Y",
        ])
        if pt_flag:
            pt_raw["_is_pt"] = yn_binary(pt_raw[pt_flag])
            data.priority_targets = (
                pt_raw[pt_raw["_is_pt"] == 1]
                .drop_duplicates(cfg.col_npi)[[cfg.col_npi]]
                .copy()
            )
            data.priority_targets = clean_npi(data.priority_targets, cfg.col_npi)
            data.priority_targets["PRIORITY_TARGET_FLAG"] = 1
        print(f"  Priority targets: {len(data.priority_targets) if data.priority_targets is not None else 0:,} NPIs")

    # C7. HCP reference table
    if "hcp_reference" in dfs:
        data.hcp_reference = clean_npi(
            dfs["hcp_reference"].copy().drop_duplicates(cfg.col_npi).fillna(0),
            cfg.col_npi,
        )
        print(f"  HCP reference:    {len(data.hcp_reference):,} NPIs (in-memory)")
    else:
        data.hcp_reference = clean_npi(
            pd.read_csv(cfg.hcp_reference_path).drop_duplicates(cfg.col_npi).fillna(0),
            cfg.col_npi,
        )
        print(f"  HCP reference:    {len(data.hcp_reference):,} NPIs")

    print("  Done.\n")
    return data
