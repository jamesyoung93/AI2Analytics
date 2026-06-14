"""Data loading for the generalized detail optimization template."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from ai2analytics.templates.detail_optimization.loader import LoadedData
from ai2analytics.templates.generalized.config import GeneralizedDetailOptimizationConfig
from ai2analytics.templates.generalized.reporting import GeneralizedReporter
from ai2analytics.utils import clean_npi, require_columns, resolve_col, yn_binary

_CAPACITY_COL = "_capacity"


@dataclass
class GeneralizedLoadedData(LoadedData):
    """Loaded data plus optional capacity tables."""

    team_a_capacity: pd.DataFrame = field(default_factory=pd.DataFrame)
    team_b_capacity: pd.DataFrame = field(default_factory=pd.DataFrame)


def load_data(
    cfg: GeneralizedDetailOptimizationConfig,
    spark: Any = None,
    dataframes: dict[str, pd.DataFrame] | None = None,
    reporter: GeneralizedReporter | None = None,
) -> GeneralizedLoadedData:
    """Load all data sources per config."""
    dfs = dataframes or {}
    reporter = reporter or GeneralizedReporter(cfg.verbose, cfg.warn_on_defaults)
    data = GeneralizedLoadedData()

    print("=" * 70)
    print("STAGE C: Loading data (generalized)")
    print("=" * 70)

    data.hcp_weekly = _load_hcp_weekly(cfg, spark=spark, dataframes=dfs, reporter=reporter)
    print(f"  HCP weekly:       {len(data.hcp_weekly):,} rows")

    data.calls = _load_calls(cfg, spark=spark, dataframes=dfs)
    print(f"  Calls:            {len(data.calls):,} rows")

    data.team_a_align = _load_alignment(
        raw=dfs.get("team_a_align"),
        path=cfg.team_a_align_path,
        npi_col=cfg.team_a_npi_col,
        territory_col=cfg.team_a_territory_col,
        output_territory_col=cfg.col_team_a_territory,
        cfg=cfg,
        label=cfg.team_a_label,
    )
    print(f"  {cfg.team_a_label} align:     {data.team_a_align[cfg.col_npi].nunique():,} NPIs")

    data.team_b_align = _load_alignment(
        raw=dfs.get("team_b_align"),
        path=cfg.team_b_align_path,
        npi_col=cfg.team_b_npi_col,
        territory_col=cfg.team_b_territory_col,
        output_territory_col=cfg.col_team_b_territory,
        cfg=cfg,
        label=cfg.team_b_label,
    )
    print(f"  {cfg.team_b_label} align:     {data.team_b_align[cfg.col_npi].nunique():,} NPIs")

    data.portfolio_decile = _load_portfolio_decile(cfg, dfs)
    if not data.portfolio_decile.empty:
        print(f"  Portfolio decile: {len(data.portfolio_decile):,} NPIs")
    else:
        print("  Portfolio decile: not supplied")

    data.priority_targets = _load_priority_targets(cfg, dfs, reporter)
    if data.priority_targets is not None:
        print(f"  Priority targets: {len(data.priority_targets):,} NPIs")
    else:
        print("  Priority targets: not supplied")

    data.hcp_reference = _load_hcp_reference(cfg, dfs)
    print(f"  HCP reference:    {len(data.hcp_reference):,} NPIs")

    data.team_a_capacity = _load_capacity(
        raw=dfs.get("team_a_capacity"),
        path=cfg.team_a_capacity_path,
        territory_col=cfg.team_a_capacity_territory_col,
        value_col=cfg.team_a_capacity_value_col,
        output_territory_col=cfg.col_team_a_territory,
        aggregation=cfg.capacity_aggregation,
        label=cfg.team_a_label,
        reporter=reporter,
    )
    data.team_b_capacity = _load_capacity(
        raw=dfs.get("team_b_capacity"),
        path=cfg.team_b_capacity_path,
        territory_col=cfg.team_b_capacity_territory_col,
        value_col=cfg.team_b_capacity_value_col,
        output_territory_col=cfg.col_team_b_territory,
        aggregation=cfg.capacity_aggregation,
        label=cfg.team_b_label,
        reporter=reporter,
    )
    if not data.team_a_capacity.empty:
        print(f"  {cfg.team_a_label} capacity:  {len(data.team_a_capacity):,} territories")
    if not data.team_b_capacity.empty:
        print(f"  {cfg.team_b_label} capacity:  {len(data.team_b_capacity):,} territories")

    _normalize_npi_types(cfg, data)

    print("  Done.\n")
    return data


def _load_hcp_weekly(
    cfg: GeneralizedDetailOptimizationConfig,
    spark: Any,
    dataframes: dict[str, pd.DataFrame],
    reporter: GeneralizedReporter,
) -> pd.DataFrame:
    if "hcp_weekly" in dataframes:
        df = dataframes["hcp_weekly"].copy()
        source = "in-memory"
    else:
        if spark is None:
            raise RuntimeError("spark session is required for loading hcp_weekly_table")
        spark_df = spark.table(cfg.hcp_weekly_table)
        if cfg.apply_hcp_filter:
            if cfg.hcp_filter_col in spark_df.columns:
                spark_df = spark_df.where(
                    f"{cfg.hcp_filter_col} LIKE '{cfg.hcp_filter_val}'"
                )
            else:
                reporter.warn(
                    f"apply_hcp_filter=True but {cfg.hcp_filter_col!r} was not found "
                    "in hcp_weekly_table; loading all rows instead."
                )
        df = spark_df.toPandas()
        source = cfg.hcp_weekly_table

    if cfg.apply_hcp_filter and cfg.hcp_filter_col in df.columns:
        before = len(df)
        df = df[df[cfg.hcp_filter_col].astype(str).eq(str(cfg.hcp_filter_val))].copy()
        reporter.progress(
            f"Applied HCP filter {cfg.hcp_filter_col} == {cfg.hcp_filter_val!r}: "
            f"{before:,} -> {len(df):,} rows from {source}."
        )
    elif cfg.apply_hcp_filter:
        reporter.warn(
            f"apply_hcp_filter=True but {cfg.hcp_filter_col!r} was not found in "
            f"{source}; loading all rows instead."
        )

    return df


def _load_calls(
    cfg: GeneralizedDetailOptimizationConfig,
    spark: Any,
    dataframes: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if "calls" in dataframes:
        return dataframes["calls"].copy()
    if spark is None:
        raise RuntimeError("spark session is required for loading calls_table")
    return spark.table(cfg.calls_table).toPandas()


def _load_alignment(
    raw: pd.DataFrame | None,
    path: str,
    npi_col: str,
    territory_col: str,
    output_territory_col: str,
    cfg: GeneralizedDetailOptimizationConfig,
    label: str,
) -> pd.DataFrame:
    df = raw.copy() if raw is not None else pd.read_csv(path)
    require_columns(df, [npi_col, territory_col], context=f"{label} alignment")
    df = df.rename(columns={npi_col: cfg.col_npi, territory_col: output_territory_col})
    df = clean_npi(df, cfg.col_npi)
    df[output_territory_col] = _normalize_territory(df[output_territory_col])
    return df


def _load_portfolio_decile(
    cfg: GeneralizedDetailOptimizationConfig,
    dataframes: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if "portfolio_decile" in dataframes:
        df = dataframes["portfolio_decile"].copy()
    elif isinstance(cfg.portfolio_decile_path, str) and cfg.portfolio_decile_path:
        df = pd.read_csv(cfg.portfolio_decile_path)
    else:
        return pd.DataFrame()

    require_columns(df, [cfg.col_npi, cfg.col_portfolio_decile], context="portfolio_decile")
    df = clean_npi(df, cfg.col_npi)
    return (
        df.groupby(cfg.col_npi, as_index=False)
        .agg({cfg.col_portfolio_decile: "max"})
    )


def _load_priority_targets(
    cfg: GeneralizedDetailOptimizationConfig,
    dataframes: dict[str, pd.DataFrame],
    reporter: GeneralizedReporter,
) -> pd.DataFrame | None:
    if "priority_targets" in dataframes:
        pt_raw = dataframes["priority_targets"].copy()
    elif isinstance(cfg.priority_target_path, str) and cfg.priority_target_path:
        pt_raw = pd.read_csv(cfg.priority_target_path)
    else:
        return None

    pt_npi = resolve_col(pt_raw, [cfg.col_npi, "npi_number", "NPI"])
    if pt_npi is None:
        raise ValueError(
            "Priority target input is present but no NPI column was found. "
            f"Available columns: {sorted(pt_raw.columns.tolist())}"
        )
    if pt_npi != cfg.col_npi:
        pt_raw = pt_raw.rename(columns={pt_npi: cfg.col_npi})

    pt_flag = resolve_col(pt_raw, [
        cfg.col_priority_flag,
        "PRIORITY_TARGET_FLAG",
        "PRIORITY_TARGET",
        "PRIORITY_TARGET_FLAG_Y",
    ])
    if pt_flag:
        pt_raw["_is_pt"] = yn_binary(pt_raw[pt_flag])
        priority = pt_raw[pt_raw["_is_pt"] == 1].drop_duplicates(cfg.col_npi)[[cfg.col_npi]]
    else:
        reporter.warn(
            "Priority target input has no recognizable priority flag column; treating "
            "every row in that input as a priority target."
        )
        priority = pt_raw.drop_duplicates(cfg.col_npi)[[cfg.col_npi]]

    priority = clean_npi(priority, cfg.col_npi)
    priority["PRIORITY_TARGET_FLAG"] = 1
    return priority


def _load_hcp_reference(
    cfg: GeneralizedDetailOptimizationConfig,
    dataframes: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    if "hcp_reference" in dataframes:
        df = dataframes["hcp_reference"].copy()
    else:
        df = pd.read_csv(cfg.hcp_reference_path)
    return clean_npi(df.drop_duplicates(cfg.col_npi).fillna(0), cfg.col_npi)


def _load_capacity(
    raw: pd.DataFrame | None,
    path: str | None,
    territory_col: str,
    value_col: str,
    output_territory_col: str,
    aggregation: str,
    label: str,
    reporter: GeneralizedReporter,
) -> pd.DataFrame:
    if raw is None and not path:
        return pd.DataFrame()

    df = raw.copy() if raw is not None else pd.read_csv(path)
    require_columns(df, [territory_col, value_col], context=f"{label} capacity")
    df = df.rename(columns={territory_col: output_territory_col, value_col: _CAPACITY_COL})
    df[output_territory_col] = _normalize_territory(df[output_territory_col])
    df[_CAPACITY_COL] = pd.to_numeric(df[_CAPACITY_COL], errors="coerce")

    missing_capacity = df[_CAPACITY_COL].isna().sum()
    if missing_capacity:
        reporter.warn(
            f"{label} capacity input has {missing_capacity:,} rows with non-numeric "
            "capacity; those rows will be treated as zero."
        )
        df[_CAPACITY_COL] = df[_CAPACITY_COL].fillna(0)

    if aggregation == "sum":
        grouped = df.groupby(output_territory_col, as_index=False)[_CAPACITY_COL].sum()
    elif aggregation == "max":
        grouped = df.groupby(output_territory_col, as_index=False)[_CAPACITY_COL].max()
    else:
        grouped = df.drop_duplicates(output_territory_col)[[output_territory_col, _CAPACITY_COL]]

    return grouped


def _normalize_territory(series: pd.Series) -> pd.Series:
    return series.fillna("0").astype(str).str.strip().replace({"": "0", "nan": "0"})


def _normalize_npi_types(
    cfg: GeneralizedDetailOptimizationConfig,
    data: GeneralizedLoadedData,
) -> None:
    def _clean_npi_col(df: pd.DataFrame, col: str) -> None:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(int)

    for src_df in [
        data.hcp_weekly,
        data.calls,
        data.team_a_align,
        data.team_b_align,
        data.hcp_reference,
    ]:
        _clean_npi_col(src_df, cfg.col_npi)
    if data.portfolio_decile is not None and not data.portfolio_decile.empty:
        _clean_npi_col(data.portfolio_decile, cfg.col_npi)
    if data.priority_targets is not None:
        _clean_npi_col(data.priority_targets, cfg.col_npi)
