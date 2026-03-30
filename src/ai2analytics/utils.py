"""Shared utility functions for data cleaning and transformation."""

from __future__ import annotations

import numpy as np
import pandas as pd


def clean_npi(df: pd.DataFrame, col: str = "npi") -> pd.DataFrame:
    """Remove rows where NPI is '-' or missing, cast to int."""
    df = df[df[col].astype(str).str.strip() != "-"].copy()
    df = df[df[col].notna()].copy()
    df[col] = df[col].apply(lambda x: int(float(x)))
    return df


def yn_flag(series: pd.Series) -> pd.Series:
    """Normalize any flag column to 'Y' / 'N'."""
    s = series.fillna("").astype(str).str.strip().str.upper()
    return s.where(s.eq("Y"), "N")


def yn_binary(series: pd.Series) -> pd.Series:
    """Normalize flag to 1 / 0."""
    return yn_flag(series).map({"Y": 1, "N": 0}).astype(int)


def is_yes(series: pd.Series) -> pd.Series:
    """Boolean mask where value is 'Y'."""
    return yn_flag(series).eq("Y")


def make_decile(
    df: pd.DataFrame, src: str, tgt: str, n_bins: int = 8
) -> pd.DataFrame:
    """Positives -> qcut into n_bins (labels 2..n_bins+1); zeros -> 1."""
    df[src] = pd.to_numeric(df[src], errors="coerce").fillna(0)
    pos = df[src] > 0
    df.loc[pos, tgt] = (
        pd.qcut(df.loc[pos, src], q=n_bins, labels=False, duplicates="drop").astype(int) + 2
    )
    df.loc[~pos, tgt] = 1
    df[tgt] = df[tgt].astype(int)
    return df


def safe_fill(df: pd.DataFrame, cols: list[str], val: float = 0) -> pd.DataFrame:
    """Replace inf/-inf -> NaN -> fill with val."""
    df[cols] = df[cols].replace([np.inf, -np.inf], np.nan).fillna(val)
    return df


def forward_sum(series: pd.Series, window: int = 4) -> pd.Series:
    """Sum of current + next (window-1) values, treating NaN as 0."""
    filled = series.fillna(0)
    return filled[::-1].rolling(window, min_periods=1).sum()[::-1]


def resolve_col(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return first column from candidates that exists in df."""
    for c in candidates:
        if c in df.columns:
            return c
    return None


def require_columns(df: pd.DataFrame, columns: list[str], context: str = "") -> None:
    """Raise ValueError if any required columns are missing from df."""
    missing = [c for c in columns if c not in df.columns]
    if missing:
        label = f" ({context})" if context else ""
        raise ValueError(
            f"Missing required columns{label}: {missing}. "
            f"Available: {sorted(df.columns.tolist())}"
        )


def allowed_call_pairs(
    max_total: int,
    is_priority: bool = False,
    priority_totals: list[int] | None = None,
    require_mixed_at_max: bool = True,
    max_scenario: int = 4,
) -> list[tuple[int, int]]:
    """Generate valid (team_a, team_b) call-count pairs."""
    if priority_totals is None:
        priority_totals = [3, 4]
    out = []
    for a in range(max_total + 1):
        for b in range(max_total + 1 - a):
            total = a + b
            if require_mixed_at_max and total == max_scenario and min(a, b) == 0:
                continue
            if is_priority and total not in priority_totals:
                continue
            out.append((a, b))
    return out


def build_hcp_reference(
    df: pd.DataFrame,
    col_npi: str = "npi",
    col_referrals: str = "PAT_COUNT_REFERRED",
    col_target_flag: str | None = "TARGET_FLAG",
    il_rx_columns: list[str] | None = None,
) -> pd.DataFrame:
    """Derive an HCP reference file from HCP weekly data.

    Produces a one-row-per-NPI DataFrame with WRITER_FLAG, TARGET_FLAG,
    and any therapeutic-class Rx columns — suitable for use as the
    hcp_reference_path input to DetailOptimizationConfig.

    Args:
        df: HCP weekly DataFrame (one row per NPI x week).
        col_npi: NPI column name.
        col_referrals: Referral/prescription count column name.
        col_target_flag: Target flag column name, or None if all rows are targets.
        il_rx_columns: Therapeutic-class Rx column names to carry forward
                       (e.g. ["IL_17_TRX_L12M", "IL_23_TRX_L12M"]).
                       Takes max per NPI since these are rolling snapshots.
    """
    ref = df.groupby(col_npi).agg(
        total_refs=(col_referrals, "sum"),
    ).reset_index()

    ref["WRITER_FLAG"] = ref["total_refs"].gt(0).map({True: "Y", False: "N"})
    ref.drop(columns="total_refs", inplace=True)

    if col_target_flag and col_target_flag in df.columns:
        tgt = df.groupby(col_npi)[col_target_flag].first().reset_index()
        ref = ref.merge(tgt, on=col_npi, how="left")
    else:
        ref[col_target_flag or "TARGET_FLAG"] = "Y"

    if il_rx_columns:
        rx_cols = [c for c in il_rx_columns if c in df.columns]
        if rx_cols:
            rx = df.groupby(col_npi)[rx_cols].max().reset_index()
            ref = ref.merge(rx, on=col_npi, how="left")

    ref = ref.fillna(0)
    print(
        f"HCP reference built: {len(ref):,} NPIs, "
        f"{ref['WRITER_FLAG'].eq('Y').sum():,} writers"
    )
    return ref
