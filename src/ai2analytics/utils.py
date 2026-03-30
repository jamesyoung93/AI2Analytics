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
    """Sum of current + next (window-1) values. Vectorized via reverse-rolling."""
    return series[::-1].rolling(window, min_periods=1).sum()[::-1]


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
