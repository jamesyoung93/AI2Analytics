"""Planning-cadence helpers for generalized detail optimization."""

from __future__ import annotations

import pandas as pd

SUPPORTED_CADENCES = {"weekly", "monthly", "quarterly"}


def normalize_cadence(cadence: str) -> str:
    """Normalize and validate a planning cadence string."""
    value = (cadence or "weekly").strip().lower()
    if value not in SUPPORTED_CADENCES:
        available = ", ".join(sorted(SUPPORTED_CADENCES))
        raise ValueError(f"Unsupported planning_cadence={cadence!r}. Use one of: {available}")
    return value


def add_periods(value, periods: int, cadence: str) -> pd.Timestamp:
    """Move a timestamp by a number of planning periods."""
    cadence = normalize_cadence(cadence)
    ts = pd.Timestamp(value)
    if cadence == "weekly":
        return ts + pd.DateOffset(weeks=periods)
    if cadence == "monthly":
        return ts + pd.DateOffset(months=periods)
    return ts + pd.DateOffset(months=3 * periods)


def resolve_existing_period(
    periods,
    target,
    prefer: str = "on_or_before",
) -> tuple[pd.Timestamp | None, bool]:
    """Resolve a target date to an observed period.

    Returns (period, used_fallback). The fallback direction is intentionally
    explicit so scoring can avoid looking into future periods while backtests
    can choose the first available test period after the target.
    """
    observed = sorted(pd.Timestamp(p) for p in periods)
    if not observed:
        return None, False

    target_ts = pd.Timestamp(target)
    for period in observed:
        if period == target_ts:
            return period, False

    if prefer == "on_or_before":
        candidates = [period for period in observed if period <= target_ts]
        return (candidates[-1], True) if candidates else (None, True)

    if prefer == "on_or_after":
        candidates = [period for period in observed if period >= target_ts]
        return (candidates[0], True) if candidates else (None, True)

    raise ValueError("prefer must be 'on_or_before' or 'on_or_after'")
