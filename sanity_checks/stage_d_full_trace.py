"""
Sanity Check: Full Stage D + F NPI Trace
==========================================
Extends the basic Stage D trace to cover ALL steps including
D2 (first-write merge), D7-D8 (targets), D9 (feature filtering),
and the Stage F planning snapshot.

Run this after stage_d_npi_trace confirms D5 count matches Stage C.
"""

import numpy as np
import pandas as pd

from ai2analytics.utils import forward_sum
from sanity_checks.stage_d_npi_trace import trace_stage_d_npi_drop


def full_trace(rheum, details, col_npi="npi", col_week="WEEK_ENDING",
               col_referrals="PAT_COUNT_REFERRED", col_calls="HCP_F2F_CALLS",
               target_horizon_weeks=4, na_threshold=0.10):
    """Trace NPI count through every step from Stage C to Stage F."""

    # Run the basic trace first (C through D5)
    merged = trace_stage_d_npi_drop(
        rheum, details,
        col_npi=col_npi, col_week=col_week,
        col_referrals=col_referrals, col_calls=col_calls,
    )

    print("\n" + "=" * 60)
    print("EXTENDED TRACE (D2 through F)")
    print("=" * 60)

    # D2. First-write week merge
    merged["ref"] = (merged[col_referrals] > 0).astype(int)
    first_write = (
        merged[merged["ref"] == 1]
        .groupby(col_npi)[col_week].min()
        .rename("first_write_week")
    )
    merged = merged.merge(first_write, on=col_npi, how="left")
    print(f"\n  D2. After first-write merge:  {merged[col_npi].nunique():,}")

    # D7. Forward-looking call count
    merged["TS_CALLS_next"] = (
        merged.groupby(col_npi)["TS_CALLS"]
        .apply(lambda s: forward_sum(s, target_horizon_weeks))
        .reset_index(level=0, drop=True)
    )
    print(f"  D7. After forward calls:      {merged[col_npi].nunique():,}")

    # D8. Targets
    merged["target_prob"] = (
        merged.groupby(col_npi)["ref"]
        .apply(lambda s: forward_sum(s, target_horizon_weeks))
        .reset_index(level=0, drop=True)
        .gt(0).astype(int)
    )
    merged["target_cnt"] = (
        merged.groupby(col_npi)[col_referrals]
        .apply(lambda s: forward_sum(s, target_horizon_weeks))
        .reset_index(level=0, drop=True)
        .fillna(0)
    )
    merged["target_look"] = (
        (merged["target_prob"] == 1)
        & (merged["first_write_week"].isna() | (merged["first_write_week"] > merged[col_week]))
    ).astype(int)
    print(f"  D8. After targets:            {merged[col_npi].nunique():,}")

    # D9. Feature filtering
    num_cols = merged.select_dtypes(include=[np.number]).columns.tolist()
    always_exclude = {col_npi, "ref", "target_prob", "target_cnt", "target_look", "TS_CALLS"}
    na_pct = merged[num_cols].replace([np.inf, -np.inf], np.nan).isna().mean()
    feat_base = [c for c in num_cols if c not in always_exclude and na_pct[c] < na_threshold]
    print(f"  D9. Feature list:             {len(feat_base)} features")
    print(f"  D9. NPIs still present:       {merged[col_npi].nunique():,}")

    # F. Planning snapshot
    max_week = merged[col_week].max()
    planning_date = max_week - pd.Timedelta(weeks=target_horizon_weeks)
    df_plan = merged[merged[col_week] == planning_date]
    print(f"\n  F.  Planning snapshot NPIs:   {df_plan[col_npi].nunique():,}")
    print(f"      Planning date:            {planning_date.date()}")
    print(f"      Max week:                 {max_week.date()}")

    # If there's still a gap, check what's missing
    all_npis = set(merged[col_npi].unique())
    plan_npis = set(df_plan[col_npi].unique())
    missing = all_npis - plan_npis

    if missing:
        missing_df = merged[merged[col_npi].isin(missing)]
        last_seen = missing_df.groupby(col_npi)[col_week].max()
        print(f"\n      Missing {len(missing):,} NPIs from planning snapshot:")
        print(f"      Last seen (describe):")
        print(f"        min:  {last_seen.min().date()}")
        print(f"        max:  {last_seen.max().date()}")
        print(f"        median: {last_seen.median().date()}")
    else:
        print(f"\n      No NPIs missing at planning date.")

    return merged


# ── Usage ───────────────────────────────────────────────────────────────
# from sanity_checks.stage_d_full_trace import full_trace
# full_trace(rheum, details)
