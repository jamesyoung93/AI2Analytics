"""
Sanity Check: Planning Date NPI Drop
=====================================
Diagnoses why Stage F (Scenario Scoring) has fewer NPIs than Stage C (Data Loading).

The planning snapshot selects only NPIs that have a row at exactly:
    planning_date = max_week - target_horizon_weeks

If an NPI doesn't have an observation on that exact week, it's excluded
from scoring and the final call plan.

Run this in a Databricks notebook cell after loading your HCP weekly data.
"""

from datetime import timedelta
import pandas as pd

# ── Configure these to match your pipeline ──────────────────────────────
# rheum = your HCP weekly DataFrame
# col_npi = "npi"
# col_week = "WEEK_ENDING"
# target_horizon_weeks = 4


def check_planning_date_coverage(rheum, col_npi="npi", col_week="WEEK_ENDING",
                                  target_horizon_weeks=4):
    """Diagnose NPI drop between data loading and scenario scoring."""

    rheum[col_week] = pd.to_datetime(rheum[col_week])
    max_week = rheum[col_week].max()
    planning_date = max_week - timedelta(weeks=target_horizon_weeks)

    total_npis = rheum[col_npi].nunique()
    npis_at_plan = rheum[rheum[col_week] == planning_date][col_npi].nunique()
    npis_at_max = rheum[rheum[col_week] == max_week][col_npi].nunique()

    print("=" * 60)
    print("PLANNING DATE NPI COVERAGE")
    print("=" * 60)
    print(f"  Max week:              {max_week.date()}")
    print(f"  Planning date:         {planning_date.date()}")
    print(f"  Total unique NPIs:     {total_npis:,}")
    print(f"  NPIs at planning date: {npis_at_plan:,}")
    print(f"  NPIs at max week:      {npis_at_max:,}")
    print(f"  NPIs missing at plan:  {total_npis - npis_at_plan:,}")

    # Check if planning date exists in the data
    exists = planning_date in rheum[col_week].values
    print(f"\n  Planning date exists in data: {exists}")

    if not exists:
        weeks = sorted(rheum[col_week].unique())
        nearby = [w for w in weeks if abs((w - planning_date).days) < 14]
        print(f"  Nearest weeks: {[pd.Timestamp(w).date() for w in nearby]}")
        print("\n  FIX: Planning date doesn't land on an actual week in your data.")
        print("  Adjust target_horizon_weeks or check WEEK_ENDING date alignment.")
        return

    # Show NPI coverage over the last N weeks
    print(f"\n  NPIs per week (last 10 weeks):")
    coverage = (
        rheum.groupby(col_week)[col_npi]
        .nunique()
        .sort_index()
        .tail(10)
    )
    for week, count in coverage.items():
        marker = " <-- planning" if week == planning_date else ""
        marker = " <-- max" if week == max_week else marker
        print(f"    {pd.Timestamp(week).date()}: {count:,} NPIs{marker}")

    # Identify missing NPIs
    all_npis = set(rheum[col_npi].unique())
    plan_npis = set(rheum[rheum[col_week] == planning_date][col_npi].unique())
    missing = all_npis - plan_npis

    if missing:
        # When did the missing NPIs last appear?
        missing_df = rheum[rheum[col_npi].isin(missing)]
        last_seen = missing_df.groupby(col_npi)[col_week].max()
        print(f"\n  Missing NPIs last seen:")
        print(f"    {last_seen.describe()}")
        print(f"\n  These {len(missing):,} NPIs dropped out of the data before the planning date.")


# ── Usage ───────────────────────────────────────────────────────────────
# check_planning_date_coverage(rheum, col_npi="npi", col_week="WEEK_ENDING")
