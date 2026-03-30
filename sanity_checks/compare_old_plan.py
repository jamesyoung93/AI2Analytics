"""
Sanity Check: Compare Old Plan vs New Plan
============================================
Compares NPI counts and total calls between an old manually-built
plan and the new pipeline output. Takes the max allocated calls
per NPI per team before summing.

Run after pipeline.run() returns results.
"""

import pandas as pd


def compare_plans(old_df, new_results, col_npi="NPI_STR", col_calls="ALLOCATED_CALLS",
                  col_source="ROW_SOURCE", old_col_npi=None, old_col_calls=None,
                  old_col_source=None):
    """Compare old plan DataFrame against new pipeline results.

    Args:
        old_df: Old plan DataFrame (manually imported).
        new_results: PipelineOutput from pipeline.run().
        col_npi: NPI column in new plan.
        col_calls: Calls column in new plan.
        col_source: Team source column in new plan.
        old_col_npi: NPI column in old plan (defaults to col_npi).
        old_col_calls: Calls column in old plan (defaults to col_calls).
        old_col_source: Team source column in old plan (defaults to col_source).
    """
    old_col_npi = old_col_npi or col_npi
    old_col_calls = old_col_calls or col_calls
    old_col_source = old_col_source or col_source

    new_df = new_results.portfolio.copy()

    # Filter out invalid rows
    new_df = new_df[new_df[col_npi] != "0"]
    if "Territory_ID" in new_df.columns:
        new_df = new_df[new_df["Territory_ID"] != 0]

    print("=" * 60)
    print("OLD PLAN vs NEW PLAN COMPARISON")
    print("=" * 60)

    # Max calls per NPI per team, then sum
    def _summarize(df, npi_col, calls_col, source_col, label):
        deduped = df.groupby([npi_col, source_col])[calls_col].max().reset_index()
        total_calls = deduped[calls_col].sum()
        unique_npis = deduped[npi_col].nunique()
        active = deduped[deduped[calls_col] > 0]
        active_npis = active[npi_col].nunique()
        active_calls = active[calls_col].sum()

        print(f"\n  {label}:")
        print(f"    Unique NPIs:        {unique_npis:,}")
        print(f"    NPIs with calls:    {active_npis:,}")
        print(f"    Total calls:        {total_calls:,.0f}")

        if source_col in df.columns:
            for team in sorted(deduped[source_col].unique()):
                team_slice = deduped[deduped[source_col] == team]
                team_active = team_slice[team_slice[calls_col] > 0]
                print(f"    {team}: {team_active[npi_col].nunique():,} NPIs, "
                      f"{team_active[calls_col].sum():,.0f} calls")

        return deduped

    old_deduped = _summarize(old_df, old_col_npi, old_col_calls, old_col_source, "OLD PLAN")
    new_deduped = _summarize(new_df, col_npi, col_calls, col_source, "NEW PLAN")

    # Overlap analysis
    old_npis = set(old_deduped[old_col_npi].astype(str).unique())
    new_npis = set(new_deduped[col_npi].astype(str).unique())

    both = old_npis & new_npis
    only_old = old_npis - new_npis
    only_new = new_npis - old_npis

    print(f"\n  OVERLAP:")
    print(f"    In both:     {len(both):,}")
    print(f"    Only in old: {len(only_old):,}")
    print(f"    Only in new: {len(only_new):,}")

    # Call delta for shared NPIs
    if both:
        old_calls = (
            old_deduped[old_deduped[old_col_npi].astype(str).isin(both)]
            .groupby(old_col_npi)[old_col_calls].sum()
        )
        new_calls = (
            new_deduped[new_deduped[col_npi].astype(str).isin(both)]
            .groupby(col_npi)[col_calls].sum()
        )
        old_total = old_calls.sum()
        new_total = new_calls.sum()
        print(f"\n  SHARED NPIs ({len(both):,}):")
        print(f"    Old total calls: {old_total:,.0f}")
        print(f"    New total calls: {new_total:,.0f}")
        print(f"    Delta:           {new_total - old_total:+,.0f}")


# ── Usage ───────────────────────────────────────────────────────────────
# from sanity_checks.compare_old_plan import compare_plans
# compare_plans(old_df, results)
#
# If old plan has different column names:
# compare_plans(old_df, results, old_col_npi="npi", old_col_calls="DRUG_A_CALLS")
