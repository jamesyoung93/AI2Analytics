"""
Sanity Check: Compare Old Plan vs New Plan
============================================
Compares NPI counts, total calls, Y/N flag distributions, and
per-NPI call correlation between an old manually-built plan and
the new pipeline output. Takes the max allocated calls per NPI
per team before summing.

Run after pipeline.run() returns results.
"""

import pandas as pd
import numpy as np


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

    Returns:
        Dict with keys: old_deduped, new_deduped, merged (per-NPI comparison df).
    """
    old_col_npi = old_col_npi or col_npi
    old_col_calls = old_col_calls or col_calls
    old_col_source = old_col_source or col_source

    old_df = old_df.copy()
    new_df = new_results.portfolio.copy()

    # Normalize NPI to string for matching
    old_df[old_col_npi] = old_df[old_col_npi].astype(str).str.strip()
    new_df[col_npi] = new_df[col_npi].astype(str).str.strip()

    # Filter out invalid rows in new
    new_df = new_df[new_df[col_npi] != "0"]
    if "Territory_ID" in new_df.columns:
        new_df = new_df[new_df["Territory_ID"] != 0]

    print("=" * 60)
    print("OLD PLAN vs NEW PLAN COMPARISON")
    print("=" * 60)

    # ── Row counts ──────────────────────────────────────────────────
    print(f"\n  OLD PLAN rows: {len(old_df):,}")
    print(f"  NEW PLAN rows: {len(new_df):,}")

    # ── Old plan Y/N flag distributions ─────────────────────────────
    yn_cols = [c for c in old_df.columns if set(old_df[c].dropna().unique()) <= {"Y", "N"}]
    if yn_cols:
        print(f"\n  OLD PLAN FLAG DISTRIBUTIONS:")
        for col in yn_cols:
            counts = old_df[col].value_counts()
            print(f"    {col}: Y={counts.get('Y', 0):,}  N={counts.get('N', 0):,}")

    # ── Max calls per NPI per team, then sum ────────────────────────
    def _summarize(df, npi_col, calls_col, source_col, label):
        deduped = df.groupby([npi_col, source_col])[calls_col].max().reset_index()
        total_calls = deduped[calls_col].sum()
        unique_npis = deduped[npi_col].nunique()
        active = deduped[deduped[calls_col] > 0]
        active_npis = active[npi_col].nunique()

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

    old_deduped = _summarize(old_df, old_col_npi, old_col_calls, old_col_source, "OLD PLAN (deduped)")
    new_deduped = _summarize(new_df, col_npi, col_calls, col_source, "NEW PLAN (deduped)")

    # ── Overlap analysis ────────────────────────────────────────────
    old_npis = set(old_deduped[old_col_npi].unique())
    new_npis = set(new_deduped[col_npi].unique())

    both = old_npis & new_npis
    only_old = old_npis - new_npis
    only_new = new_npis - old_npis

    print(f"\n  OVERLAP:")
    print(f"    In both:     {len(both):,}")
    print(f"    Only in old: {len(only_old):,}")
    print(f"    Only in new: {len(only_new):,}")

    # ── Per-NPI call correlation ────────────────────────────────────
    # Sum total calls per NPI (across teams)
    old_npi_calls = (
        old_deduped.groupby(old_col_npi)[old_col_calls].sum()
        .rename("old_calls")
    )
    new_npi_calls = (
        new_deduped.groupby(col_npi)[col_calls].sum()
        .rename("new_calls")
    )

    merged = pd.merge(
        old_npi_calls, new_npi_calls,
        left_index=True, right_index=True, how="outer",
    ).fillna(0)

    corr = merged["old_calls"].corr(merged["new_calls"])
    rank_corr = merged["old_calls"].corr(merged["new_calls"], method="spearman")

    print(f"\n  PER-NPI CALL CORRELATION (all NPIs):")
    print(f"    Pearson:  {corr:.4f}")
    print(f"    Spearman: {rank_corr:.4f}")

    # Correlation for shared NPIs only
    shared = merged.loc[merged.index.isin(both)]
    if len(shared) > 1:
        corr_shared = shared["old_calls"].corr(shared["new_calls"])
        rank_shared = shared["old_calls"].corr(shared["new_calls"], method="spearman")
        print(f"\n  PER-NPI CALL CORRELATION (shared NPIs only):")
        print(f"    Pearson:  {corr_shared:.4f}")
        print(f"    Spearman: {rank_shared:.4f}")

    # Call delta for shared NPIs
    if both:
        old_total = shared["old_calls"].sum()
        new_total = shared["new_calls"].sum()
        print(f"\n  SHARED NPIs ({len(both):,}):")
        print(f"    Old total calls: {old_total:,.0f}")
        print(f"    New total calls: {new_total:,.0f}")
        print(f"    Delta:           {new_total - old_total:+,.0f}")

    # Call movement summary
    merged["delta"] = merged["new_calls"] - merged["old_calls"]
    increased = (merged["delta"] > 0).sum()
    decreased = (merged["delta"] < 0).sum()
    unchanged = (merged["delta"] == 0).sum()
    print(f"\n  CALL MOVEMENT:")
    print(f"    Increased: {increased:,} NPIs")
    print(f"    Decreased: {decreased:,} NPIs")
    print(f"    Unchanged: {unchanged:,} NPIs")
    print(f"    Avg delta: {merged['delta'].mean():+.2f}")

    return {"old_deduped": old_deduped, "new_deduped": new_deduped, "merged": merged}


# ── Usage ───────────────────────────────────────────────────────────────
# from sanity_checks.compare_old_plan import compare_plans
# out = compare_plans(old_df, results)
#
# # Access the per-NPI comparison df:
# out["merged"].head()
#
# If old plan has different column names:
# compare_plans(old_df, results, old_col_npi="npi", old_col_calls="DRUG_A_CALLS")
