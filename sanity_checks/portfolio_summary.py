"""
Sanity Check: Portfolio Summary
================================
Filters out invalid rows (territory=0, NPI=0, zero-call NPIs)
then prints total calls, unique NPIs, and Y/N flag distributions.

Run after pipeline.run() returns results.
"""


def portfolio_summary(results, col_territory="Territory_ID", col_npi="NPI_STR",
                      col_calls="ALLOCATED_CALLS"):
    """Summarize the portfolio after filtering invalid rows."""

    pf = results.portfolio.copy()

    print("=" * 60)
    print("PORTFOLIO SUMMARY")
    print("=" * 60)
    print(f"\n  Raw rows: {len(pf):,}")

    # Filter out territory=0, NPI=0
    pf = pf[pf[col_territory] != 0]
    pf = pf[pf[col_npi] != "0"]

    # Drop NPIs with 0 total allocated calls
    npi_calls = pf.groupby(col_npi)[col_calls].sum()
    active_npis = npi_calls[npi_calls > 0].index
    pf = pf[pf[col_npi].isin(active_npis)]

    print(f"  After filtering: {len(pf):,} rows")
    print(f"\n  Total allocated calls: {pf[col_calls].sum():,}")
    print(f"  Unique NPIs: {pf[col_npi].nunique():,}")

    # Y/N flag distributions
    yn_cols = [c for c in pf.columns if set(pf[c].dropna().unique()) <= {"Y", "N"}]
    for col in yn_cols:
        counts = pf[col].value_counts()
        print(f"\n  {col}:")
        print(f"    Y: {counts.get('Y', 0):,}")
        print(f"    N: {counts.get('N', 0):,}")

    return pf


# ── Usage ───────────────────────────────────────────────────────────────
# from sanity_checks.portfolio_summary import portfolio_summary
# pf = portfolio_summary(results)
