"""
Sanity Check: Stage D NPI Drop Trace
=====================================
Traces NPI count through each step of Stage D (Feature Engineering)
to identify exactly where NPIs are being lost between Stage C and D.

Run this in a Databricks notebook cell after loading your HCP weekly data.
"""

import pandas as pd


def trace_stage_d_npi_drop(rheum, details, col_npi="npi", col_week="WEEK_ENDING",
                            col_referrals="PAT_COUNT_REFERRED", col_indication="INDC",
                            col_calls="HCP_F2F_CALLS"):
    """Trace NPI count through each Stage D step."""

    print("=" * 60)
    print("STAGE D NPI DROP TRACE")
    print("=" * 60)

    # Raw counts
    print(f"\n  C1. Raw HCP weekly NPIs:    {rheum[col_npi].nunique():,}")
    det_npi = "NPI" if "NPI" in details.columns else col_npi
    print(f"  C2. Raw calls NPIs:         {details[det_npi].nunique():,}")

    # D0. NPI normalization
    rheum = rheum.copy()
    details = details.copy()
    rheum[col_week] = pd.to_datetime(rheum[col_week])
    rheum[col_npi] = pd.to_numeric(rheum[col_npi], errors="coerce").fillna(0).astype(int)
    print(f"\n  D0. After NPI normalize:    {rheum[col_npi].nunique():,}")

    # Check for NPI=0 (was NaN/invalid)
    zero_count = (rheum[col_npi] == 0).sum()
    if zero_count:
        print(f"      WARNING: {zero_count:,} rows have NPI=0 (were NaN/invalid)")

    # D1. Condense to one row per NPI x WEEK
    grp_max_cols = [
        c for c in rheum.columns
        if c not in {col_referrals, col_indication, col_week, col_npi}
        and rheum[c].dtype != "O"
    ]
    condensed = (
        rheum.groupby([col_npi, col_week])
        .agg({col_referrals: "sum", **{c: "max" for c in grp_max_cols}})
        .reset_index()
    )
    print(f"  D1. After condense:         {condensed[col_npi].nunique():,}")

    # Check if indication grouping caused the drop
    if col_indication in rheum.columns:
        npis_with_multi_indc = (
            rheum.groupby([col_npi, col_week])[col_indication]
            .nunique()
            .reset_index()
        )
        multi = npis_with_multi_indc[npis_with_multi_indc[col_indication] > 1]
        print(f"      NPIs with multiple indications per week: {multi[col_npi].nunique():,}")

    # D3. Drop CALL*DUP columns
    drop_dup = [c for c in condensed.columns if "CALL" in c.upper() and "DUP" in c.upper()]
    condensed.drop(columns=drop_dup, inplace=True, errors="ignore")
    print(f"  D3. After drop CALL*DUP:    {condensed[col_npi].nunique():,}")
    if drop_dup:
        print(f"      Dropped columns: {drop_dup}")

    # D5. Calls merge
    details[col_week] = pd.to_datetime(details[col_week])
    details[det_npi] = pd.to_numeric(details[det_npi], errors="coerce").fillna(0).astype(int)
    det_agg = (
        details.groupby([det_npi, col_week])[col_calls]
        .sum().reset_index()
        .rename(columns={det_npi: col_npi, col_calls: "TS_CALLS"})
    )
    merged = condensed.merge(det_agg, on=[col_npi, col_week], how="left").fillna({"TS_CALLS": 0})
    print(f"  D5. After calls merge:      {merged[col_npi].nunique():,}")

    # Summary
    raw = rheum[col_npi].nunique()
    final = merged[col_npi].nunique()
    diff = raw - final
    print(f"\n  Total NPIs lost C->D:       {diff:,}")
    if diff == 0:
        print("  No NPIs lost — drop is happening later.")

    return merged


# ── Usage ───────────────────────────────────────────────────────────────
# trace_stage_d_npi_drop(rheum, details)
