"""
Sanity Check: Missing NPIs
============================
Returns rows from old plan where the NPI is not present in new plan.

Run after pipeline.run() returns results.
"""

import pandas as pd


def find_missing_npis(old_df, new_results, old_col_npi="npi",
                      new_col_npi="NPI_STR"):
    """Return old plan rows for NPIs missing from the new plan.

    Args:
        old_df: Old plan DataFrame.
        new_results: PipelineOutput from pipeline.run().
        old_col_npi: NPI column name in old plan.
        new_col_npi: NPI column name in new plan.

    Returns:
        DataFrame of old plan rows where NPI is not in new plan.
    """
    old_npis = set(old_df[old_col_npi].astype(str))
    new_npis = set(new_results.portfolio[new_col_npi].astype(str))

    missing_set = old_npis - new_npis
    missing = old_df[old_df[old_col_npi].astype(str).isin(missing_set)].copy()

    print(f"NPIs in old but not new: {len(missing_set):,}")
    print(f"Rows returned: {len(missing):,}")

    return missing


# ── Usage ───────────────────────────────────────────────────────────────
# from sanity_checks.missing_npis import find_missing_npis
# missing = find_missing_npis(old_df, results, old_col_npi="npi")
# display(missing)
