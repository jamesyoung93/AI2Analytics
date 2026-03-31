"""EU Account Segmentation Demo
================================

Demonstrates the ai2analytics segmentation pipeline using synthetic
European pharma account data.  The key learning point: the **same**
SegmentationPipeline code handles a completely different schema by
simply changing the configuration.

Schema differences compared to the US HCP demo:
    - Entity ID:   PRESCRIBER_ID  (not ``npi``)
    - Features:    UNITS_SOLD_L12M, TIER  (not IL-17/IL-23 Rx counts)
    - Entity type: Hospital/clinic accounts (not individual HCPs)

What this script covers:
    1. Generating synthetic EU account reference data (3 000 accounts).
    2. Configuring the pipeline with ``method='auto'`` so it tries both
       KMeans and hierarchical clustering and picks the better fit.
    3. Running the pipeline and inspecting results.
    4. Highlighting schema portability -- same code, different data.

Run from the repository root:
    python demos/notebooks/demo_segmentation_eu.py
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ============================================================================
# Step 1 -- Generate synthetic EU account data
# ============================================================================
print("\n" + "#" * 70)
print("# STEP 1: Generate synthetic EU account data")
print("#" * 70)

from demos.synthetic_eu_account import generate_all  # noqa: E402

dfs = generate_all()

acct_ref = dfs["account_reference"]
print(f"\naccount_reference shape: {acct_ref.shape}")
print(acct_ref.head(10).to_string(index=False))

print("\nKey columns for segmentation:")
print(f"  Entity ID column:  PRESCRIBER_ID  ({acct_ref['PRESCRIBER_ID'].nunique():,} accounts)")
print(f"  Feature 1:         UNITS_SOLD_L12M  (last-12-month unit volume)")
print(f"  Feature 2:         TIER             (1=highest value, 4=lowest)")

print("\nNote the schema differences from the US HCP demo:")
print("  US  -> col_entity_id='npi',            features=['IL_17_TRX_L12M', 'IL_23_TRX_L12M']")
print("  EU  -> col_entity_id='PRESCRIBER_ID',  features=['UNITS_SOLD_L12M', 'TIER']")
print("  Same pipeline code handles both.")

# ============================================================================
# Step 2 -- Configure the pipeline
# ============================================================================
print("\n" + "#" * 70)
print("# STEP 2: Configure the segmentation pipeline")
print("#" * 70)

from ai2analytics.templates.segmentation import (  # noqa: E402
    SegmentationConfig,
    SegmentationPipeline,
)

cfg = SegmentationConfig(
    analysis_name="eu_account_segmentation",
    col_entity_id="PRESCRIBER_ID",
    feature_columns=["UNITS_SOLD_L12M", "TIER"],
    n_segments=3,
    method="auto",         # tries both kmeans AND hierarchical, picks best
    output_csv="demos/data/seg_eu_output.csv",
)

print(f"\nConfiguration:")
print(f"  analysis_name:     {cfg.analysis_name}")
print(f"  col_entity_id:     {cfg.col_entity_id}   <-- different from US ('npi')")
print(f"  feature_columns:   {cfg.feature_columns}  <-- different features")
print(f"  n_segments:        {cfg.n_segments}")
print(f"  method:            {cfg.method}   <-- 'auto' compares kmeans vs hierarchical")
print(f"  output_csv:        {cfg.output_csv}")

# ============================================================================
# Step 3 -- Run the pipeline
# ============================================================================
print("\n" + "#" * 70)
print("# STEP 3: Run the segmentation pipeline")
print("#" * 70)

print("\nPassing the EU account DataFrame directly via dataframes=.")
print("The pipeline will normalise features, fit both clustering methods,")
print("and select whichever achieves a higher silhouette score.\n")

pipeline = SegmentationPipeline()
output = pipeline.run(cfg, dataframes={"entity_data": dfs["account_reference"]})

# ============================================================================
# Step 4 -- Inspect results
# ============================================================================
print("\n" + "#" * 70)
print("# STEP 4: Inspect results")
print("#" * 70)

# 4a. Assignments
print("\n--- Segment Assignments (first 10 rows) ---")
print(output.assignments.head(10).to_string(index=False))

# 4b. Profiles
print("\n--- Segment Profiles (feature means) ---")
print(output.profiles.to_string(index=False))

print("\nInterpretation:")
print("  UNITS_SOLD_L12M shows volume per segment.")
print("  TIER is ordinal (1=top, 4=bottom) -- a lower mean indicates")
print("  the segment contains higher-tier accounts.")

# 4c. Summary
print("\n--- Summary Statistics ---")
for key, val in output.summary_stats.items():
    print(f"  {key}: {val}")

# 4d. Which method won?
print(f"\nThe 'auto' method selected: {output.summary_stats['method']}")
print(f"Silhouette score:           {output.summary_stats['silhouette_score']:.4f}")

# 4e. Segment size distribution
print("\n--- Segment Sizes ---")
sizes = output.summary_stats.get("segment_sizes", {})
total = sum(sizes.values())
for seg in sorted(sizes):
    pct = sizes[seg] / total * 100
    print(f"  Segment {seg}: {sizes[seg]:>5,} accounts  ({pct:5.1f}%)")

# ============================================================================
# Step 5 -- Portability recap
# ============================================================================
print("\n" + "#" * 70)
print("# STEP 5: Schema portability recap")
print("#" * 70)

print("""
The segmentation pipeline does NOT hard-code any column names or entity
types.  The SegmentationConfig dataclass externalises every schema
detail:

    US HCP demo:
        col_entity_id   = 'npi'
        feature_columns = ['IL_17_TRX_L12M', 'IL_23_TRX_L12M']

    EU Account demo:
        col_entity_id   = 'PRESCRIBER_ID'
        feature_columns = ['UNITS_SOLD_L12M', 'TIER']

Both demos used the exact same SegmentationPipeline class.  This is the
core design principle of ai2analytics templates: one pipeline, many
schemas.
""")

# ============================================================================
# Done
# ============================================================================
print("#" * 70)
print("# DEMO COMPLETE")
print("#" * 70)
print(f"\nOutput written to: {os.path.abspath(cfg.output_csv)}")
