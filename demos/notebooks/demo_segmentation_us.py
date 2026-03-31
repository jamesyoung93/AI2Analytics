"""US HCP Segmentation Demo
============================

Demonstrates the ai2analytics segmentation pipeline using synthetic US
pharma HCP (Health Care Provider) data.  Everything runs locally with
plain pandas DataFrames -- no Spark required.

What this script covers:
    1. Generating synthetic HCP reference data (5 000 providers).
    2. Configuring the segmentation pipeline with KMeans (k=4).
    3. Running the pipeline end-to-end via the ``dataframes=`` pattern.
    4. Inspecting output: assignments, segment profiles, summary stats.
    5. Re-running with ``auto_select_k=True`` to let the pipeline pick k.

Run from the repository root:
    python demos/notebooks/demo_segmentation_us.py
"""

from __future__ import annotations

import os
import sys

# ---------------------------------------------------------------------------
# Ensure the repo root is on sys.path so both the `demos` package and the
# installed `ai2analytics` package are importable.
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ============================================================================
# Step 1 -- Generate synthetic US HCP data
# ============================================================================
print("\n" + "#" * 70)
print("# STEP 1: Generate synthetic US HCP data")
print("#" * 70)

from demos.synthetic_us_hcp import generate_all  # noqa: E402

dfs = generate_all()

# Let's peek at the data the pipeline will consume.
hcp_ref = dfs["hcp_reference"]
print(f"\nhcp_reference shape: {hcp_ref.shape}")
print(hcp_ref.head(10).to_string(index=False))

print("\nKey columns for segmentation:")
print(f"  Entity ID column:  npi  ({hcp_ref['npi'].nunique():,} unique providers)")
print(f"  Feature 1:         IL_17_TRX_L12M  (last-12-month IL-17 Rx count)")
print(f"  Feature 2:         IL_23_TRX_L12M  (last-12-month IL-23 Rx count)")

# ============================================================================
# Step 2 -- Configure the segmentation pipeline
# ============================================================================
print("\n" + "#" * 70)
print("# STEP 2: Configure the segmentation pipeline")
print("#" * 70)

from ai2analytics.templates.segmentation import (  # noqa: E402
    SegmentationConfig,
    SegmentationPipeline,
)

cfg = SegmentationConfig(
    analysis_name="us_hcp_segmentation",
    col_entity_id="npi",
    feature_columns=["IL_17_TRX_L12M", "IL_23_TRX_L12M"],
    n_segments=4,
    method="kmeans",
    output_csv="demos/data/seg_us_output.csv",
)

print(f"\nConfiguration:")
print(f"  analysis_name:     {cfg.analysis_name}")
print(f"  col_entity_id:     {cfg.col_entity_id}")
print(f"  feature_columns:   {cfg.feature_columns}")
print(f"  n_segments:        {cfg.n_segments}")
print(f"  method:            {cfg.method}")
print(f"  normalize:         {cfg.normalize}")
print(f"  handle_missing:    {cfg.handle_missing}")
print(f"  output_csv:        {cfg.output_csv}")

# ============================================================================
# Step 3 -- Run the pipeline
# ============================================================================
print("\n" + "#" * 70)
print("# STEP 3: Run the segmentation pipeline")
print("#" * 70)

print("\nNote: We pass the DataFrame directly via dataframes={'entity_data': ...}")
print("This avoids any Spark dependency.  The pipeline normalises features,")
print("fits KMeans with k=4, and writes a CSV with segment assignments.\n")

pipeline = SegmentationPipeline()
output = pipeline.run(cfg, dataframes={"entity_data": dfs["hcp_reference"]})

# ============================================================================
# Step 4 -- Inspect results
# ============================================================================
print("\n" + "#" * 70)
print("# STEP 4: Inspect results")
print("#" * 70)

# 4a. Assignments -- each entity mapped to a segment label
print("\n--- Segment Assignments (first 10 rows) ---")
print(output.assignments.head(10).to_string(index=False))

# 4b. Profiles -- mean feature values per segment
print("\n--- Segment Profiles (feature means) ---")
print(output.profiles.to_string(index=False))
print("\nInterpretation: each row is a segment.  The feature columns show")
print("the average IL-17 and IL-23 Rx counts for HCPs in that segment.")
print("Segments with higher means are heavier prescribers.")

# 4c. Summary statistics
print("\n--- Summary Statistics ---")
for key, val in output.summary_stats.items():
    print(f"  {key}: {val}")

# 4d. Segment size distribution
print("\n--- Segment Sizes ---")
sizes = output.summary_stats.get("segment_sizes", {})
total = sum(sizes.values())
for seg in sorted(sizes):
    pct = sizes[seg] / total * 100
    print(f"  Segment {seg}: {sizes[seg]:>5,} HCPs  ({pct:5.1f}%)")

# ============================================================================
# Step 5 -- Auto-select k
# ============================================================================
print("\n" + "#" * 70)
print("# STEP 5: Re-run with auto_select_k=True")
print("#" * 70)

print("\nInstead of fixing k=4, let the pipeline evaluate k=2..8 and pick")
print("the value that maximises the silhouette score.\n")

cfg_auto = SegmentationConfig(
    analysis_name="us_hcp_segmentation_auto_k",
    col_entity_id="npi",
    feature_columns=["IL_17_TRX_L12M", "IL_23_TRX_L12M"],
    n_segments=4,          # ignored when auto_select_k=True
    method="kmeans",
    auto_select_k=True,
    k_range=(2, 8),
    output_csv="demos/data/seg_us_auto_k_output.csv",
)

pipeline_auto = SegmentationPipeline()
output_auto = pipeline_auto.run(
    cfg_auto, dataframes={"entity_data": dfs["hcp_reference"]}
)

print("\n--- Auto-selected k results ---")
print(f"  Chosen k:       {output_auto.summary_stats['n_segments']}")
print(f"  Silhouette:     {output_auto.summary_stats['silhouette_score']:.4f}")
print(f"  Segment sizes:  {output_auto.summary_stats['segment_sizes']}")

print("\n--- Segment Profiles (auto-k) ---")
print(output_auto.profiles.to_string(index=False))

# ============================================================================
# Done
# ============================================================================
print("\n" + "#" * 70)
print("# DEMO COMPLETE")
print("#" * 70)
print(f"\nOutput files written to:")
print(f"  {os.path.abspath(cfg.output_csv)}")
print(f"  {os.path.abspath(cfg_auto.output_csv)}")
print("\nThis demo showed how to segment US HCPs by prescribing behaviour")
print("using the ai2analytics segmentation template with no Spark dependency.")
