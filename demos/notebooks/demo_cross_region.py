"""Cross-Region Segmentation Demo
===================================

The "star" demo -- shows that the **exact same** SegmentationPipeline
handles US HCP data and EU account data, then demonstrates how the
knowledge store captures both runs for future reference.

What this script covers:
    1. Run US HCP segmentation (npi, IL-17/IL-23 Rx features, KMeans k=4).
    2. Run EU account segmentation (PRESCRIBER_ID, units/tier, auto method).
    3. Compare the two runs side-by-side -- same pipeline, different schemas.
    4. Log both runs to a local JSON-backed DecisionStore.
    5. Query the store and show how accumulated knowledge could be used.

Run from the repository root:
    python demos/notebooks/demo_cross_region.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

# ============================================================================
# Step 1 -- Run US HCP segmentation
# ============================================================================
print("\n" + "=" * 70)
print("  PART A: US HCP Segmentation")
print("=" * 70)

from demos.synthetic_us_hcp import generate_all as gen_us  # noqa: E402
from ai2analytics.templates.segmentation import (  # noqa: E402
    SegmentationConfig,
    SegmentationPipeline,
)

us_dfs = gen_us()

us_cfg = SegmentationConfig(
    analysis_name="us_hcp_segmentation",
    col_entity_id="npi",
    feature_columns=["IL_17_TRX_L12M", "IL_23_TRX_L12M"],
    n_segments=4,
    method="kmeans",
    output_csv="demos/data/cross_region_us_output.csv",
)

us_pipeline = SegmentationPipeline()
us_output = us_pipeline.run(us_cfg, dataframes={"entity_data": us_dfs["hcp_reference"]})

print("\n--- US Results Summary ---")
print(f"  Entities:    {us_output.summary_stats['n_entities']:,}")
print(f"  Segments:    {us_output.summary_stats['n_segments']}")
print(f"  Method:      {us_output.summary_stats['method']}")
print(f"  Silhouette:  {us_output.summary_stats['silhouette_score']:.4f}")
print(f"  Profiles:")
print(us_output.profiles.to_string(index=False))

# ============================================================================
# Step 2 -- Run EU account segmentation
# ============================================================================
print("\n" + "=" * 70)
print("  PART B: EU Account Segmentation")
print("=" * 70)

from demos.synthetic_eu_account import generate_all as gen_eu  # noqa: E402

eu_dfs = gen_eu()

eu_cfg = SegmentationConfig(
    analysis_name="eu_account_segmentation",
    col_entity_id="PRESCRIBER_ID",
    feature_columns=["UNITS_SOLD_L12M", "TIER"],
    n_segments=3,
    method="auto",
    output_csv="demos/data/cross_region_eu_output.csv",
)

eu_pipeline = SegmentationPipeline()
eu_output = eu_pipeline.run(eu_cfg, dataframes={"entity_data": eu_dfs["account_reference"]})

print("\n--- EU Results Summary ---")
print(f"  Entities:    {eu_output.summary_stats['n_entities']:,}")
print(f"  Segments:    {eu_output.summary_stats['n_segments']}")
print(f"  Method:      {eu_output.summary_stats['method']}")
print(f"  Silhouette:  {eu_output.summary_stats['silhouette_score']:.4f}")
print(f"  Profiles:")
print(eu_output.profiles.to_string(index=False))

# ============================================================================
# Step 3 -- Side-by-side comparison
# ============================================================================
print("\n" + "=" * 70)
print("  PART C: Side-by-Side Comparison")
print("=" * 70)

print("""
+---------------------+----------------------------+----------------------------+
| Dimension           | US HCP                     | EU Account                 |
+---------------------+----------------------------+----------------------------+""")
print(f"| Entity ID column    | {'npi':<26s} | {'PRESCRIBER_ID':<26s} |")
print(f"| Entity count        | {us_output.summary_stats['n_entities']:<26,} | {eu_output.summary_stats['n_entities']:<26,} |")
print(f"| Features            | {'IL_17_TRX, IL_23_TRX':<26s} | {'UNITS_SOLD, TIER':<26s} |")
print(f"| Method              | {us_output.summary_stats['method']:<26s} | {eu_output.summary_stats['method']:<26s} |")
print(f"| Segments            | {us_output.summary_stats['n_segments']:<26} | {eu_output.summary_stats['n_segments']:<26} |")
print(f"| Silhouette          | {us_output.summary_stats['silhouette_score']:<26.4f} | {eu_output.summary_stats['silhouette_score']:<26.4f} |")
print(f"| Pipeline class      | {'SegmentationPipeline':<26s} | {'SegmentationPipeline':<26s} |")
print("+---------------------+----------------------------+----------------------------+")

print("""
Key takeaway: the SegmentationPipeline class is identical in both runs.
All schema differences are captured in SegmentationConfig.  This is
what makes the template reusable across regions, brands, and entity
types.
""")

# ============================================================================
# Step 4 -- Log both runs to the knowledge store
# ============================================================================
print("=" * 70)
print("  PART D: Knowledge Store Logging")
print("=" * 70)

from ai2analytics.knowledge import DecisionStore, DecisionRecord  # noqa: E402

# Use a temp file so the demo is self-contained and clean
store_path = os.path.join(tempfile.gettempdir(), "demo_cross_region_decisions.jsonl")
# Clean up from prior runs
if os.path.exists(store_path):
    os.remove(store_path)

store = DecisionStore(backend="json", path=store_path)

# Log the US run
us_record = DecisionRecord(
    template_name="segmentation",
    config_dict={
        "analysis_name": us_cfg.analysis_name,
        "col_entity_id": us_cfg.col_entity_id,
        "feature_columns": us_cfg.feature_columns,
        "n_segments": us_cfg.n_segments,
        "method": us_cfg.method,
    },
    data_profile="US HCP reference table, 5000 providers, IL-17/IL-23 Rx features",
    outcome_notes="Completed successfully",
    outcome_metrics={
        "silhouette_score": us_output.summary_stats["silhouette_score"],
        "n_entities": us_output.summary_stats["n_entities"],
        "n_segments": us_output.summary_stats["n_segments"],
    },
    tags=["us", "hcp", "segmentation"],
)
us_run_id = store.log(us_record)
print(f"\nLogged US run:  {us_run_id}")

# Log the EU run
eu_record = DecisionRecord(
    template_name="segmentation",
    config_dict={
        "analysis_name": eu_cfg.analysis_name,
        "col_entity_id": eu_cfg.col_entity_id,
        "feature_columns": eu_cfg.feature_columns,
        "n_segments": eu_cfg.n_segments,
        "method": eu_cfg.method,
    },
    data_profile="EU account reference table, 3000 accounts, UNITS_SOLD/TIER features",
    outcome_notes="Completed successfully, auto method selected " + eu_output.summary_stats["method"],
    outcome_metrics={
        "silhouette_score": eu_output.summary_stats["silhouette_score"],
        "n_entities": eu_output.summary_stats["n_entities"],
        "n_segments": eu_output.summary_stats["n_segments"],
    },
    tags=["eu", "account", "segmentation"],
)
eu_run_id = store.log(eu_record)
print(f"Logged EU run:  {eu_run_id}")

# ============================================================================
# Step 5 -- Query the store
# ============================================================================
print("\n" + "=" * 70)
print("  PART E: Querying the Knowledge Store")
print("=" * 70)

# Query all segmentation decisions
all_seg = store.query(template_name="segmentation")
print(f"\nTotal segmentation decisions logged: {len(all_seg)}")

for rec in all_seg:
    print(f"\n  Run {rec.run_id}  ({rec.timestamp})")
    print(f"    Analysis:     {rec.config_dict.get('analysis_name')}")
    print(f"    Entity ID:    {rec.config_dict.get('col_entity_id')}")
    print(f"    Features:     {rec.config_dict.get('feature_columns')}")
    print(f"    Method:       {rec.config_dict.get('method')}")
    print(f"    Silhouette:   {rec.outcome_metrics.get('silhouette_score', 'N/A')}")
    print(f"    Tags:         {rec.tags}")
    print(f"    Outcome:      {rec.outcome_notes}")

# Filter by tag
print("\n--- Query by tag 'eu' ---")
eu_results = store.query(tags=["eu"])
print(f"  Found {len(eu_results)} decision(s)")
for rec in eu_results:
    print(f"    {rec.run_id}: {rec.config_dict.get('analysis_name')}")

print("""
The knowledge store lets the system learn from past runs.  When a user
starts a new segmentation analysis, the retriever can inject these past
decisions into the LLM prompt, so it can suggest column mappings and
parameter choices based on what worked before.
""")

# Show the raw JSONL content
print("--- Raw JSONL content (the persistent store) ---")
with open(store_path, "r") as f:
    for i, line in enumerate(f, 1):
        data = json.loads(line)
        print(f"\n  Record {i}:")
        print(f"    run_id:     {data['run_id']}")
        print(f"    template:   {data['template_name']}")
        print(f"    config:     {json.dumps(data['config_dict'])}")
        print(f"    metrics:    {json.dumps(data['outcome_metrics'])}")

# ============================================================================
# Done
# ============================================================================
print("\n" + "=" * 70)
print("  DEMO COMPLETE")
print("=" * 70)
print(f"\nDecision store:  {store_path}")
print(f"US output:       {os.path.abspath(us_cfg.output_csv)}")
print(f"EU output:       {os.path.abspath(eu_cfg.output_csv)}")
print("\nThis demo showed that one pipeline handles multiple schemas and")
print("that the knowledge store captures run history for organisational")
print("learning.")
