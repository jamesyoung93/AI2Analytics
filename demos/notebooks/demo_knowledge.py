"""Knowledge Accumulation Demo
================================

Demonstrates the ai2analytics knowledge module: how pipeline decisions
and learned context accumulate over time, enabling the system to get
smarter with each run.

What this script covers:
    1. Creating a local JSON-backed DecisionStore and ContextStore.
    2. Simulating 3 pipeline runs with different configurations.
    3. Querying past decisions by template name and tags.
    4. Adding manual context entries (synthesised knowledge).
    5. Using the KnowledgeRetriever to format stored knowledge for
       injection into LLM prompts.

No Spark, no LLM API calls -- everything runs locally.

Run from the repository root:
    python demos/notebooks/demo_knowledge.py
"""

from __future__ import annotations

import json
import os
import sys
import tempfile
import time

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from ai2analytics.knowledge import (  # noqa: E402
    DecisionStore,
    DecisionRecord,
    ContextStore,
    ContextEntry,
    KnowledgeRetriever,
)

# ============================================================================
# Step 1 -- Create stores
# ============================================================================
print("\n" + "#" * 70)
print("# STEP 1: Create JSON-backed decision and context stores")
print("#" * 70)

tmpdir = tempfile.mkdtemp(prefix="demo_knowledge_")
decision_path = os.path.join(tmpdir, "decisions.jsonl")
context_path = os.path.join(tmpdir, "context.jsonl")

decision_store = DecisionStore(backend="json", path=decision_path)
context_store = ContextStore(backend="json", path=context_path)

print(f"\nDecision store: {decision_path}")
print(f"Context store:  {context_path}")
print("\nBoth use append-only JSONL files -- no database needed.")
print("In production, switch to backend='delta' for Spark Delta tables.")

# ============================================================================
# Step 2 -- Simulate 3 pipeline runs
# ============================================================================
print("\n" + "#" * 70)
print("# STEP 2: Simulate 3 pipeline runs")
print("#" * 70)

# --- Run 1: US HCP segmentation ---
print("\n--- Simulated Run 1: US HCP Segmentation ---")
run1_id = decision_store.log(DecisionRecord(
    template_name="segmentation",
    config_dict={
        "analysis_name": "us_hcp_q4_2025",
        "col_entity_id": "npi",
        "feature_columns": ["IL_17_TRX_L12M", "IL_23_TRX_L12M"],
        "n_segments": 4,
        "method": "kmeans",
    },
    data_profile="US HCP reference, 5000 rows, columns: npi, WRITER_FLAG, TARGET_FLAG, IL_17_TRX_L12M, IL_23_TRX_L12M",
    user_answers={"n_segments": 4, "method": "kmeans"},
    auto_detected={"col_entity_id": "npi", "feature_columns": ["IL_17_TRX_L12M", "IL_23_TRX_L12M"]},
    outcome_notes="Good separation. Segment 0 = low prescribers, Segment 3 = heavy writers.",
    outcome_metrics={
        "silhouette_score": 0.4821,
        "n_entities": 5000,
        "n_segments": 4,
    },
    tags=["us", "hcp", "segmentation", "q4-2025"],
))
print(f"  Logged with run_id: {run1_id}")

# Brief pause so timestamps differ
time.sleep(0.05)

# --- Run 2: EU Account segmentation ---
print("\n--- Simulated Run 2: EU Account Segmentation ---")
run2_id = decision_store.log(DecisionRecord(
    template_name="segmentation",
    config_dict={
        "analysis_name": "eu_account_q4_2025",
        "col_entity_id": "PRESCRIBER_ID",
        "feature_columns": ["UNITS_SOLD_L12M", "TIER"],
        "n_segments": 3,
        "method": "auto",
    },
    data_profile="EU account reference, 3000 rows, columns: PRESCRIBER_ID, ACCOUNT_NAME, REGION, SEGMENT, TIER, UNITS_SOLD_L12M",
    user_answers={"n_segments": 3},
    auto_detected={"col_entity_id": "PRESCRIBER_ID", "method": "auto"},
    outcome_notes="Auto selected kmeans. TIER as a feature creates ordinal clusters.",
    outcome_metrics={
        "silhouette_score": 0.5134,
        "n_entities": 3000,
        "n_segments": 3,
    },
    tags=["eu", "account", "segmentation", "q4-2025"],
))
print(f"  Logged with run_id: {run2_id}")

time.sleep(0.05)

# --- Run 3: Market Mix Model ---
print("\n--- Simulated Run 3: Market Mix Model ---")
run3_id = decision_store.log(DecisionRecord(
    template_name="market_mix",
    config_dict={
        "analysis_name": "brand_x_mmm_2025",
        "col_date": "WEEK_ENDING",
        "col_response": "SALES",
        "media_columns": ["TV_SPEND", "DIGITAL_SPEND", "PRINT_SPEND"],
        "control_columns": ["PRICE_INDEX", "DISTRIBUTION_PCT"],
        "model_type": "ridge",
    },
    data_profile="Weekly time-series, 104 rows, 7 columns",
    user_answers={"media_columns": ["TV_SPEND", "DIGITAL_SPEND", "PRINT_SPEND"]},
    auto_detected={"col_date": "WEEK_ENDING", "col_response": "SALES"},
    outcome_notes="R-squared 0.87. TV highest ROI. Print marginal.",
    outcome_metrics={
        "r_squared": 0.8700,
        "mape": 4.5,
        "best_roi_channel": "TV_SPEND",
    },
    tags=["mmm", "brand-x", "q4-2025"],
))
print(f"  Logged with run_id: {run3_id}")

# ============================================================================
# Step 3 -- Query past decisions
# ============================================================================
print("\n" + "#" * 70)
print("# STEP 3: Query past decisions")
print("#" * 70)

# Query all
print("\n--- All decisions (most recent first) ---")
all_decisions = decision_store.query(limit=10)
for d in all_decisions:
    print(f"  [{d.run_id}]  {d.template_name:<14s}  {d.config_dict.get('analysis_name', '')}")
    print(f"               Outcome: {d.outcome_notes}")

# Filter by template
print("\n--- Segmentation decisions only ---")
seg_decisions = decision_store.query(template_name="segmentation")
print(f"  Found {len(seg_decisions)} segmentation run(s)")
for d in seg_decisions:
    metrics = d.outcome_metrics
    print(f"  [{d.run_id}]  {d.config_dict.get('analysis_name', '')}")
    print(f"               Entity ID: {d.config_dict.get('col_entity_id')}")
    print(f"               Silhouette: {metrics.get('silhouette_score', 'N/A')}")

# Filter by tags
print("\n--- Decisions tagged 'eu' ---")
eu_decisions = decision_store.query(tags=["eu"])
print(f"  Found {len(eu_decisions)} EU run(s)")
for d in eu_decisions:
    print(f"  [{d.run_id}]  {d.config_dict.get('analysis_name', '')}  tags={d.tags}")

# Filter by tags -- cross-template
print("\n--- Decisions tagged 'q4-2025' (any template) ---")
q4_decisions = decision_store.query(tags=["q4-2025"])
print(f"  Found {len(q4_decisions)} Q4-2025 run(s)")
for d in q4_decisions:
    print(f"  [{d.run_id}]  {d.template_name}  {d.config_dict.get('analysis_name', '')}")

# ============================================================================
# Step 4 -- Add manual context entries
# ============================================================================
print("\n" + "#" * 70)
print("# STEP 4: Add manual context entries")
print("#" * 70)

print("\nContext entries represent synthesised knowledge -- patterns and")
print("best practices extracted from past decisions.  Normally an LLM")
print("generates these, but they can also be added manually.\n")

ctx1_id = context_store.add(ContextEntry(
    scope={"region": "us", "domain": "pharma"},
    category="column_mapping",
    title="US pharma HCP tables use 'npi' as entity ID",
    content=(
        "In US pharma datasets, the standard HCP identifier column is 'npi' "
        "(National Provider Identifier). It is always a 10-digit integer. "
        "When the segmentation template encounters a US HCP table, "
        "col_entity_id should default to 'npi'."
    ),
    template_name="segmentation",
    confidence=0.95,
    source_run_ids=[run1_id],
))
print(f"  Added context: {ctx1_id}  (US NPI mapping pattern)")

ctx2_id = context_store.add(ContextEntry(
    scope={"region": "eu", "domain": "pharma"},
    category="column_mapping",
    title="EU pharma account tables use 'PRESCRIBER_ID'",
    content=(
        "European pharma account tables use 'PRESCRIBER_ID' (format ACC-XXXXX) "
        "instead of NPI. The segmentation template should map col_entity_id to "
        "'PRESCRIBER_ID' when the data contains this column pattern."
    ),
    template_name="segmentation",
    confidence=0.90,
    source_run_ids=[run2_id],
))
print(f"  Added context: {ctx2_id}  (EU PRESCRIBER_ID mapping pattern)")

ctx3_id = context_store.add(ContextEntry(
    scope={"domain": "pharma"},
    category="config_preference",
    title="Auto method selection often outperforms fixed kmeans",
    content=(
        "When the user does not have a strong preference, method='auto' tends to "
        "produce better silhouette scores because it evaluates both kmeans and "
        "hierarchical clustering and picks the winner. Recommended as default "
        "for new analyses."
    ),
    template_name="segmentation",
    confidence=0.75,
    source_run_ids=[run1_id, run2_id],
))
print(f"  Added context: {ctx3_id}  (auto method preference)")

ctx4_id = context_store.add(ContextEntry(
    scope={"domain": "pharma"},
    category="data_quality",
    title="TIER columns are ordinal -- consider encoding",
    content=(
        "When TIER (1-4) is used as a clustering feature, the pipeline treats "
        "it as numeric. This works but creates ordinal distance effects. For "
        "more nuanced segmentation, consider one-hot encoding TIER or using "
        "a dedicated ordinal distance metric."
    ),
    template_name="segmentation",
    confidence=0.65,
    source_run_ids=[run2_id],
))
print(f"  Added context: {ctx4_id}  (TIER data quality note)")

ctx5_id = context_store.add(ContextEntry(
    scope={"domain": "pharma"},
    category="adapter_pattern",
    title="MMM weekly data needs WEEK_ENDING as datetime",
    content=(
        "The market mix pipeline expects the date column to be a proper datetime. "
        "If the source data has string dates, the adapter should include "
        "pd.to_datetime() conversion. The column is typically named WEEK_ENDING "
        "or WEEK_DATE in pharma datasets."
    ),
    template_name="market_mix",
    confidence=0.85,
    source_run_ids=[run3_id],
))
print(f"  Added context: {ctx5_id}  (MMM date adapter pattern)")

# ============================================================================
# Step 5 -- Query context entries
# ============================================================================
print("\n" + "#" * 70)
print("# STEP 5: Query context entries")
print("#" * 70)

# All context for segmentation
print("\n--- All segmentation context ---")
seg_context = context_store.query(template_name="segmentation")
for e in seg_context:
    print(f"  [{e.category}]  {e.title}  (confidence={e.confidence:.0%})")

# Filter by scope
print("\n--- Context scoped to region=us ---")
us_context = context_store.query(scope={"region": "us"})
for e in us_context:
    print(f"  [{e.category}]  {e.title}")
    print(f"    {e.content[:80]}...")

# Filter by category
print("\n--- Column mapping patterns only ---")
mapping_context = context_store.query(category="column_mapping")
for e in mapping_context:
    print(f"  {e.title}  (scope={e.scope})")

# ============================================================================
# Step 6 -- Format knowledge for LLM prompt
# ============================================================================
print("\n" + "#" * 70)
print("# STEP 6: Format knowledge for LLM prompt injection")
print("#" * 70)

print("\nThe KnowledgeRetriever combines decisions and context into a")
print("structured block that can be prepended to an LLM prompt.\n")

retriever = KnowledgeRetriever(
    decision_store=decision_store,
    context_store=context_store,
    max_decisions=5,
    max_context=5,
)

# General knowledge retrieval
print("--- General knowledge (all templates) ---")
general_knowledge = retriever.retrieve()
print(general_knowledge)

# Template-specific retrieval
print("\n--- Segmentation-specific knowledge ---")
seg_knowledge = retriever.retrieve(template_name="segmentation")
print(seg_knowledge)

# Analysis-focused retrieval (for column mapping)
print("\n--- Analysis knowledge (column mapping focus) ---")
analysis_knowledge = retriever.retrieve_for_analysis(
    template_name="segmentation",
    scope={"domain": "pharma"},
)
print(analysis_knowledge)

# Adapter-focused retrieval (for code generation)
print("\n--- Adapter knowledge (code generation focus) ---")
adapter_knowledge = retriever.retrieve_for_adapter(
    template_name="market_mix",
)
print(adapter_knowledge)

print("""
How this would be used in production:
  1. User starts a new segmentation analysis.
  2. The system calls retriever.retrieve_for_analysis(template_name="segmentation").
  3. The returned text block is injected into the LLM system prompt.
  4. The LLM sees past column mappings, learns that US data uses 'npi'
     and EU data uses 'PRESCRIBER_ID', and can auto-suggest the right
     mapping for the user's new dataset.
  5. After the run completes, a new DecisionRecord is logged.
  6. Over time, the ContextStore accumulates patterns (via LLM extraction
     or manual curation), making the system progressively smarter.
""")

# ============================================================================
# Step 7 -- Inspect raw store files
# ============================================================================
print("#" * 70)
print("# STEP 7: Raw store files")
print("#" * 70)

print(f"\nDecision store ({decision_path}):")
with open(decision_path, "r") as f:
    for i, line in enumerate(f, 1):
        data = json.loads(line)
        print(f"  Record {i}: run_id={data['run_id']}, template={data['template_name']}, "
              f"analysis={data['config_dict'].get('analysis_name', 'N/A')}")

print(f"\nContext store ({context_path}):")
with open(context_path, "r") as f:
    for i, line in enumerate(f, 1):
        data = json.loads(line)
        print(f"  Entry {i}: [{data['category']}] {data['title']} "
              f"(confidence={data['confidence']})")

# ============================================================================
# Done
# ============================================================================
print("\n" + "#" * 70)
print("# DEMO COMPLETE")
print("#" * 70)
print(f"\nStore files in: {tmpdir}")
print("\nThis demo showed how DecisionStore, ContextStore, and")
print("KnowledgeRetriever work together to accumulate and surface")
print("organisational knowledge across pipeline runs.")
