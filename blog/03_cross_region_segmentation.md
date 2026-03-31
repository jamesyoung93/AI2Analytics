# Same Pipeline, Different Data: Segmentation from US HCP to EU Accounts

The request sounds simple: "We ran segmentation for the US team. Can we do the same thing for EU?"

"The same thing" turns out to mean: different entity identifiers (NPIs vs. account IDs), different time granularity (weekly vs. monthly), different feature sets (individual referrals vs. institutional unit sales), different column naming conventions, and a different number of natural segments. The methodology is identical -- cluster entities by behavior -- but the data surface is entirely different.

This is the cross-region scaling problem, and it is one of the most common sources of duplicated effort in pharma analytics. This post walks through how the `ai2analytics` segmentation pipeline handles both US and EU data with the same code and different configurations, and what we learned from the results.

## The Challenge: Two Data Worlds

Here is what US HCP data looks like:

- **Entity ID:** 10-digit NPI (National Provider Identifier), integer
- **Grain:** One row per NPI per week
- **Key features:** Patient referral counts, face-to-face call activity, symptom severity scores, indication diversity
- **Identifiers:** `npi`, `WEEK_ENDING`, `PAT_COUNT_REFERRED`

And here is the EU account data:

- **Entity ID:** `ACC-XXXXX` format string (institutional account)
- **Grain:** One row per account per month
- **Key features:** Units sold, new patient counts, market share, KAM visit frequency
- **Identifiers:** `PRESCRIBER_ID`, `MONTH_END`, `UNITS_SOLD`

Different IDs, different column names, different granularity, different business metrics. In a traditional analytics workflow, this means a second notebook.

## Two Configs, One Pipeline

With `ai2analytics`, the segmentation pipeline does not know or care whether it is processing US HCPs or EU accounts. It operates on a `SegmentationConfig` that maps external column names to internal expectations:

**US HCP Configuration:**

```python
from ai2analytics.templates.segmentation import (
    SegmentationConfig, SegmentationPipeline,
)

us_cfg = SegmentationConfig(
    analysis_name="us_hcp_segments",
    entity_table="commercial.hcp_features",
    col_entity_id="npi",
    feature_columns=[
        "referral_total", "calls_total", "severity_avg",
        "indication_count", "switch_rate",
    ],
    n_segments=4,
    method="kmeans",
    normalize=True,
    normalization_method="standard",
    output_csv="/dbfs/mnt/output/us_hcp_segments.csv",
)
```

**EU Account Configuration:**

```python
eu_cfg = SegmentationConfig(
    analysis_name="eu_account_segments",
    entity_table="eu_data.account_features",
    col_entity_id="PRESCRIBER_ID",
    feature_columns=[
        "units_sold_total", "new_patients_total",
        "market_share_avg", "kam_visits_total",
    ],
    n_segments=3,
    method="auto",      # let the pipeline decide
    normalize=True,
    normalization_method="robust",    # handles EU data's heavier outliers
    handle_missing="median",
    auto_select_k=True,
    k_range=(2, 8),
    output_csv="/dbfs/mnt/output/eu_account_segments.csv",
)
```

The pipeline call is identical in both cases:

```python
pipeline = SegmentationPipeline()

us_results = pipeline.run(us_cfg, spark=spark)
eu_results = pipeline.run(eu_cfg, spark=spark)
```

## The Results: What the Data Told Us

### US HCP Segmentation

- **5,000 NPIs** processed
- **4 segments** identified
- **Silhouette score: 0.61**

Four segments emerged with clinically interpretable profiles: high-volume referrers with active call engagement, moderate-volume referrers with limited calls, low-volume but responsive HCPs (the growth opportunity), and minimal-activity HCPs. The silhouette score of 0.61 indicates well-separated clusters with meaningful between-group differences.

### EU Account Segmentation

- **3,000 accounts** processed
- **3 segments** identified (auto-selected)
- **Silhouette score: 0.57**
- **Method: KMeans** (auto-selected over hierarchical)

The EU run used `method='auto'`, which tries both KMeans and hierarchical clustering and selects the method with the higher silhouette score. For this dataset, KMeans won:

```
  Method: auto (trying both kmeans and hierarchical)
  KMeans       (k=3): silhouette=0.5700
  Hierarchical (k=3): silhouette=0.5423
  Selected: kmeans
```

The slightly lower silhouette score (0.57 vs. 0.61) reflects the structural differences in EU data: account-level aggregation smooths out some of the individual variation that creates tighter clusters in HCP-level data. Three segments is the natural structure here -- forcing four would have degraded the score.

## The Auto-Selection Feature

The `method='auto'` option is worth explaining. When you do not have a strong prior about whether KMeans or hierarchical clustering is more appropriate for your data, auto mode runs both and compares:

```python
# Inside fit_segments() when method='auto':
labels_km, centers_km = _fit_kmeans(matrix, k)
sil_km = silhouette_score(matrix, labels_km)

labels_hc = _fit_hierarchical(matrix, k)
sil_hc = silhouette_score(matrix, labels_hc)

if sil_km >= sil_hc:
    result.method_used = "kmeans"
else:
    result.method_used = "hierarchical"
```

Combined with `auto_select_k=True`, the pipeline will sweep a range of k values for each method, select the best k per method, then compare the winners. This is especially useful for cross-region deployments where you cannot assume the same number of segments exists in every market.

For the EU data, the auto-selection tried k=2 through k=8 and found that k=3 produced the highest silhouette score for both methods, with KMeans slightly outperforming hierarchical. This automatic tuning eliminates a common source of analyst bias -- choosing k based on what worked for a different dataset.

## How the AI Discovery Layer Handles This

If you are using the AI-assisted workflow via `AnalyticsSession`, the cross-region transition becomes even simpler. When the session discovers EU data for the first time, the LLM sees columns like `PRESCRIBER_ID`, `MONTH_END`, and `UNITS_SOLD`. It reads the template's declared requirements:

```
Required columns:
  - entity_id (string): Unique entity identifier
    (aliases: id, npi, customer_id, hcp_id) -> cfg.col_entity_id
```

The LLM matches `PRESCRIBER_ID` to the `entity_id` requirement (it is semantically an entity identifier, even though it does not match any alias exactly) and sets `col_entity_id = "PRESCRIBER_ID"` in the auto-config.

If the US run was logged in the knowledge store, the EU setup benefits from that context. The retriever injects past decisions into the LLM prompt:

```
PAST COLUMN MAPPINGS:
  Run a3f2c1 (segmentation):
    Auto-detected: {"col_entity_id": "npi", "entity_table": "commercial.hcp_features"}
    User-provided: {"n_segments": 4, "method": "kmeans"}
    Outcome: silhouette=0.61, 4 segments, 5000 entities
```

The LLM sees that the US run used `npi` as the entity ID, but it also sees that the EU data does not have an `npi` column -- so it correctly selects `PRESCRIBER_ID` instead. It does not blindly copy the US config. It uses the US run as context for understanding the pattern, then adapts to the new data.

## The Knowledge Angle

After both runs complete, the knowledge store contains two decision records. If someone later needs to run segmentation for a BRIC market -- say, with `ACCOUNT_CODE` as the entity ID and quarterly data -- the system has two prior examples to draw from. The conversation manager can say with confidence, "In past runs, the entity ID column was mapped from the data's primary identifier column. Your data has `ACCOUNT_CODE` -- should I use that?"

The number of questions decreases with each deployment. The first run might need 10 answered questions. The second needs 5. By the fifth, most configs are auto-detected from the accumulated context.

## What Did Not Change

It is worth emphasizing what stayed constant between the US and EU deployments:

- The `SegmentationPipeline` class -- zero modifications
- The feature preparation logic (normalization, imputation, optional PCA)
- The clustering algorithms (KMeans, hierarchical, auto-selection)
- The silhouette scoring and output formatting
- The Spark integration and output writing

This is the payoff of the config-driven architecture. The pipeline is a stable, tested artifact. The variability lives entirely in the configuration, where it is explicit, validated, and logged.

## Getting Started

If you are maintaining separate segmentation notebooks for different regions or brands, you are carrying unnecessary risk. Every copied notebook is a liability -- a place where a bug fix in one version never reaches the others, where a column rename causes a silent merge failure, where the analyst who set it up is the only person who understands the choices.

A single pipeline with per-deployment configuration eliminates this class of problems. The methodology is proven. The data differences are a configuration problem. Treat them as one.

---

**ai2analytics** is open source: [github.com/jamesyoung93/AI2Analytics](https://github.com/jamesyoung93/AI2Analytics)

Install with: `pip install git+https://github.com/jamesyoung93/AI2Analytics.git`
