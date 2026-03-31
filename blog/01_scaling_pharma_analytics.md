# The Last-Mile Problem in Pharma Analytics -- And How AI Can Solve It

Every analytics team in pharma has experienced this: you spend weeks building a pipeline that works beautifully for one brand. The models are validated, the outputs are clean, stakeholders are happy. Then someone asks, "Can we run this for Brand Y?"

You clone the notebook. You start renaming columns. You discover the new data source uses `PHYSICIAN_NPI` instead of `npi`, monthly granularity instead of weekly, and account-level identifiers instead of individual prescriber IDs. Three weeks later, you have a second notebook that mostly works, but shares zero code with the first one. And now you have two notebooks to maintain.

This is the last-mile problem in pharma analytics: the methodology is portable, but the data never is.

## The Pattern Behind the Pain

If you look across the pipelines that analytics teams build -- HCP segmentation, call allocation optimization, market mix modeling, next-best-action engines -- the underlying logic is remarkably stable. KMeans clustering does not care whether your entity identifier is called `npi` or `PRESCRIBER_ID`. A Ridge regression does not care whether your time series is weekly or monthly.

What changes between deployments is the surface layer:

- **Column names.** One brand's data uses `PAT_COUNT_REFERRED`, another uses `REFERRAL_COUNT`.
- **Entity granularity.** US pipelines operate at the individual HCP level (NPIs). EU pipelines operate at the account level (institution identifiers).
- **Time granularity.** Some data arrives weekly, some monthly.
- **Reference files.** Territory alignments, decile scores, and priority target lists all have different schemas across business units.
- **Therapeutic class variables.** The Rx columns that define decile targets change with every drug.

Every one of these differences is a configuration problem, not an algorithm problem. But when your pipeline is a single notebook with hardcoded column names scattered across 1,500 lines, configuration and logic are inextricable.

## The Architectural Insight: Separate What from How

The `ai2analytics` framework is built on a single architectural principle: **separate what the pipeline does from what the data looks like.**

This separation has five layers:

### 1. Templates

A template defines the pipeline logic as modular, reusable stages. Each stage accepts a typed configuration object and operates on standardized internal column names. The template also declares its data requirements -- what tables it needs, what columns those tables must have, and what config fields control the mapping.

```python
from ai2analytics.templates.segmentation import (
    SegmentationConfig, SegmentationPipeline,
)

cfg = SegmentationConfig(
    analysis_name="us_hcp_segments",
    entity_table="commercial.hcp_features",
    col_entity_id="npi",
    n_segments=4,
    method="kmeans",
    output_csv="/dbfs/mnt/output/segments.csv",
)

pipeline = SegmentationPipeline()
results = pipeline.run(cfg, spark=spark)
```

The same `SegmentationPipeline` class runs US HCP segmentation and EU account segmentation. The config is the only thing that changes.

### 2. Discovery

When you face a new data environment, the discovery layer scans your Spark catalog, profiles every table (column types, cardinality, time series characteristics), and produces a structured summary.

```python
from ai2analytics import AnalyticsSession

session = AnalyticsSession(spark=spark, llm_endpoint="your-endpoint")
session.discover(
    schemas=["commercial_data"],
    prompt="Segment HCPs by prescribing behavior",
)
```

### 3. Conversation

An LLM reads the data profile and the template's declared requirements side by side. It auto-fills config fields it can infer (`col_npi = 'PHYSICIAN_NPI'` when it sees that column in the data) and generates targeted questions for what it cannot resolve. This is not open-ended chat -- it is structured Q&A where every question maps to a specific config field.

```python
session.answer({
    "entity_table": "commercial.physician_features",
    "col_entity_id": "PHYSICIAN_NPI",
    "output_csv": "/dbfs/mnt/output/brand_y_segments.csv",
})
```

### 4. Adapter Generation

When the source data does not match the template schema -- different column names, type mismatches, structural differences -- the LLM generates PySpark transformation code to bridge the gap. If the generated code fails at runtime, the error traceback is fed back to the LLM for automatic correction.

```python
session.generate_adapter()
session.run_adapter()  # auto-retries with LLM fix on failure
```

### 5. Knowledge Accumulation

Every pipeline run is logged: config values, data profiles, user decisions, adapter code, and outcome metrics. This history feeds back into future LLM prompts, so the system learns from past deployments.

## The Value Proposition: Diminishing Marginal Effort

Here is what this architecture delivers in practice:

- **First brand:** Days. You need to understand the data, configure the pipeline, possibly write adapter code, and validate results.
- **Second brand:** Hours. The template is proven, the knowledge store has one successful run to reference, and the LLM has seen your data conventions before.
- **Tenth brand:** Minutes. The system has accumulated enough context that nearly every config field is auto-detected. The conversation is three questions instead of twenty.

We validated this with three distinct pipelines:

- **US HCP Segmentation:** 5,000 NPIs segmented into 4 clusters (silhouette score 0.61)
- **EU Account Segmentation:** 3,000 accounts segmented into 3 clusters (silhouette score 0.57), with the framework automatically selecting KMeans over hierarchical clustering
- **Market Mix Model:** 104 weeks of media data, R-squared of 0.64, correctly identifying all 3 media channels and their relative ROI contributions

Three pipelines, three different data schemas, three different analytical objectives -- all running through the same framework.

## The Organizational Learning Loop

The most underappreciated aspect of this architecture is what happens at the organizational level. In most analytics teams, knowledge about how to set up a pipeline for a specific brand or region lives in one person's head. When that person leaves, the next analyst starts from scratch.

The knowledge store changes this dynamic. Every successful configuration becomes a reusable artifact. When the system encounters EU account data for the second time, it already knows that `PRESCRIBER_ID` is the entity identifier and that monthly granularity is standard. When it encounters a new therapeutic area, it draws on patterns from past adapter code.

This is not just automation. It is organizational memory for analytics.

## Getting Started

The framework is open source and designed for Databricks environments, though the pipeline templates run anywhere with pandas and scikit-learn.

If you are building the same analytical pipeline for the third time, you are solving the wrong problem. The methodology is not what needs to scale -- the configuration does.

---

**ai2analytics** is open source: [github.com/jamesyoung93/AI2Analytics](https://github.com/jamesyoung93/AI2Analytics)

Install with: `pip install git+https://github.com/jamesyoung93/AI2Analytics.git`
