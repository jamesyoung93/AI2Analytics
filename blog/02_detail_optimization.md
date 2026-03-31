# From Monolith to Module: Refactoring a 1,500-Line HCP Call Allocation Pipeline

There is a notebook somewhere in your organization that nobody wants to touch. It is 1,500 lines long. It loads seven data sources, engineers forty features, trains three models, runs a linear program, and writes output to four places. It works -- for one brand. It has been copied six times with minor edits. The copies have diverged. Nobody is sure which version is current.

This post describes how we refactored that notebook into a modular, configurable pipeline using the `ai2analytics` framework, and what we learned deploying it across brands.

## The Before: A Single Notebook

The original pipeline was a classic pharma analytics monolith. The column name `PAT_COUNT_REFERRED` appeared 23 times. The NPI column was sometimes `npi`, sometimes `NPI`, sometimes cast to int, sometimes left as string. Territory budget constraints were hardcoded as `100`. Every time someone needed to run it for a new brand, they did a find-and-replace for the drug name and hoped for the best.

The pipeline's logic was sound. The problem was that configuration and logic were welded together.

## The Key Insight: Separate Config from Logic

The refactoring started with a simple question: what actually changes between brands?

The answer: column names, file paths, model parameters, optimizer constraints, and drug-specific metadata. The stages themselves -- load, feature engineering, training, scoring, optimization, output -- are identical.

This led to a typed configuration dataclass that captures every brand-specific value:

```python
from ai2analytics.templates.detail_optimization import DetailOptimizationConfig

cfg = DetailOptimizationConfig(
    drug_name="BRAND_X",
    drug_portfolio="BRAND_Y",
    hcp_weekly_table="catalog.schema.hcp_weekly",
    calls_table="catalog.schema.detail_calls",
    team_a_align_path="/dbfs/mnt/data/team_a_alignment.csv",
    team_b_align_path="/dbfs/mnt/data/team_b_alignment.csv",
    hcp_reference_path="/dbfs/mnt/data/hcp_reference.csv",

    # Column mappings -- set these to match YOUR data
    col_npi="npi",
    col_week="WEEK_ENDING",
    col_referrals="PAT_COUNT_REFERRED",
    col_calls="HCP_F2F_CALLS",

    # Optimizer constraints
    team_a_budget_per_territory=100,
    team_b_target_per_territory=100,
    max_calls_nonpriority=4,

    output_csv="/dbfs/mnt/output/call_plan.csv",
)
```

The config has over 50 fields, covering everything from lag periods and rolling windows to model hyperparameters and optimizer penalties. Every field has a sensible default, so most configurations only need to specify the data-specific values.

## Pipeline Stages: A Walkthrough

With configuration externalized, each stage becomes a pure function: config in, data out.

### Stage C: Data Loading

The loader reads from Spark tables or CSV files based on config paths. It normalizes NPI columns to integers, handles missing values, and produces a `LoadedData` container with standardized DataFrames.

### Stage D: Feature Engineering

This is where the pipeline earns its keep. From raw HCP-weekly data and detail call records, it builds:

- **Cumulative sums** of referrals and calls per NPI
- **Rolling windows** (4-week and 12-week trailing sums)
- **Lag features** (1, 2, 3, and 4 period lags)
- **Moving averages** (4-week MA for calls and referrals)
- **Indication diversity** (unique indication count per HCP-week)
- **Binary referral labels** and first-write-week markers

The feature list is built dynamically: any numeric column with less than 10% missing values is included, minus an explicit exclusion set.

### The Treatment Variable: TS_CALLS_next

This is the design choice that makes the optimizer work. The feature `TS_CALLS_next` is a forward-looking sum of calls over the target horizon (default: 4 weeks). It is deliberately included as a model feature -- not excluded -- because it serves as the **treatment variable**.

The model learns the relationship between call intensity and referral outcomes from historical data. During scoring, the pipeline varies `TS_CALLS_next` across scenarios (0, 1, 2, 3, 4 calls) to predict what would happen at each hypothetical call level. The difference in predicted expected value between call levels is what the optimizer maximizes.

```python
# During scoring, each NPI is evaluated at every call level:
for scenario in cfg.scenario_range:  # [0, 1, 2, 3, 4]
    X_scenario = X.copy()
    X_scenario["TS_CALLS_next"] = scenario
    pred_prob = model_prob.predict_proba(X_scenario)[:, 1]
    pred_depth = model_depth.predict(X_scenario)
    ev = pred_prob * pred_depth  # Expected Value at this call level
```

### Stages E-F: Model Training and Scoring

Three models are trained with walk-forward backtesting:

1. **Probability model** (GradientBoosting classifier): Will this HCP refer a patient in the next 4 weeks?
2. **Depth model** (RandomForest regressor): If they refer, how many patients?
3. **Look-alike model** (GradientBoosting classifier): Among non-writers, who looks like a future writer?

Each model is backtested across 8 folds with an 8-week gap between training and evaluation windows, then retrained on the full dataset for production scoring.

### Stages G-H: The LP Optimizer

The heart of the pipeline is a PuLP linear program that allocates calls across NPIs subject to constraints:

- Each NPI must be assigned exactly one (Team A calls, Team B calls) pair
- Team A call totals per territory must hit a budget target
- Team B call totals per territory must hit a target with slack penalties
- Priority targets get different allowed call-pair sets
- The objective maximizes expected value plus a portfolio-drug decile bonus, minus big-M penalties for budget violations

```python
# Objective: maximize EV + decile bonus - slack penalties
prob_lp += obj_ev + obj_decile - obj_penalty
```

The optimizer runs in seconds for 5,000 NPIs and produces a call plan that balances clinical value against territory-level operational constraints.

### Stages I-J: Post-Processing and Output

The final stages build a long-format portfolio DataFrame, flag priority targets and look-alike HCPs, identify NPIs receiving call increases, and write to CSV and/or a Spark Delta table.

## Scaling: The Second Brand

With the refactored pipeline, deploying for a second brand requires zero code changes to the pipeline itself. You write a new config:

```python
cfg_brand_y = DetailOptimizationConfig(
    drug_name="BRAND_Y",
    drug_portfolio="BRAND_Z",
    hcp_weekly_table="catalog.brand_y.hcp_weekly",
    calls_table="catalog.brand_y.detail_calls",
    col_npi="PHYSICIAN_NPI",           # different column name
    col_referrals="REFERRAL_COUNT",     # different column name
    team_a_budget_per_territory=120,    # different constraint
    output_csv="/dbfs/mnt/output/brand_y_plan.csv",
    # ... other brand-specific paths
)

pipeline = DetailOptimizationPipeline()
results = pipeline.run(cfg_brand_y, spark=spark)
```

If the new brand's data uses different column names, the AI-assisted session can auto-detect the mappings and generate adapter code. But even without the AI layer, the config-driven design means you are filling out a form, not editing a notebook.

## Lessons Learned from Real Deployment

Several issues surfaced during validation that would have been silent bugs in the monolithic notebook:

**NPI type mismatches.** One data source stored NPIs as strings (`"1234567890"`), another as integers (`1234567890`). The merge silently produced zero matches. The fix was explicit type normalization in the loader, now applied consistently for every brand.

**Planning snapshot drops.** Adding a planning date filter to restrict training data caused NPI counts to drop unexpectedly -- some NPIs had all their records outside the planning window. This led to adding a dedicated sanity check stage that reports NPI attrition at each processing step.

**Territory alignment gaps.** Not every NPI appears in both Team A and Team B alignments. The original notebook silently dropped unaligned NPIs; the refactored pipeline fills missing territories with zero and lets the optimizer handle them explicitly.

**Config validation.** The typed config's `validate()` method catches errors before the pipeline runs -- missing file paths, invalid parameter ranges, empty output targets. In the monolithic notebook, these caused crashes 20 minutes into execution.

## The Payoff

A pipeline that used to require weeks of notebook surgery for each new brand now requires an afternoon of configuration. The pipeline logic is tested once, maintained in one place, and deployed everywhere. Every config is a self-documenting record of exactly how a brand was set up.

The shift from monolith to module is not glamorous. It is not a new algorithm or a novel model architecture. It is the engineering work that makes analytics scale -- and it is overdue in pharma.

---

**ai2analytics** is open source: [github.com/jamesyoung93/AI2Analytics](https://github.com/jamesyoung93/AI2Analytics)

Install with: `pip install git+https://github.com/jamesyoung93/AI2Analytics.git`
