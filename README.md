# AI2Analytics

AI-powered framework for scaling analytical pipelines across brands, regions, and data sources. Built for teams that need to deploy the same modeling approach repeatedly, but face "last-mile" data differences each time.

## The Problem

You have a validated analytical pipeline — say, a call allocation optimizer or a propensity scoring engine. It works for one brand. Now you need to run it for five more, each with slightly different data schemas, column names, therapeutic class variables, and reference files.

Manually copying and editing a 1,500-line notebook for each brand is slow, error-prone, and impossible to maintain.

## The Solution

AI2Analytics separates **what the pipeline does** from **what the data looks like**:

- **Templates** define pipeline logic as modular, reusable stages with declared data requirements
- **Discovery** surveys your data catalog, profiles tables, and matches them to template requirements
- **Conversation** uses an LLM to ask structured questions about gaps it can't resolve automatically
- **Code generation** produces adapter/preprocessing code when source data doesn't match the template schema
- **Execution** runs the configured pipeline end-to-end

## Installation

```bash
# From GitHub
pip install git+https://github.com/jamesyoung93/AI2Analytics.git

# With Databricks dependencies
pip install "ai2analytics[databricks] @ git+https://github.com/jamesyoung93/AI2Analytics.git"

# For development
git clone https://github.com/jamesyoung93/AI2Analytics.git
cd AI2Analytics
pip install -e ".[dev]"
```

## Quick Start

### Direct usage (you know your data)

```python
from ai2analytics.templates.detail_optimization import (
    DetailOptimizationConfig,
    DetailOptimizationPipeline,
)

cfg = DetailOptimizationConfig(
    drug_name="BRAND_X",
    drug_portfolio="BRAND_Y",
    hcp_weekly_table="catalog.schema.hcp_weekly",
    calls_table="catalog.schema.calls",
    team_a_align_path="/dbfs/mnt/data/team_a_alignment.csv",
    team_b_align_path="/dbfs/mnt/data/team_b_alignment.csv",
    hcp_reference_path="/dbfs/mnt/data/hcp_reference.csv",
    output_csv="/dbfs/mnt/output/call_plan.csv",
    output_table="catalog.schema.call_plan_output",
)

pipeline = DetailOptimizationPipeline()
results = pipeline.run(cfg, spark=spark)
```

### AI-assisted usage (new brand, unfamiliar data)

```python
from ai2analytics import AnalyticsSession

session = AnalyticsSession(
    spark=spark,
    llm_endpoint="your-llm-endpoint",
)

# 1. Discover available data
session.discover(
    schemas=["commercial_data", "reference_tables"],
    prompt="Optimize HCP call allocation for a new brand",
)

# 2. Review auto-detected config and answer remaining questions
session.show_questions()
session.answer({
    "drug_name": "BRAND_X",
    "hcp_weekly_table": "commercial_data.hcp_weekly_brand_x",
    "output_csv": "/dbfs/mnt/output/brand_x_plan.csv",
})

# 3. Generate adapter code if data needs transformation
session.generate_adapter()

# 4. Run
results = session.run()
```

### Deep-profiling a table

```python
session.profile_table("catalog.schema.hcp_weekly")
```

Output includes: time series frequency, completeness/gaps, grain (entity x time), which columns vary over time vs. static, categorical vs. numeric classification.

## Architecture

```
ai2analytics/
├── session.py               # Main orchestrator
├── llm.py                   # LLM client (Databricks Model Serving / OpenAI-compatible)
├── utils.py                 # Shared data utilities
├── discovery/
│   ├── surveyor.py          # Scans Spark catalogs, profiles table schemas
│   └── profiler.py          # Deep analysis: time series, completeness, variable types
├── conversation/
│   └── manager.py           # LLM-driven Q&A to fill configuration gaps
├── codegen/
│   └── adapter.py           # Generates preprocessing code for data mismatches
└── templates/
    ├── base.py              # BaseTemplate, TableRequirement, ColumnRequirement
    ├── registry.py          # Template discovery and registration
    └── detail_optimization/ # First template implementation
        ├── config.py        # Typed configuration (dataclass with validation)
        ├── loader.py        # Stage C: Data loading
        ├── features.py      # Stage D: Feature engineering
        ├── models.py        # Stage E: Model training with backtesting
        ├── scoring.py       # Stage F: Vectorized scenario scoring
        ├── optimizer.py     # Stage G-H: PuLP LP call allocation
        ├── output.py        # Stage I-J: Post-processing and output
        └── pipeline.py      # Pipeline orchestrator
```

## Detail Optimization Template

The included `detail_optimization` template implements an HCP call allocation pipeline:

| Stage | Module | What it does |
|-------|--------|-------------|
| C | `loader.py` | Loads HCP weekly data, calls, alignments, reference files |
| D | `features.py` | Lags, rolling windows, cumulative sums, target construction |
| E | `models.py` | Trains probability, depth, and look-alike models with walk-forward backtesting |
| F | `scoring.py` | Scores every HCP at each call level (vectorized batch prediction) |
| G-H | `optimizer.py` | Merges alignments, runs PuLP LP with territory budget constraints |
| I-J | `output.py` | Builds long-format portfolio, flags (priority, lookalike, call increase), writes output |

### Configuration

All column names, file paths, model parameters, and optimizer constraints are configurable via `DetailOptimizationConfig`. No hardcoded assumptions about your specific data schema:

```python
cfg = DetailOptimizationConfig(
    # Column mappings
    col_npi="npi",
    col_week="WEEK_ENDING",
    col_referrals="PAT_COUNT_REFERRED",
    col_calls="HCP_F2F_CALLS",

    # Team labels
    team_a_label="SALES_TEAM",
    team_b_label="MEDICAL_TEAM",

    # Therapeutic class Rx columns for deciling
    il_rx_columns=[
        ("CLASS_A_TRX_L12M", "CLASS_A_DECILE"),
        ("CLASS_B_TRX_L12M", "CLASS_B_DECILE"),
    ],

    # Model parameters
    prob_model_params={"n_estimators": 200, "max_depth": 7, "random_state": 42},

    # Optimizer constraints
    team_a_budget_per_territory=120,
    team_b_target_per_territory=80,
    max_calls_nonpriority=4,
)
```

## Adding New Templates

Create a new template by subclassing `BaseTemplate`:

```python
from ai2analytics.templates.base import BaseTemplate, TableRequirement, ColumnRequirement
from ai2analytics.templates.registry import register

@register
class MyPipeline(BaseTemplate):
    name = "my_pipeline"
    description = "Description for template matching"
    config_class = MyConfig

    required_tables = [
        TableRequirement(
            key="main_table",
            description="Primary data source",
            required_columns=[
                ColumnRequirement("id", "int", "Entity identifier"),
                ColumnRequirement("date", "date", "Time period"),
            ],
        ),
    ]

    def run(self, config, spark=None):
        # Your pipeline logic here
        ...
```

The discovery layer will automatically match available data against your declared `required_tables` and generate configuration questions for anything it can't resolve.

## Requirements

- Python >= 3.9
- pandas, numpy, scikit-learn, PuLP, matplotlib
- Optional: databricks-sdk, pyspark (for Databricks environments)

## License

MIT
