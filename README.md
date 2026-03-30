# AI2Analytics

AI-powered framework for scaling analytical pipelines across brands, regions, and data sources. Built for teams that need to deploy the same modeling approach repeatedly, but face "last-mile" data differences each time.

## The Problem

You have a validated analytical pipeline — say, a call allocation optimizer or a propensity scoring engine. It works for one brand. Now you need to run it for five more, each with slightly different data schemas, column names, therapeutic class variables, and reference files.

Manually copying and editing a 1,500-line notebook for each brand is slow, error-prone, and impossible to maintain.

## The Solution

AI2Analytics separates **what the pipeline does** from **what the data looks like**:

1. **Templates** define pipeline logic as modular, reusable stages with declared data requirements
2. **Discovery** surveys your data catalog, profiles tables, and matches them to template requirements
3. **Conversation** uses an LLM to map discovered data to config fields, auto-filling what it can and asking structured questions about what it can't
4. **Adapters** generate and execute preprocessing code when source data doesn't match the template schema — with automatic retry and LLM-assisted error correction
5. **Execution** runs the configured pipeline end-to-end

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

### Direct usage (you already know your data)

If you know the table names, column names, and file paths, skip the AI layer entirely:

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

This is the primary workflow. The AI surveys your data, maps it to the template's requirements, asks about anything it can't resolve, and handles data transformation — all using exact config field names so the pipeline runs without manual wiring.

```python
from ai2analytics import AnalyticsSession

session = AnalyticsSession(
    spark=spark,
    llm_endpoint="your-llm-endpoint",  # Databricks Model Serving endpoint
)
```

#### Step 1: Discover data

The session scans your Spark catalog, profiles every table (column types, cardinality, time series characteristics), and uses the LLM to match discovered tables/columns to the template's declared requirements.

```python
session.discover(
    schemas=["commercial_data", "reference_tables"],
    prompt="Optimize HCP call allocation for Brand X",
)
```

Output:

```
DATA DISCOVERY
======================================================================

Scanning tables...
  Profiling commercial_data.physician_weekly... 2,450,000 rows, 28 cols
  Profiling commercial_data.detail_calls... 890,000 rows, 12 cols
  Profiling reference_tables.territory_map... 5,200 rows, 4 cols
  ...

Using template: detail_optimization

Analyzing data fit...

Configuration for detail_optimization
============================================================

Auto-detected:
  hcp_weekly_table: commercial_data.physician_weekly
  calls_table: commercial_data.detail_calls
  col_npi: PHYSICIAN_NPI
  col_week: WEEK_DATE
  col_referrals: REFERRAL_COUNT
  col_calls: F2F_CALLS

Still needed:
----------------------------------------

  1. [drug_name] What is the drug name? *
  2. [team_a_align_path] What is the team a align path? *
  3. [output_csv] What is the output csv? *
  ...

Call session.answer({field_name: value, ...}) to provide answers.
```

The LLM auto-detected 6 config fields from the data. It sees the template requires `col_npi` (via `-> cfg.col_npi` annotations on the schema), finds `PHYSICIAN_NPI` in the data, and maps it directly.

#### Step 2: Answer remaining questions

```python
session.answer({
    "drug_name": "BRAND_X",
    "team_a_align_path": "/dbfs/mnt/data/team_a.csv",
    "team_b_align_path": "/dbfs/mnt/data/team_b.csv",
    "hcp_reference_path": "/dbfs/mnt/data/hcp_ref.csv",
    "output_csv": "/dbfs/mnt/output/brand_x_plan.csv",
})
```

You can also set any config field directly, even ones the LLM didn't ask about:

```python
session.answer({
    "team_a_label": "SALES",
    "team_b_label": "MSL",
    "prob_model_params": {"n_estimators": 200, "max_depth": 7, "random_state": 42},
})
```

Review everything at any point:

```python
session.show_config()   # all current config values
session.show_questions() # what's answered vs. still needed
```

#### Step 3: Generate and run adapter code (if needed)

If the source data doesn't exactly match what the pipeline expects — say, column names differ, types need casting, or a reference file needs restructuring — the LLM generates transformation code:

```python
session.generate_adapter()
```

Output:

```
======================================================================
GENERATED ADAPTER CODE
======================================================================
# Rename PHYSICIAN_NPI -> npi in the weekly table for pipeline compatibility
df_weekly = spark.table("commercial_data.physician_weekly")
df_weekly = df_weekly.withColumnRenamed("PHYSICIAN_NPI", "npi")
df_weekly = df_weekly.withColumnRenamed("WEEK_DATE", "WEEK_ENDING")
df_weekly = df_weekly.withColumnRenamed("REFERRAL_COUNT", "PAT_COUNT_REFERRED")
df_weekly.createOrReplaceTempView("hcp_weekly_prepared")
print(f"Prepared weekly table: {df_weekly.count():,} rows")
...
======================================================================

Review the code above. Call session.run_adapter() to execute it.
Or edit with: session.set_adapter_code('your code')
```

Then execute it:

```python
session.run_adapter()
# Output: Adapter executed successfully.
```

If execution fails, the session automatically sends the code and traceback back to the LLM, gets a corrected version, and retries (up to 2 times by default):

```python
session.run_adapter(max_retries=3)  # allow more retry attempts
```

You can also manually edit the adapter code:

```python
session.set_adapter_code("""
# My custom preprocessing
df = spark.table("commercial_data.physician_weekly")
df = df.filter(df.REGION == "US")
df.createOrReplaceTempView("hcp_weekly_filtered")
""")
session.run_adapter()
```

#### Step 4: Run the pipeline

```python
results = session.run()
```

This builds the config from all collected answers, validates it, and runs the full pipeline (load -> features -> train -> score -> optimize -> output).

### Deep-profiling a specific table

Before or during configuration, you can deep-profile any table to understand its structure:

```python
session.profile_table("commercial_data.physician_weekly")
```

Output:

```
Deep Profile: commercial_data.physician_weekly
  Rows: 2,450,000
  Entity column: PHYSICIAN_NPI (8,200 unique)

  Time Series:
    Column: WEEK_DATE
    Frequency: weekly
    Range: 2023-01-07 to 2025-03-22 (116 periods)
    Completeness: 94.2% (has gaps)
    Avg entities/period: 7,850

  Numeric columns (18): REFERRAL_COUNT, F2F_CALLS, TRX_L12M, ...
  Categorical columns (3): REGION, SPECIALTY, INDICATION
  Flag columns (4): WRITER_FLAG, TARGET_FLAG, PRIORITY, EARLY_ADOPTER
  Stagnant columns (no time variation): SPECIALTY, REGION
```

This tells you whether the data is a time series, what the grain is, whether there are gaps, which columns change over time vs. static per entity, and how many categories exist — all of which inform whether adapter code is needed.

## How the Config Mapping Works

Each template declares its data requirements with explicit links to config fields:

```python
TableRequirement(
    key="hcp_weekly",
    description="One row per HCP x week with referral counts",
    config_field="hcp_weekly_table",  # <- tells the LLM: set cfg.hcp_weekly_table
    required_columns=[
        ColumnRequirement(
            "npi", "int", "HCP identifier",
            aliases=["NPI", "npi_number", "PHYSICIAN_NPI"],
            config_field="col_npi",   # <- tells the LLM: set cfg.col_npi
        ),
    ],
)
```

When the LLM analyzes your data, it sees:

```
  [hcp_weekly] (spark_table) -> cfg.hcp_weekly_table
    Required columns:
      - npi (int): HCP identifier (aliases: NPI, npi_number, PHYSICIAN_NPI) -> cfg.col_npi
```

Plus the full config field listing:

```
  hcp_weekly_table (str) = ''  <-- MUST BE SET
  col_npi (str) = 'npi'
  col_week (str) = 'WEEK_ENDING'
  ...
```

This gives the LLM the complete chain: "I see a table `commercial_data.physician_weekly` with column `PHYSICIAN_NPI` -> that matches `npi` alias -> I should set `cfg.hcp_weekly_table = 'commercial_data.physician_weekly'` and `cfg.col_npi = 'PHYSICIAN_NPI'`."

## Architecture

```
ai2analytics/
├── session.py               # Main orchestrator — discover, configure, adapt, run
├── llm.py                   # LLM client (Databricks Model Serving / OpenAI-compatible)
├── utils.py                 # Shared data utilities (clean_npi, yn_flag, etc.)
├── discovery/
│   ├── surveyor.py          # Scans Spark catalogs, profiles table schemas
│   └── profiler.py          # Deep analysis: time series, completeness, variable types
├── conversation/
│   └── manager.py           # LLM-driven Q&A with config-field-aware prompts
├── codegen/
│   └── adapter.py           # Generates and validates preprocessing code
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
        └── pipeline.py      # Pipeline orchestrator + requirement declarations
```

## Detail Optimization Template

The included `detail_optimization` template implements an HCP call allocation pipeline:

| Stage | Module | What it does |
|-------|--------|-------------|
| C | `loader.py` | Loads HCP weekly data, calls, alignments, reference files |
| D | `features.py` | Lags, rolling windows, cumulative sums, target construction |
| E | `models.py` | Trains probability, depth, and look-alike models with walk-forward backtesting, then retrains on full data |
| F | `scoring.py` | Scores every HCP at each call level using vectorized batch prediction |
| G-H | `optimizer.py` | Merges alignments, runs PuLP LP with territory budget constraints and slack penalties |
| I-J | `output.py` | Builds long-format portfolio, flags (priority, lookalike, call increase), writes output |

### Configuration

All column names, file paths, model parameters, and optimizer constraints are configurable via `DetailOptimizationConfig`. No hardcoded assumptions about your specific data schema:

```python
cfg = DetailOptimizationConfig(
    # Column mappings — set these to match YOUR data
    col_npi="PHYSICIAN_NPI",
    col_week="WEEK_DATE",
    col_referrals="REFERRAL_COUNT",
    col_calls="F2F_CALLS",

    # Team labels
    team_a_label="SALES",
    team_b_label="MSL",

    # Therapeutic class Rx columns for deciling
    il_rx_columns=[
        ("CLASS_A_TRX_L12M", "CLASS_A_DECILE"),
        ("CLASS_B_TRX_L12M", "CLASS_B_DECILE"),
    ],

    # Flag columns to one-hot encode
    flag_columns_to_onehot=["EARLY_ADOPTER", "PRIORITY", "STAR_TARGET"],

    # Model parameters
    prob_model_params={"n_estimators": 200, "max_depth": 7, "random_state": 42},
    retrain_on_full_data=True,  # retrain on all data after backtesting (default)

    # Optimizer constraints
    team_a_budget_per_territory=120,
    team_b_target_per_territory=80,
    max_calls_nonpriority=4,
)
```

## Adding New Templates

Create a new template by subclassing `BaseTemplate`. Declare your data requirements with `config_field` links so the AI layer can wire data to config automatically:

```python
from dataclasses import dataclass
from ai2analytics.templates.base import BaseTemplate, TableRequirement, ColumnRequirement
from ai2analytics.templates.registry import register

@dataclass
class MyConfig:
    main_table: str = ""
    col_id: str = "id"
    col_date: str = "date"
    output_path: str = ""

    def validate(self):
        errors = []
        if not self.main_table:
            errors.append("main_table is required")
        return errors

@register
class MyPipeline(BaseTemplate):
    name = "my_pipeline"
    description = "Description for template matching"
    config_class = MyConfig

    required_tables = [
        TableRequirement(
            key="main_table",
            description="Primary data source",
            config_field="main_table",
            required_columns=[
                ColumnRequirement("id", "int", "Entity identifier",
                                  config_field="col_id"),
                ColumnRequirement("date", "date", "Time period",
                                  config_field="col_date"),
            ],
        ),
    ]

    def run(self, config, spark=None):
        # Your pipeline logic here
        ...
```

The discovery layer will automatically:
- Match available tables against your `required_tables`
- Use `config_field` to map discovered columns to exact config fields
- Generate questions for any config fields with `""` defaults that weren't auto-detected
- Produce adapter code if column names or types don't match

## Session API Reference

| Method | Description |
|--------|-------------|
| `discover(schemas, prompt)` | Scan tables, match template, auto-fill config, generate questions |
| `show_questions()` | Display what's been auto-detected vs. what still needs answers |
| `show_config()` | Show all current config field values |
| `answer({field: value})` | Provide answers — accepts any config field name |
| `profile_table(name)` | Deep-profile a specific table (time series, completeness, types) |
| `generate_adapter()` | LLM generates preprocessing code for data mismatches |
| `set_adapter_code(code)` | Manually set or edit the adapter code |
| `run_adapter(max_retries=2)` | Execute adapter code with `spark` in scope; auto-fixes on failure |
| `build_config()` | Build and validate the config object from collected answers |
| `run(config=None)` | Run the pipeline (builds config from answers if not provided) |
| `select_template(name)` | Manually select a template by name |

## Requirements

- Python >= 3.9
- pandas, numpy, scikit-learn, PuLP, matplotlib
- Optional: databricks-sdk, pyspark (for Databricks environments)
- An LLM endpoint (Databricks Model Serving or any OpenAI-compatible API) for the AI-assisted workflow

## License

MIT
