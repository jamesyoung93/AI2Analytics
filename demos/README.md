# Synthetic Data Generators

Three generators produce realistic fake pharma datasets for testing
`ai2analytics` pipelines without access to real patient or prescriber data.
All identifiers are randomly generated and contain no real-world PII.

## Generators

| Generator | Market style | Cadence | Entity level | Default seed |
|---|---|---|---|---|
| `synthetic_us_hcp.py` | US | Weekly | HCP (NPI) | 42 |
| `synthetic_eu_account.py` | EU | Monthly | Account (ACC-XXXXX) | 43 |
| `synthetic_bric.py` | BRIC | Quarterly | Account (numeric ID) | 44 |

---

## US HCP (`synthetic_us_hcp.py`)

### Tables and columns

**hcp_reference** -- one row per HCP

| Column | Type | Description |
|---|---|---|
| `npi` | int | Random 10-digit integer (not a real NPI) |
| `WRITER_FLAG` | str | `Y` / `N` -- has the HCP written prescriptions |
| `TARGET_FLAG` | str | `Y` / `N` -- is the HCP a promotional target |
| `IL_17_TRX_L12M` | int | IL-17 therapeutic class Rx count (last 12 months) |
| `IL_23_TRX_L12M` | int | IL-23 therapeutic class Rx count (last 12 months) |

**hcp_weekly** -- one row per HCP per observed week

| Column | Type | Description |
|---|---|---|
| `npi` | int | Matches `hcp_reference.npi` |
| `WEEK_ENDING` | date | Friday date |
| `INDC` | str | Indication (`IND_A` / `IND_B` / `IND_C` / `IND_D`) |
| `PAT_COUNT_REFERRED` | int | Patient referrals (zero-inflated Poisson) |
| `TARGET_FLAG` | str | `Y` / `N` |
| `SYMPTOM_SEVERITY_SCORE` | float | Log-normal severity score |
| `PRIOR_AUTH_COUNT` | int | Prior authorisation submissions |
| `SWITCH_FLAG_NUM` | int | 0 / 1 -- therapy switch indicator |

**calls** -- sparse face-to-face call records

| Column | Type | Description |
|---|---|---|
| `NPI` | int | Matches `hcp_reference.npi` |
| `WEEK_ENDING` | date | Friday date |
| `HCP_F2F_CALLS` | int | Number of face-to-face calls (negative binomial) |

**team_a_alignment / team_b_alignment** -- territory mapping

| Column | Type | Description |
|---|---|---|
| `HCP_NPI` | int | Matches `hcp_reference.npi` |
| `TERRITORY_ID` | int | Territory identifier |

**portfolio_decile** -- portfolio drug volume decile

| Column | Type | Description |
|---|---|---|
| `npi` | int | Matches `hcp_reference.npi` |
| `PORTFOLIO_UNITS_DECILE` | int | 1-10 (correlated with writer status) |

**priority_targets** -- high-value writer subset

| Column | Type | Description |
|---|---|---|
| `npi` | int | Matches `hcp_reference.npi` |
| `PRIORITY_TARGET` | str | Always `Y` |

---

## EU Account (`synthetic_eu_account.py`)

### Tables and columns

**account_reference** -- one row per account

| Column | Type | Description |
|---|---|---|
| `PRESCRIBER_ID` | str | `ACC-XXXXX` format |
| `ACCOUNT_NAME` | str | Synthetic institution name |
| `REGION` | str | `UK` / `DE` / `FR` / `IT` / `ES` / `NL` |
| `SEGMENT` | str | `A` / `B` / `C` |
| `TIER` | int | 1-4 (1 = highest value) |
| `UNITS_SOLD_L12M` | int | Last-12-month unit volume |

**account_monthly** -- one row per account per observed month

| Column | Type | Description |
|---|---|---|
| `PRESCRIBER_ID` | str | Matches `account_reference.PRESCRIBER_ID` |
| `MONTH_END` | date | Month-end date |
| `UNITS_SOLD` | int | Units sold that month |
| `NEW_PATIENTS` | int | New patient starts |
| `MARKET_SHARE` | float | 0-1 market share estimate |

**visits** -- multi-channel engagement (sparse)

| Column | Type | Description |
|---|---|---|
| `PRESCRIBER_ID` | str | Matches `account_reference.PRESCRIBER_ID` |
| `MONTH_END` | date | Month-end date |
| `KAM_VISITS` | int | Key Account Manager visits |
| `MEDICAL_VISITS` | int | Medical Science Liaison visits |
| `DIGITAL_CONTACTS` | int | Digital contacts (emails, webinars) |

**kam_alignment** -- KAM territory mapping

| Column | Type | Description |
|---|---|---|
| `PRESCRIBER_ID` | str | Matches `account_reference.PRESCRIBER_ID` |
| `KAM_TERRITORY_ID` | int | KAM territory identifier |

**medical_alignment** -- MSL territory mapping

| Column | Type | Description |
|---|---|---|
| `PRESCRIBER_ID` | str | Matches `account_reference.PRESCRIBER_ID` |
| `MEDICAL_TERRITORY_ID` | int | Medical territory identifier |

---

## BRIC (`synthetic_bric.py`)

### Tables and columns

**account_master** -- one row per account

| Column | Type | Description |
|---|---|---|
| `ACCOUNT_ID` | int | Numeric identifier |
| `COUNTRY_CODE` | str | `BR` / `RU` / `IN` / `CN` |
| `CITY` | str | City name |
| `INSTITUTION_TYPE` | str | `hospital` / `clinic` / `pharmacy` |
| `CHANNEL` | str | `retail` / `hospital` |
| `CUSTOMER_SEGMENT` | str | `A` / `B` / `C` / `D` |

**quarterly_performance** -- one row per account per observed quarter

| Column | Type | Description |
|---|---|---|
| `ACCOUNT_ID` | int | Matches `account_master.ACCOUNT_ID` |
| `QUARTER_END` | date | Quarter-end date |
| `REVENUE_LOCAL` | float | Revenue in local currency |
| `UNITS_DISPENSED` | int | Units dispensed |
| `PATIENT_STARTS` | int | New patient starts |
| `COMPLIANCE_RATE` | float | 0-1 compliance rate (beta-distributed) |

**engagement** -- multi-channel engagement

| Column | Type | Description |
|---|---|---|
| `ACCOUNT_ID` | int | Matches `account_master.ACCOUNT_ID` |
| `QUARTER_END` | date | Quarter-end date |
| `REP_VISITS` | int | Sales rep visits (negative binomial) |
| `CONGRESS_ATTENDANCE` | int | 0 / 1 -- attended a congress |
| `SAMPLE_UNITS` | int | Sample units distributed |
| `DIGITAL_ENGAGEMENT_SCORE` | int | 0-100 score (log-normal, right-skewed) |

**sales_alignment** -- sales territory mapping

| Column | Type | Description |
|---|---|---|
| `ACCOUNT_ID` | int | Matches `account_master.ACCOUNT_ID` |
| `SALES_TERRITORY_ID` | int | Territory identifier |
| `COUNTRY_CODE` | str | Country code |

---

## Running a generator

### Command line

```bash
# Defaults: writes CSVs to demos/data/<market>/
python demos/synthetic_us_hcp.py
python demos/synthetic_eu_account.py
python demos/synthetic_bric.py

# Custom output directory
python demos/synthetic_us_hcp.py --output-dir /tmp/us_demo --seed 99
```

### From Python

```python
from demos.synthetic_us_hcp import generate_all

# In-memory only (no files written)
tables = generate_all()
print(tables["hcp_weekly"].head())

# Write to disk
tables = generate_all(output_dir="demos/data/us_hcp")
```

---

## Using with the detail-optimization pipeline

The US HCP generator's output maps directly to the `dataframes` dict
accepted by the pipeline loader. This lets you run the full pipeline
without Spark or CSV files on disk:

```python
from demos.synthetic_us_hcp import generate_all
from ai2analytics.templates.detail_optimization.config import DetailOptimizationConfig
from ai2analytics.templates.detail_optimization.pipeline import run_pipeline

tables = generate_all()

cfg = DetailOptimizationConfig(
    drug_name="SYNTH_DRUG",
    drug_portfolio="SYNTH_PORTFOLIO",
    il_rx_columns=[
        ("IL_17_TRX_L12M", "IL_17_DECILE"),
        ("IL_23_TRX_L12M", "IL_23_DECILE"),
    ],
    output_csv="output/synth_results.csv",
)

run_pipeline(
    cfg,
    dataframes={
        "hcp_reference":    tables["hcp_reference"],
        "hcp_weekly":       tables["hcp_weekly"],
        "calls":            tables["calls"],
        "team_a_align":     tables["team_a_alignment"],
        "team_b_align":     tables["team_b_alignment"],
        "portfolio_decile": tables["portfolio_decile"],
        "priority_targets": tables["priority_targets"],
    },
)
```

---

## Data realism notes

- All generators use `numpy.random.default_rng(seed)` for reproducibility.
- Referral and call counts use zero-inflated Poisson / negative binomial
  distributions to produce realistic right-skewed, zero-heavy data.
- Writer / target status correlates with Rx volumes (not independent).
- Seasonal patterns are applied to time-series columns.
- Not every entity appears in every period (realistic observation sparsity).
- Territory alignments deliberately leave a fraction of entities uncovered.
- No real-world PII: NPIs are random 10-digit ints, account names are
  synthetic combinations of institution prefixes and city names.
